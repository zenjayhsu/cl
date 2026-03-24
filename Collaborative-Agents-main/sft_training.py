import os
import glob
import json
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login

# ==========================================
# 🛑 必填项：Hugging Face 授权 Token 
# ==========================================
# 请将你在第一步获取的以 hf_ 开头的 Token 粘贴在下面引号里
HF_TOKEN = "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" 

# ==========================================
# 数据预处理模块
# ==========================================
def prepare_sft_dataset(data_dir):
    print(f"正在扫描数据文件夹: {data_dir}")
    
    # 兼容搜索 xlsx 和 csv 格式
    all_files = glob.glob(os.path.join(data_dir, "*.xlsx")) + glob.glob(os.path.join(data_dir, "*.csv"))
    print(f"🔎 系统找到了 {len(all_files)} 个数据文件！")
    
    if len(all_files) == 0:
        print("❌ 错误：没有找到任何文件！请绝对确认底部的 data_directory 路径是对的。")
        return None

    formatted_data = []
    
    for file in all_files:
        try:
            if file.endswith('.xlsx'):
                df = pd.read_excel(file)
            else:
                df = pd.read_csv(file)
        except Exception as e:
            print(f"读取文件 {file} 时出错: {e}")
            continue
            
        history = []
        for index, row in df.iterrows():
            # 兼容列名的大小写问题
            col_origin = row.get('Origin', row.get('origin', 'Unknown'))
            col_content = row.get('Content', row.get('content', ''))
            col_intervention = row.get('intervention')
            
            speaker = str(col_origin)
            content = str(col_content)
            
            if pd.notna(col_intervention) and str(col_intervention).strip() != '':
                stage = str(row.get('identified stage', 'None'))
                issue = str(row.get('issue', 'None'))
                intervention = str(col_intervention).strip().lower()
                guidance = str(row.get('guidance', 'None'))
                
                if stage != '0' and issue != '0':
                    target_dict = {
                        "Stage": stage,
                        "Issue": issue,
                        "Intervene": "Yes" if intervention == 'yes' else "No",
                        "Feedback Rule": guidance
                    }
                    target_json = json.dumps(target_dict, ensure_ascii=False)
                    
                    formatted_history = "\n".join(history)
                    prompt_text = f"以下是学生小组的讨论历史：\n{formatted_history}\n\n请根据探究社区(CoI)理论，诊断当前讨论的Stage(阶段)、Issue(问题)，并决定是否需要Intervene(干预)。如果需要，请提供Feedback Rule(反馈)。"
                    
                    full_text = (
                        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                        "你是一个专业的协作学习辅导智能体。请严格以JSON格式输出诊断结果。<|eot_id|>"
                        "<|start_header_id|>user<|end_header_id|>\n"
                        f"{prompt_text}<|eot_id|>"
                        "<|start_header_id|>assistant<|end_header_id|>\n"
                        f"{target_json}<|eot_id|>"
                    )
                    formatted_data.append({"text": full_text})
            
            history.append(f"{speaker}: {content}")
            
    print(f"✅ 数据处理完成！成功提取出 {len(formatted_data)} 条高质量训练数据。")
    if len(formatted_data) == 0:
        return None
    return Dataset.from_list(formatted_data)

# ==========================================
# 模型微调模块
# ==========================================
def train_sft_model(dataset):
    print("🚀 开始通过 Token 登录 Hugging Face...")
    login(token=HF_TOKEN)
    
    # 使用 Qwen2.5-7B
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    print(f"📥 正在拉取模型: {model_id} ...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 🌟 核心修改 1：使用最新的 NF4 4-bit 量化配置
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=quant_config,  # 替换原来的 load_in_4bit=True
        device_map="auto",
        token=HF_TOKEN
    )
    
    # 启用梯度检查点 (必须先在模型上开启)，这是 8GB/12GB 显卡救命的神器！
    model.gradient_checkpointing_enable()
    
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
# 🌟 核心修改 2：把 TrainingArguments 换成 SFTConfig，并把参数搬进来
# 🌟 核心修改：把 max_seq_length 移出去
    training_args = SFTConfig(
    output_dir="./CL-8B-SFT-Checkpoints",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    num_train_epochs=3,

    bf16=True,   # ✅ 开启 BF16
    fp16=False,  # ❌ 关闭 FP16

    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_32bit",
    gradient_checkpointing=True,
    max_grad_norm=0.3,
    dataset_text_field="text",
)
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        args=training_args,
    )
    
    print("🔥 启动微调训练！RTX 5070 引擎点火！")
    trainer.train()
    
    final_save_path = "./CL-8B-SFT-Final"
    trainer.model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"🎉 训练大功告成！模型已保存至 {final_save_path}")

if __name__ == "__main__":
    # ==========================================
    # 🛑 必填项：检查你的数据路径！
    # ==========================================
    # 请确保路径里不要有中文字符，并且使用绝对路径，路径最前面保留 r
    data_directory = r"C:\Users\52771\Desktop\Collaborative-Agents-main\CLTeach\CLTeach\post-annotated Delidata"
    
    sft_dataset = prepare_sft_dataset(data_dir=data_directory)
    
    # 增加强力拦截逻辑：只有成功读到数据，才去下载几十G的模型，避免浪费时间
    if sft_dataset is not None and len(sft_dataset) > 0:
        train_sft_model(sft_dataset)
    else:
        print("⛔ 训练被中止，因为没有提取到任何有效训练数据，请检查数据文件夹路径！")