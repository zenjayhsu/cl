import os
import glob
import json
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# ==========================================
# 第一部分：数据预处理 (CSV -> JSONL 格式)
# ==========================================
def prepare_sft_dataset(data_dir):
    print("开始处理 Excel 数据...")
    # 【修改1】查找所有的 .xlsx 文件，而不是 .csv
    all_excel_files = glob.glob(os.path.join(data_dir, "*.xlsx")) 
    
    formatted_data = []
    
    for file in all_excel_files:
        # 【修改2】使用 read_excel 直接读取 Excel 文件
        # 注意：本地运行此代码需要额外安装 openpyxl 库 (pip install openpyxl)
        df = pd.read_excel(file) 
        
        # 按照对话顺序重构历史
        history = []
        for index, row in df.iterrows():
            speaker = str(row['Origin'])
            content = str(row['Content'])
            
            # 只有当模型确实做出了干预判断时，我们才把它作为一条训练数据
            if pd.notna(row['intervention']):
                # 提取目标标签，处理 NaN 或 0 的情况
                stage = str(row['identified stage']) if pd.notna(row['identified stage']) and row['identified stage'] != '0' else "None"
                issue = str(row['issue']) if pd.notna(row['issue']) and row['issue'] != '0' else "None"
                intervention = str(row['intervention']).strip().lower()
                feedback = str(row['guidance']) if pd.notna(row['guidance']) and row['guidance'] != '0' else "None"
                
                # 组装我们期望模型输出的 JSON 格式
                target_dict = {
                    "Stage": stage,
                    "Issue": issue,
                    "Intervene": "Yes" if intervention == 'yes' else "No",
                    "Feedback Rule": feedback
                }
                target_json = json.dumps(target_dict, ensure_ascii=False)
                
                # 组装给模型的 Prompt (包含到目前为止的对话历史)
                formatted_history = "\n".join(history)
                prompt_text = f"以下是学生小组的讨论历史：\n{formatted_history}\n\n请根据探究社区(CoI)理论，诊断当前讨论的Stage(阶段)、Issue(问题)，并决定是否需要Intervene(干预)。如果需要，请提供Feedback Rule(反馈)。"
                
                # 使用 Llama-3 的官方 Chat Template 特殊 token 组装文本
                # 这告诉模型：什么是系统指令，什么是用户输入，什么是标准答案
                full_text = (
                    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                    "你是一个专业的协作学习辅导智能体。请严格以JSON格式输出诊断结果。<|eot_id|>"
                    "<|start_header_id|>user<|end_header_id|>\n"
                    f"{prompt_text}<|eot_id|>"
                    "<|start_header_id|>assistant<|end_header_id|>\n"
                    f"{target_json}<|eot_id|>"
                )
                
                formatted_data.append({"text": full_text})
            
            # 将当前发言加入历史，供下一轮使用
            history.append(f"{speaker}: {content}")
            
    print(f"数据处理完成！共提取出 {len(formatted_data)} 条高质量对话轮次。")
    return Dataset.from_list(formatted_data)


# ==========================================
# 第二部分：SFT 监督微调 (QLoRA 训练模型)
# ==========================================
def train_sft_model(dataset):
    print("开始加载模型与 Tokenizer...")
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    # 1. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 以 4-bit 量化模式加载模型（这能把 8B 模型的显存占用从 16G 压到约 6-8G）
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        load_in_4bit=True, 
        device_map="auto"
    )
    
    # 3. 配置 LoRA（低秩自适应微调）
    # 论文中推荐我们微调注意力机制中的 q, v, k, o 矩阵
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 4. 配置训练参数 (按照原论文设置)
    training_args = TrainingArguments(
        output_dir="./CL-8B-SFT-Checkpoints",
        per_device_train_batch_size=2,          # 显存不够可以改成 1
        gradient_accumulation_steps=4,          # 变相扩大 Batch Size
        learning_rate=5e-5,                     # 论文中指定的学习率
        num_train_epochs=3,                     # 论文中指定的训练轮数
        bf16=True,                              # 使用 bf16 精度加速训练
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_32bit",              # 节省显存的优化器
    )
    
    # 5. 启动 SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",              # 指定数据集中包含合并文本的列名
        peft_config=lora_config,
        max_seq_length=2048,                    # 控制上下文长度，防止显存溢出
        args=training_args,
    )
    
    print("启动训练引擎！")
    trainer.train()
    
    # 6. 保存最终的 LoRA 权重
    final_save_path = "./CL-8B-SFT-Final"
    trainer.model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"训练完成！模型已保存至 {final_save_path}")


if __name__ == "__main__":
    # 替换成你电脑上 post-annotated Delidata 文件夹的实际路径
    # 例如：data_directory = "./CLTeach/post-annotated Delidata"
    data_directory = r"C:\Users\52771\Desktop\Collaborative-Agents-main\CLTeach\post-annotated Delidata"
    # 1. 执行预处理
    sft_dataset = prepare_sft_dataset(data_directory)
    
    # 2. 执行微调
    train_sft_model(sft_dataset)