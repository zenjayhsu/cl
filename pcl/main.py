import os
import pandas as pd
from core.graph import build_cscl_graph
from core.memory_manager import MemoryManager

def run_dataset_simulation(data_folder_path: str, window_size: int = 5):
    """
    基于真实数据集的仿真流水线
    :param data_folder_path: 包含多个 Excel 文件的文件夹路径
    :param window_size: 喂给大模型的历史上下文窗口大小
    """
    print(f"📂 正在加载数据集文件夹: {data_folder_path}...")

    # 批量读取文件夹中的所有 Excel 文件
    all_files = [os.path.join(data_folder_path, f) for f in os.listdir(data_folder_path) if f.endswith('.xlsx')]
    if not all_files:
        print("未找到任何 Excel 文件，请检查文件夹路径。")
        return

    try:
        # 合并所有 Excel 文件的数据
        dataframes = [pd.read_excel(file) for file in all_files]
        df = pd.concat(dataframes, ignore_index=True)
        # 提取真实的发言人和发言内容
        raw_messages = df[['Origin', 'Content']].dropna().to_dict('records')
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    mas_app = build_cscl_graph()
    memory_db = MemoryManager()

    # 提取出现过的所有学生，初始化画像
    all_students = list(set([msg['Origin'] for msg in raw_messages]))
    profiles = memory_db.get_profiles(all_students)

    dialog_history = [] # 滑动窗口

    print("\n🚀 开始实时流式监听仿真...\n" + "="*50)

    # 核心：逐行遍历 Excel 数据，每一行都是一次“最新发言”
    for idx, row in enumerate(raw_messages):
        current_speaker = row['Origin']
        current_content = row['Content']

        current_utterance = {"speaker": current_speaker, "content": current_content}
        print(f"\n[{idx+1}] 🆕 最新发言 -> {current_speaker}: {current_content}")

        # 组装状态：历史窗口 + 最新发言
        initial_state = {
            "dialog_history": dialog_history.copy(), # 仅传入历史
            "current_utterance": current_utterance,  # 传入当前最新
            "student_profiles": profiles
        }

        # 触发多智能体并发监听与决断
        result_state = mas_app.invoke(initial_state)

        # 打印日志 (观测 3 个 Agent 是否捕捉到问题)
        cog_urgency = result_state['cog_proposal'].urgency
        aff_urgency = result_state['aff_proposal'].urgency
        soc_urgency = result_state['soc_proposal'].urgency

        # 只有在有智能体想要干预时，才打印内部日志，避免刷屏
        if max(cog_urgency, aff_urgency, soc_urgency) > 0.0:
            print("  ↳ 🔍 [并发监听警报]:")
            if cog_urgency > 0: print(f"    - 🧠 认知竞标: 紧急度 {cog_urgency} | 理由: {result_state['cog_proposal'].reasoning}")
            if aff_urgency > 0: print(f"    - ❤️ 情感竞标: 紧急度 {aff_urgency} | 理由: {result_state['aff_proposal'].reasoning}")
            if soc_urgency > 0: print(f"    - 🤝 社交竞标: 紧急度 {soc_urgency} | 理由: {result_state['soc_proposal'].reasoning}")

            print(f"  ↳ ⚙️ [Meta-Agent 决断]: {result_state['decision_type'].upper()}")

        # 如果 Meta-Agent 决定输出最终话术
        if result_state['final_intervention']:
            print(f"  ↳ 🤖 [AI 助教干预]: \033[92m{result_state['final_intervention']}\033[0m")
            # 将 AI 的发言也加入历史记录
            dialog_history.append({"speaker": "AI_Tutor", "content": result_state['final_intervention']})

        # 收尾：将当前发言加入历史，并维持滑动窗口大小
        dialog_history.append(current_utterance)
        if len(dialog_history) > window_size:
            dialog_history = dialog_history[-window_size:]

if __name__ == "__main__":
    # 替换为你实际的文件夹路径
    run_dataset_simulation("data/post-annotated Delidata")