from langchain_core.prompts import ChatPromptTemplate

# 基础模板：明确区分历史与最新发言
BASE_SYSTEM_TMPL = """
你是一个CSCL协作学习中的【{expert_role}】。
当前讨论的学生长期记忆画像如下：
{memory_profiles}

【历史对话上下文（按时间顺序）】：
{dialog_history}

========================
【请重点评估以下最新发言】：
发言人：{current_speaker}
发言内容：{current_content}
========================
"""

COG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", BASE_SYSTEM_TMPL + """
    任务：判断【最新发言】是否存在认知障碍（如概念错误、表面加工、偏题）。
    注意：必须结合上下文，不要断章取义。
    如果发现问题，请结合该学生的认知历史，生成一句苏格拉底式的引导草稿。务必评估紧急度（0.0-1.0）。
    """),
])

AFF_PROMPT = ChatPromptTemplate.from_messages([
    ("system", BASE_SYSTEM_TMPL + """
    任务：判断【最新发言】是否流露出情感障碍（如挫败、焦虑、无聊）。
    注意：关注语气词和表达方式。
    如果发现负面情绪，请结合情感历史，生成一句安抚草稿。务必评估紧急度（0.0-1.0）。
    """),
])

SOC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", BASE_SYSTEM_TMPL + """
    任务：判断【最新发言】是否破坏了社交动力学（如无视他人、打断、平行对话），或者是否有人长期未发言（可参考历史记录）。
    如果发现问题，请结合社交历史，生成一句促动草稿。务必评估紧急度（0.0-1.0）。
    """),
])