from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from typing_extensions import TypedDict

# --- 1. 记忆画像 (Memory Profile) ---
class StudentProfile(BaseModel):
    student_id: str
    cog_history: str = "良好"
    soc_history: str = "活跃"
    aff_history: str = "情绪稳定"

# --- 2. 提案结构 (Agent Proposal) ---
class AgentProposal(BaseModel):
    has_issue: bool = Field(description="是否检测到该维度的问题")
    urgency: float = Field(description="紧急度 (0.0 到 1.0)")
    draft: str = Field(description="基于学生记忆生成的干预草稿（一句话）")
    reasoning: str = Field(description="给出此提案的内部思考过程")

# --- 3. LangGraph 状态流转字典 (Graph State) ---
class CSCLState(TypedDict):
    # 输入层
    dialog_window: List[Dict[str, str]] # 最近对话记录
    student_profiles: Dict[str, StudentProfile] # 参与者的长期记忆画像
    
    # 竞标层 (并发写入)
    cog_proposal: Optional[AgentProposal]
    aff_proposal: Optional[AgentProposal]
    soc_proposal: Optional[AgentProposal]
    
    # 决断层
    decision_type: str # 'hold', 'single', 'suppress', 'fuse'
    selected_drafts: List[str] # 决断后保留的草稿
    
    # 输出层
    final_intervention: str # 发送给学生的最终话术