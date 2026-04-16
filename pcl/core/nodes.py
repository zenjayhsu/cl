from langchain_openai import ChatOpenAI
from schemas.models import CSCLState, AgentProposal
from prompts.expert_prompts import COG_PROMPT, AFF_PROMPT, SOC_PROMPT
from prompts.fusion_prompts import FUSION_PROMPT
from config.settings import AGENT_CONFIG, INTERVENTION_THRESHOLD, AFFECTIVE_SUPPRESSION_THRESHOLD

# ==========================================
# 模型工厂：根据配置动态创建不同的 LLM 实例
# ==========================================
def create_llm(agent_role: str):
    """根据 settings 中的配置，实例化特定的 LLM"""
    cfg = AGENT_CONFIG.get(agent_role, AGENT_CONFIG["fusion"]) # 默认 fallback
    return ChatOpenAI(
        model=cfg["model_name"],
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        temperature=cfg["temperature"],
        max_retries=2
    )

# 初始化各专家的独立 LLM 并绑定结构化输出 (AgentProposal)
cog_llm = create_llm("cognitive").with_structured_output(AgentProposal)
aff_llm = create_llm("affective").with_structured_output(AgentProposal)
soc_llm = create_llm("social").with_structured_output(AgentProposal)

# 融合节点不需要结构化输出，直接返回文本即可
fusion_llm = create_llm("fusion")

# ==========================================
# 辅助函数
# ==========================================
def format_history(history: list) -> str:
    if not history: return "（无历史记录，这是第一句话）"
    return "\n".join([f"[{msg.get('speaker', 'Unknown')}]: {msg.get('content', '')}" for msg in history])

# ==========================================
# 1. 并发监听层 (Parallel Listeners)
# ==========================================
def cognitive_node(state: CSCLState) -> dict:
    history = state.get('dialog_history', [])
    current = state.get('current_utterance', {})
    profiles = state.get('student_profiles', {})
    
    proposal = cog_llm.invoke(COG_PROMPT.format_messages(
        expert_role="认知专家", 
        memory_profiles=profiles, 
        dialog_history=format_history(history),
        current_speaker=current.get('speaker', 'Unknown'),
        current_content=current.get('content', '')
    ))
    return {"cog_proposal": proposal}

def affective_node(state: CSCLState) -> dict:
    history = state.get('dialog_history', [])
    current = state.get('current_utterance', {})
    profiles = state.get('student_profiles', {})
    
    proposal = aff_llm.invoke(AFF_PROMPT.format_messages(
        expert_role="情感专家", 
        memory_profiles=profiles, 
        dialog_history=format_history(history),
        current_speaker=current.get('speaker', 'Unknown'),
        current_content=current.get('content', '')
    ))
    return {"aff_proposal": proposal}

def social_node(state: CSCLState) -> dict:
    history = state.get('dialog_history', [])
    current = state.get('current_utterance', {})
    profiles = state.get('student_profiles', {})
    
    proposal = soc_llm.invoke(SOC_PROMPT.format_messages(
        expert_role="社交专家", 
        memory_profiles=profiles, 
        dialog_history=format_history(history),
        current_speaker=current.get('speaker', 'Unknown'),
        current_content=current.get('content', '')
    ))
    return {"soc_proposal": proposal}

# ==========================================
# 2. 决断层 (Meta-Agent Decision)
# ==========================================
def meta_decision_node(state: CSCLState) -> dict:
    proposals = {
        "cog": state.get("cog_proposal"),
        "aff": state.get("aff_proposal"),
        "soc": state.get("soc_proposal")
    }
    
    active_proposals = {
        k: v for k, v in proposals.items() 
        if v is not None and v.has_issue and v.urgency >= INTERVENTION_THRESHOLD
    }
    
    decision_type = "hold"
    selected_drafts = []
    
    if not active_proposals:
        decision_type = "hold"
    elif len(active_proposals) == 1:
        decision_type = "single"
        selected_drafts.append(list(active_proposals.values())[0].draft)
    else:
        if "aff" in active_proposals and active_proposals["aff"].urgency >= AFFECTIVE_SUPPRESSION_THRESHOLD:
            decision_type = "suppress"
            selected_drafts.append(active_proposals["aff"].draft)
        else:
            decision_type = "fuse"
            selected_drafts = [p.draft for p in active_proposals.values() if p.draft]
            
    return {"decision_type": decision_type, "selected_drafts": selected_drafts}

# ==========================================
# 3. 融合与输出层 (Fusion Generation)
# ==========================================
def fusion_node(state: CSCLState) -> dict:
    decision = state.get("decision_type", "hold")
    drafts = state.get("selected_drafts", [])
    
    if decision == "hold" or not drafts:
        return {"final_intervention": ""}
        
    elif decision in ["single", "suppress"]:
        return {"final_intervention": drafts[0]}
        
    elif decision == "fuse":
        drafts_text = "\n".join([f"- {d}" for d in drafts])
        # 使用独立的 fusion_llm 进行合并润色
        fusion_response = fusion_llm.invoke(FUSION_PROMPT.format_messages(drafts=drafts_text))
        return {"final_intervention": fusion_response.content}
    
    return {"final_intervention": ""}