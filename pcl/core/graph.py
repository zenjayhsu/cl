from langgraph.graph import StateGraph, START, END
from schemas.models import CSCLState
from core.nodes import cognitive_node, affective_node, social_node, meta_decision_node, fusion_node

def build_cscl_graph():
    workflow = StateGraph(CSCLState)
    
    # 添加节点
    workflow.add_node("CognitiveAgent", cognitive_node)
    workflow.add_node("AffectiveAgent", affective_node)
    workflow.add_node("SocialAgent", social_node)
    workflow.add_node("MetaDecision", meta_decision_node)
    workflow.add_node("FusionGenerator", fusion_node)
    
    # 核心创新：并发监听 (Parallel Fan-out)
    # START 同时连接三个 Agent，LangGraph 会自动并发执行它们！
    workflow.add_edge(START, "CognitiveAgent")
    workflow.add_edge(START, "AffectiveAgent")
    workflow.add_edge(START, "SocialAgent")
    
    # 汇聚到决断层 (Fan-in)
    workflow.add_edge("CognitiveAgent", "MetaDecision")
    workflow.add_edge("AffectiveAgent", "MetaDecision")
    workflow.add_edge("SocialAgent", "MetaDecision")
    
    # 决断后走向融合或输出
    workflow.add_edge("MetaDecision", "FusionGenerator")
    workflow.add_edge("FusionGenerator", END)
    
    return workflow.compile()