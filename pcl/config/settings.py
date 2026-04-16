import os

# ---------------------------------------------------------
# 1. 默认环境变量 (作为兜底选项)
# ---------------------------------------------------------
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "sk-6d3985bd9aec4d7088e77f66750d0b71")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")

# ---------------------------------------------------------
# 2. 异构多智能体独立配置 (Heterogeneous Agent Config)
# ---------------------------------------------------------
AGENT_CONFIG = {
    # 认知专家：需要强逻辑推理，使用大模型
    "cognitive": {
        "model_name": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "api_key": "sk-6d3985bd9aec4d7088e77f66750d0b71",
        "temperature": 0.1
    },
    # 情感专家：模式识别为主，可使用更便宜的模型或本地微调模型
    "affective": {
        "model_name": "deepseek-chat",
        "base_url": "https://api.deepseek.com", # 支持完全不同的 URL
        "api_key": "sk-6d3985bd9aec4d7088e77f66750d0b71",
        "temperature": 0.2
    },
    # 社交专家：同上，也可指向本地部署的 Llama-3 (例如通过 vLLM)
    "social": {
        "model_name": "deepseek-chat",
        "base_url": "https://api.deepseek.com", 
        "api_key": "sk-6d3985bd9aec4d7088e77f66750d0b71",
        "temperature": 0.1
    },
    # 融合节点：负责揉合多句话，需要极强的语言润色能力
    "fusion": {
        "model_name": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "api_key": "sk-6d3985bd9aec4d7088e77f66750d0b71",
        "temperature": 0.3
    }
}

# ---------------------------------------------------------
# 3. 业务逻辑阈值配置
# ---------------------------------------------------------
# 最小干预决策阈值 (紧急度低于此值，保持静默)
INTERVENTION_THRESHOLD = 0.6  

# 优先级覆盖阈值 (情感紧急度超过此值，压制认知和社交)
AFFECTIVE_SUPPRESSION_THRESHOLD = 0.85