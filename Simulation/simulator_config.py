"""
Simulator Configuration
配置文件：定义模型、indicator 映射、radicality 等级等
"""

# ============ 模型配置 ============
MODEL_CONFIG = {
    # 选项 1: Mistral-7B (推荐,速度快,质量好)
    "model_name": "mistralai/Mistral-7B-Instruct-v0.1",
    
    # 选项 2: Llama-2-13B (更大,质量更好,推理较慢)
    # "model_name": "meta-llama/Llama-2-13b-chat-hf",
    
    # 选项 3: Llama-2-70B (最大,需要量化)
    # "model_name": "meta-llama/Llama-2-70b-chat-hf",
}

# ============ 生成参数 ============
GENERATION_PARAMS = {
    "max_length": 100,           # 生成的最大新 token 数
    "temperature": 0.9,          # 创意度 (0.1-1.5，避免极端值)
    "top_p": 0.92,              # nucleus sampling (0-1)
    "top_k": 40,                # top-k filtering
    "do_sample": True,          # 使用采样而非贪心
    "repetition_penalty": 1.15, # 防止重复 (1.0-2.0)
}

# ============ Radicality 等级定义 ============
RADICALITY_LEVELS = {
    "Neutral": {
        "description": "General references on one's personal life, general knowledge, other individuals, events, viewpoints, organisations, places that are not radical in nature.",
        "keywords": ["personal", "general", "ordinary", "everyday", "non-political"],
    },
    "Low": {
        "description": "Content reflecting awareness or exposure to radical ideologies, groups, terminology, locations, or materials - without evidence of personal involvement, affiliation, internalisation, or support for violence.",
        "keywords": ["aware", "exposure", "mention", "reference", "knowledge", "information"],
    },
    "Medium": {
        "description": "Content indicating involvement, membership, or support for non-violent radical activities.",
        "keywords": ["involved", "member", "support", "participate", "advocate", "promote"],
    },
    "High": {
        "description": "Content indicating direct or indirect involvement, intent, preparation, leadership, or explicit endorsement of violent extremism.",
        "keywords": ["violent", "jihad", "martyr", "fight", "kill", "attack", "war", "combat"],
    },
}

# ============ Indicator 定义 ============
INDICATORS = {
    "individual_loss_interpersonal": {
        "factor": "Need: Individual Loss",
        "description": "Struggling with interpersonal relationships, social isolation, or loss of close connections",
        "sample_keywords": ["friend", "relationship", "lonely", "isolated", "betrayed", "family conflict"],
        "example_content": "Just like his friend #Morsi, #Erdogan is an apostate, Kafir (disbeliever)",
    },
    "individual_loss_career": {
        "factor": "Need: Individual Loss",
        "description": "Lack of career prospects, unemployment, or dissatisfaction with work",
        "sample_keywords": ["job", "motivation", "career", "unemployment", "lack of opportunity"],
        "example_content": "I lack motivation. I think it's because I'm comfortable. I have a place to live and a job.",
    },
    "individual_loss_religious": {
        "factor": "Need: Individual Loss",
        "description": "Loss or crisis of religious faith, questioning religious identity",
        "sample_keywords": ["faith", "religion", "belief", "crisis", "confused", "spiritual"],
        "example_content": "In the lifetime of my generation and those that are close we will witness the biggest genocide",
    },
    "significance_gain_martyrdom": {
        "factor": "Need: Significance Gain",
        "description": "Seeking significance through martyrdom or sacrifice",
        "sample_keywords": ["martyr", "sacrifice", "die", "death", "willing", "shahid"],
        "example_content": "Running away from jihad will not save you from death. You can die as a coward or you can die as a martyr",
    },
    "narrative_violent_jihad_qital": {
        "factor": "Narrative: Violent",
        "description": "Discourse promoting violent jihad or armed struggle",
        "sample_keywords": ["jihad", "qital", "armed", "fight", "struggle", "resistance"],
        "example_content": "Some fight for their country, others fight for Allah. The world will soon see, Insha'Allah, which cause is the better one",
    },
    # 还有其他 74 个 indicators...
}

# ============ 输出目录 ============
OUTPUT_DIR = "/home/user/workspace/SHLi/AI for radicalisation/data/generated_samples"
SAMPLE_FILE = "samples.jsonl"

print("✓ Simulator 配置已加载")
