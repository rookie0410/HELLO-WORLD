"""
Configuration settings for the arXiv Benchmark Analyzer
"""
import os
from pathlib import Path

# ===== API Configuration =====
# Large model API settings (for detailed analysis)
API_KEY = os.environ.get("OPENAI_API_KEY", "your-api-key-here")
API_BASE = os.environ.get("API_BASE", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo-16k")

# ===== Local Model Configuration =====
# Small model for benchmark classification (using transformers)
LOCAL_MODEL_NAME = os.environ.get("LOCAL_MODEL_NAME", "BAAI/bge-small-zh-v1.5")
LOCAL_CLASSIFIER_MODEL = os.environ.get("LOCAL_CLASSIFIER_MODEL", "facebook/bart-large-mnli")

# ===== Directory Configuration =====
BASE_DIR = Path(__file__).parent
PDF_DIR = BASE_DIR / "arxiv_pdfs"
DATABASE_PATH = BASE_DIR / "arxiv_benchmark.db"
OUTPUT_DIR = BASE_DIR / "output"

# Create directories
PDF_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ===== arXiv Search Configuration =====
DEFAULT_SEARCH_DAYS = 7
DEFAULT_MAX_RESULTS = 100

# ===== Classification Labels =====
CLASSIFICATION_LABELS = [
    # 知识类
    "综合知识 (World Knowledge)",
    "前沿与边缘知识",
    
    # 能力类
    "逻辑与推理 (Reasoning)",
    "数学 (Mathematics)",
    "代码 (Coding)",
    "软件工程 (Software Eng.)",
    
    # 行业类
    "法律 (Legal)",
    "金融 (Finance)",
    "医疗 (Medical)",
    "科学发现 (Scientific)",
    "生物与基因 (Genomics)",
    "电信 (Telecom)",
    "人力资源 (HR)",
    "教育 (Education)",
    "供应链与物流",
    "文化",
    
    # Agent能力
    "工具使用 (Tool Use)",
    "环境交互与规划",
    "多智能体协作",
    "游戏智能 (Gaming)",
    
    # RAG与长文本
    "RAG系统鲁棒性",
    "长文本理解",
    "学术与论文分析",
    
    # 多模态
    "视觉-语言 (VLM)",
    "文字文本",
    "视频理解与生成",
    "音频与语音",
    "3D与具身智能",
    
    # 安全与伦理
    "机器遗忘 (Unlearning)",
    "版权与合规",
    "安全与越狱",
    "幻觉 (Hallucination)",
    "欺骗与奉承",
    
    # 心理与认知
    "情商与人格",
    "认知偏差",
    
    # 可解释与鲁棒性
    "可解释性与忠实度",
    "提示词鲁棒性",
    
    # 性能与效率
    "推理性能",
    "绿色AI (能效)",
    "水印鲁棒性",
    
    # 特殊数据类型
    "表格理解 (Table)",
    "时间序列 (Time Series)",
    "图结构 (Graph)",
    "推荐系统 (RecSys)",
    
    # 评估方法
    "LLM裁判 (LLM-as-a-Judge)",
    "数据污染检测",
    "不确定性校准",
]

# Label categories for better organization
LABEL_CATEGORIES = {
    "知识类": ["综合知识 (World Knowledge)", "前沿与边缘知识"],
    "能力类": ["逻辑与推理 (Reasoning)", "数学 (Mathematics)", "代码 (Coding)", "软件工程 (Software Eng.)"],
    "行业类": ["法律 (Legal)", "金融 (Finance)", "医疗 (Medical)", "科学发现 (Scientific)", 
               "生物与基因 (Genomics)", "电信 (Telecom)", "人力资源 (HR)", "教育 (Education)", 
               "供应链与物流", "文化"],
    "Agent能力": ["工具使用 (Tool Use)", "环境交互与规划", "多智能体协作", "游戏智能 (Gaming)"],
    "RAG与长文本": ["RAG系统鲁棒性", "长文本理解", "学术与论文分析"],
    "多模态": ["视觉-语言 (VLM)", "文字文本", "视频理解与生成", "音频与语音", "3D与具身智能"],
    "安全与伦理": ["机器遗忘 (Unlearning)", "版权与合规", "安全与越狱", "幻觉 (Hallucination)", "欺骗与奉承"],
    "心理与认知": ["情商与人格", "认知偏差"],
    "可解释与鲁棒性": ["可解释性与忠实度", "提示词鲁棒性"],
    "性能与效率": ["推理性能", "绿色AI (能效)", "水印鲁棒性"],
    "特殊数据类型": ["表格理解 (Table)", "时间序列 (Time Series)", "图结构 (Graph)", "推荐系统 (RecSys)"],
    "评估方法": ["LLM裁判 (LLM-as-a-Judge)", "数据污染检测", "不确定性校准"],
}
