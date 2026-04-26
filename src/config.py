"""
集中化配置管理

统一管理应用中的所有配置常量，包括超时时间、文件限制、路径等。
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 在读取任何环境变量之前加载 .env
load_dotenv(PROJECT_ROOT / ".env")

# 数据目录
DATA_DIR = PROJECT_ROOT / os.getenv("DATA_DIR", "database")

# ========== 文件上传配置 ==========
# 文件大小限制（字节）
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_FILE_SIZE_FOR_ANALYSIS = 1 * 1024 * 1024  # 1MB（分析时限制）

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".csv"}

# 允许的MIME类型（基于扩展名）
ALLOWED_MIME_TYPES = {
    "text/plain": [".txt"],
    "text/markdown": [".md"],
    "application/pdf": [".pdf"],
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [
        ".docx"
    ],
    "text/csv": [".csv"],
}

# ========== 超时配置（秒） ==========
STORAGE_TIMEOUT = 60.0  # 存储操作超时
ANALYSIS_TIMEOUT = 120.0  # 分析操作超时
TOOL_CALL_TIMEOUT = 60.0  # 工具调用超时
WEBSOCKET_TIMEOUT = 300.0  # WebSocket连接超时

# ========== 重试配置 ==========
MAX_RETRIES = 1  # 最大重试次数
RETRY_DELAY = 1.0  # 重试延迟（秒）

# ========== 进度条配置 ==========
PROGRESS_UPDATE_INTERVAL = 0.5  # 进度条更新间隔（秒）
PROGRESS_STEPS = {
    "upload": 20,
    "storage": 40,
    "analysis": 40,
}

# ========== 向量数据库配置 ==========
CHROMA_PERSIST_DIR = DATA_DIR
EMBEDDING_MODEL = "qwen3-embedding:4b"

# ========== LLM配置 ==========
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "counselor-agent-demo")

# ========== Web服务器配置 ==========
HOST = "0.0.0.0"
PORT = 8000
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# ========== 会话管理配置 ==========
SESSION_TIMEOUT = 3600  # 会话超时时间（秒）
MAX_SESSIONS = 100  # 最大会话数

# ========== 文件类型检测配置 ==========
# 日记检测关键词
DIARY_KEYWORDS = ["日记", "日志", "journal", "diary", "心情", "情感", "情绪"]
# 对话大纲检测关键词
CONVERSATION_KEYWORDS = [
    "对话",
    "咨询",
    "session",
    "conversation",
    "大纲",
    "outline",
    "paip",
]
# 材料检测关键词
MATERIAL_KEYWORDS = ["材料", "资料", "学习", "阅读", "article", "paper", "研究"]

# 日期模式（用于日记检测）
DATE_PATTERNS = [
    r"\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}",  # 2024-12-01, 2024/12/01
    r"\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}",  # 01-12-2024, 01/12/24
    r"\d{1,2}月\d{1,2}日",  # 12月1日
]


# ========== 验证函数 ==========
def validate_file_extension(filename: str) -> bool:
    """验证文件扩展名是否允许"""
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def validate_file_size(file_size: int) -> bool:
    """验证文件大小是否在限制内"""
    return file_size <= MAX_FILE_SIZE


def get_allowed_extensions_str() -> str:
    """获取允许的文件扩展名字符串（用于显示）"""
    return ", ".join(ALLOWED_EXTENSIONS)
