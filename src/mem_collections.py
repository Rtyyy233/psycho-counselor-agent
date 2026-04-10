from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
from dotenv import load_dotenv
import os


def find_project_root(start_path=Path(__file__).parent):
    for parent in [start_path] + list(start_path.parents):
        if (parent / ".env").exists():
            return parent
    return start_path

PROJECT_ROOT = find_project_root()
load_dotenv(PROJECT_ROOT / ".env")  # 显式指定 .env 路径

# 获取相对路径字符串
rel_data_dir = os.getenv("DATA_DIR", "data")
# 转换为绝对路径
abs_data_dir = PROJECT_ROOT / rel_data_dir

data_path = PROJECT_ROOT / os.getenv("DATA_DIR", "data")

DATA_DIR = data_path

embeddings = OllamaEmbeddings(model="qwen3-embedding:4b")

original_diary = Chroma(
        collection_name = "original_diary",
        embedding_function=embeddings,
        persist_directory=str(data_path) # notice the problem of hard coed path
    )

diary_annotation = Chroma(
        collection_name = "diary_annotation",
        embedding_function=embeddings,
        persist_directory=str(data_path) # notice the problem of hard coed path
    )

material_store = Chroma(
        collection_name="child_chunks",
        embedding_function=embeddings,
        persist_directory=str(DATA_DIR),
    )

conv_store = Chroma(
        collection_name="conv_outline",
        embedding_function=embeddings,
        persist_directory=str(DATA_DIR),
    )