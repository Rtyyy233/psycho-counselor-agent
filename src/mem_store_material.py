import os
import asyncio
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings

from dotenv import load_dotenv
from pydantic import BaseModel, Field


# ---------- 环境配置 ----------
PROJECT_ROOT = Path(__file__).parent.parent  # 根据实际项目结构调整
load_dotenv(PROJECT_ROOT / ".env")
DATA_DIR = PROJECT_ROOT / os.getenv("DATA_DIR", "data")
DATA_DIR.mkdir(exist_ok=True)


# ---------- 文本类型枚举 ----------
class MaterialType(str, Enum):
    """材料类型枚举，涵盖常见文档类别"""

    # 个人记录类
    JOURNAL = "日志"
    MEMOIR = "回忆录"
    LETTER = "书信"
    # 创作类
    STORY = "故事"
    POEM = "诗歌"
    PLAY = "剧本"
    ESSAY = "散文"
    # 专业/技术类
    ARTICLE = "文章"
    REPORT = "报告"
    PAPER = "论文"
    MANUAL = "手册"
    LEGAL = "法律文书"
    TECHNICAL_DOC = "技术文档"
    NEWS = "新闻"
    # 信息记录类
    NOTE = "笔记"
    MINUTES = "会议纪要"
    TRANSCRIPT = "转录文本"
    QA = "问答记录"
    # 其他
    UNKNOWN = "未知"


# ---------- 文件加载 ----------
def load_file(file_path: str) -> List[Document]:
    ext = Path(file_path).suffix.lower()[1:]
    loader_classes = {
        "txt": TextLoader,
        "pdf": PyPDFLoader,
        "md": UnstructuredMarkdownLoader,
        "csv": CSVLoader,
        "docx": UnstructuredWordDocumentLoader,
    }
    if ext not in loader_classes:
        raise ValueError(f"不支持的文件类型: {ext}")
    loader = loader_classes[ext](file_path, encoding="utf-8" if ext == "txt" else None)
    return loader.load()


# ---------- LLM 类型识别 ----------
class TypeInference(BaseModel):
    material_type: MaterialType = Field(description="从给定候选类型中选择最匹配的一项")


async def infer_material_type(
    text_sample: str, llm: Optional[BaseChatModel] = None
) -> MaterialType:
    """异步识别文本类型"""
    if llm is None:
        llm = ChatDeepSeek(model="deepseek-chat", temperature=0)

    structured_llm = llm.with_structured_output(TypeInference)
    type_options = "\n".join([f"- {t.value}" for t in MaterialType])
    prompt = f"""你是一位文本分类专家。请阅读以下文本片段，并从下方列表中选出最符合的文本类型。

候选类型：
{type_options}

注意：
1. 仅返回列表中出现的类型名称，不要自行创造新类别。
2. 如果文本特征不明显，请返回“未知”。

文本片段（前1500字符）：
{text_sample[:1500]}
"""
    try:
        result = structured_llm.invoke(prompt)
        return result.material_type  # type:ignore
    except Exception:
        return MaterialType.UNKNOWN


# ---------- 异步父子语义分块器 ----------
class ParentChildSemanticSplitter:
    def __init__(
        self,
        embedding_model: str = "qwen3-embedding:4b",
        parent_threshold: float = 80.0,
        child_threshold: float = 60.0,
        parent_buffer_size: int = 3,
        child_buffer_size: int = 2,
    ):
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.parent_splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_amount=parent_threshold,
            buffer_size=parent_buffer_size,
        )
        self.child_splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_amount=child_threshold,
            buffer_size=child_buffer_size,
        )

    async def split_documents(self, documents: List[Document]) -> Dict:
        """
        异步执行父子分块，将同步的 SemanticChunker 操作放入线程池。
        """
        # 将同步分块操作封装为异步调用
        loop = asyncio.get_running_loop()

        # 父块分割（CPU/IO 操作放入线程池）
        parent_docs = await loop.run_in_executor(
            None, self.parent_splitter.split_documents, documents
        )

        parent_map = {}
        child_docs = []

        for idx, p_doc in enumerate(parent_docs):
            parent_id = f"parent_{idx:06d}"
            parent_map[parent_id] = p_doc.page_content

            temp_doc = Document(page_content=p_doc.page_content)
            # 子块分割同样放入线程池
            children = await loop.run_in_executor(
                None, self.child_splitter.split_documents, [temp_doc]
            )

            for child in children:
                child.metadata.update(
                    {
                        "parent_id": parent_id,
                        "parent_index": idx,
                    }
                )
                child_docs.append(child)

        return {
            "parent_chunks": parent_docs,
            "child_chunks": child_docs,
            "parent_map": parent_map,
        }


# ---------- 唯一 ID 生成器 ----------
class UniqueIDGenerator:
    def __init__(self):
        self._last_timestamp = ""
        self._sequence = 0

    def generate(self, prefix: str = "") -> str:
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        ms = now.microsecond // 1000
        full_ts = f"{timestamp}_{ms:03d}"

        if full_ts == self._last_timestamp:
            self._sequence += 1
        else:
            self._sequence = 0
            self._last_timestamp = full_ts

        seq_str = f"{self._sequence:03d}"
        base = f"{full_ts}_{seq_str}"
        return f"{prefix}_{base}" if prefix else base


# ---------- 主存储函数（异步） ----------

async def store_materials(file_path: str) -> List[str]:
    """
    将用户提供的非日记类材料存入向量数据库。
    """
    from mem_integration import material_store, parent_store
    # 1. 加载文档（同步操作，但通常很快，若文件较大可考虑异步加载）
    raw_docs = load_file(file_path)
    if not raw_docs:
        raise ValueError("文件内容为空或加载失败")

    full_text = "\n".join([doc.page_content for doc in raw_docs])

    # 2. 异步识别类型
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
    material_type = await infer_material_type(full_text[:1500], llm)

    # 3. 异步父子分块
    splitter = ParentChildSemanticSplitter()
    split_result = await splitter.split_documents(raw_docs)
    child_chunks = split_result["child_chunks"]

    # 4. 准备块元数据与 ID（同时存储父块和子块）
    store_date = datetime.now().strftime("%Y-%m-%d")
    id_gen = UniqueIDGenerator()
    child_ids = []
    docs_to_store = []

    # 按parent_index分组收集子块并先为子块生成ID
    parent_index_to_children: Dict[int, List] = {}
    child_with_ids = []  # 存储(子块, 子块ID)对

    for child in child_chunks:
        child_id = id_gen.generate(prefix="child")
        child_ids.append(child_id)
        p_idx = child.metadata.get("parent_index", 0)

        if p_idx not in parent_index_to_children:
            parent_index_to_children[p_idx] = []
        parent_index_to_children[p_idx].append(child_id)  # 存储子块ID而不是子块对象

        child_with_ids.append((child, child_id, p_idx))

    # 存储父块
    parent_ids_list: List[str] = []
    parent_id_by_index: Dict[int, str] = {}
    for idx, parent in enumerate(split_result["parent_chunks"]):
        parent_id = id_gen.generate(prefix="parent")
        parent_ids_list.append(parent_id)
        parent_id_by_index[idx] = parent_id
        child_ids_for_this_parent = parent_index_to_children.get(idx, [])

        docs_to_store.append(
            Document(
                id=parent_id,
                page_content=parent.page_content,
                metadata={
                    "date": store_date,
                    "text_type": material_type.value,
                    "chunk_type": "parent",
                    "parent_id": parent_id,
                    "child_ids": child_ids_for_this_parent,  # 现在这是字符串列表
                    "source_file": str(file_path),
                },
            )
        )

    # 存储子块（带parent_id链接）
    for child, child_id, p_idx in child_with_ids:
        parent_id = parent_id_by_index.get(p_idx, "")

        child.metadata.update(
            {
                "date": store_date,
                "text_type": material_type.value,
                "chunk_type": "child",
                "parent_id": parent_id,
                "source_file": str(file_path),
            }
        )

        docs_to_store.append(
            Document(
                id=child_id,
                page_content=child.page_content,
                metadata=child.metadata,
            )
        )

    # 5. 存入向量库 - 分离存储到两个集合
    parent_docs_to_store = [
        doc for doc in docs_to_store if doc.metadata.get("chunk_type") == "parent"
    ]
    child_docs_to_store = [
        doc for doc in docs_to_store if doc.metadata.get("chunk_type") == "child"
    ]

    loop = asyncio.get_running_loop()

    # 存储父块到 parent_store
    await loop.run_in_executor(None, parent_store.add_documents, parent_docs_to_store)

    # 存储子块到 child_store
    await loop.run_in_executor(None, material_store.add_documents, child_docs_to_store)

    return child_ids
