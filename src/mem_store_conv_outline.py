# src/mem_store_conv_outline.py
import asyncio
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Union

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# 复用父子分块器与 ID 生成器
from mem_store_material import ParentChildSemanticSplitter, UniqueIDGenerator

# ---------- 环境配置 ----------
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")
DATA_DIR = PROJECT_ROOT / os.getenv("DATA_DIR", "data")
DATA_DIR.mkdir(exist_ok=True)

COLLECTION_NAME = "conv_outline"

# ---------- PAIP 结构化模型 ----------
class PAIPOutline(BaseModel):
    """PAIP 咨询记录摘要模型"""
    problem: str = Field(description="本次咨询聚焦的核心问题，包括来访者描述与症状表现")
    assessment: str = Field(description="咨询师对问题进展、功能影响及临床分析的评估")
    intervention: str = Field(description="咨询师为解决该问题采取的具体干预措施与策略")
    plan: str = Field(description="针对该问题的下一步计划，包括家庭作业或后续咨询安排")

# ---------- LLM 生成 PAIP 摘要 ----------
async def generate_paip_outline(
    conversation_text: str,
    llm: Optional[BaseChatModel] = None
) -> PAIPOutline:
    """使用 LLM 根据对话原文生成 PAIP 结构化摘要"""
    if llm is None:
        llm = ChatDeepSeek(model="deepseek-chat", temperature=0)

    structured_llm = llm.with_structured_output(PAIPOutline) # 后续建立完整Agent后考虑调用专门的Analysist模块提供以下内容
    prompt = f"""你是一位专业的心理咨询记录分析师。请根据以下咨询对话内容，提取并整理成 PAIP 格式的摘要。

【PAIP 格式说明】
- Problem（问题）：清晰陈述本次咨询聚焦的核心问题，包含来访者的描述、症状表现以及对日常生活的影响。
- Assessment（评估）：咨询师对来访者在目标问题上的进展评估、功能影响分析及临床判断。
- Intervention（干预）：咨询师在本次咨询中采取的具体干预技术、策略或对话重点。
- Plan（计划）：针对该问题的下一步行动计划，包括家庭作业、下次咨询主题或建议。

注意：
1. 仅基于对话内容进行提取，不要添加额外信息。
2. 若某项信息未在对话中明确体现，请填写"未提及"。
3. 语言应专业、客观、精炼。

对话原文：
{conversation_text}  
"""
    try:
        result = await structured_llm.ainvoke(prompt)  # 异步调用
        return result #type:ignore
    except Exception as e:
        # 降级处理：返回空字段并记录错误（可根据需要调整）
        print(f"PAIP 生成失败: {e}")
        return PAIPOutline(
            problem="生成失败",
            assessment="生成失败",
            intervention="生成失败",
            plan="生成失败"
        )

# ---------- 主存储函数 ----------

async def store_conversation_outline(conv_doc: Document) -> str:
    """
    存储咨询对话原文及对应的 PAIP（problem,assessment,intervention,plan) 摘要
    """
    # 1. 提取对话文本
    full_text = conv_doc.page_content
    if not full_text or not full_text.strip():
        raise ValueError("对话内容为空")

    # 2. 生成唯一基础 ID（格式：conv_YYYYMMDD_HHMMSS_mmm_seq）
    id_gen = UniqueIDGenerator()
    base_id = id_gen.generate(prefix="conv")

    # 3. 异步生成 PAIP 摘要
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
    paip_outline = await generate_paip_outline(full_text, llm)

    # 4. 对对话原文进行父子分块（复用已实现的异步分块器）
    splitter = ParentChildSemanticSplitter()
    # 将原文包装为 Document 列表
    raw_docs = [Document(page_content=full_text, metadata=conv_doc.metadata.copy())]
    split_result = await splitter.split_documents(raw_docs)
    child_chunks = split_result["child_chunks"]

    # 5. 准备所有待存储的文档
    store_date = datetime.now().strftime("%Y-%m-%d")
    embeddings = OllamaEmbeddings(model="qwen3-embedding:4b")
    conv_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(DATA_DIR),
    )

    # 5.1 原文子块文档（每个子块有独立 ID，但共享 base_id）
    child_docs = []
    child_ids = []
    for idx, child in enumerate(child_chunks):
        child_id = f"{base_id}_child_{idx:04d}"
        child_ids.append(child_id)
        child.metadata.update({
            "base_id": base_id,
            "date": store_date,
            "text_type": "conversation",
            "chunk_type": "child",
            "section": "raw",          # 用于过滤原文块
            "source": conv_doc.metadata.get("source", "unknown"),
        })
        child_docs.append(
            Document(
                id=child_id,
                page_content=child.page_content,
                metadata=child.metadata,
            )
        )

    # 5.2 PAIP 各部分文档（每个部分一个 Document，ID 含 section 标识）
    paip_docs = []
    sections = {
        "problem": paip_outline.problem,
        "assessment": paip_outline.assessment,
        "intervention": paip_outline.intervention,
        "plan": paip_outline.plan,
    }
    for section, content in sections.items():
        doc_id = f"{base_id}_{section}"
        metadata = {
            "base_id": base_id,
            "date": store_date,
            "text_type": "paip_summary",
            "section": section,
            "source": conv_doc.metadata.get("source", "unknown"),
        }
        paip_docs.append(
            Document(
                id=doc_id,
                page_content=content,
                metadata=metadata,
            )
        )

    # 6. 批量写入向量库（使用线程池执行同步操作）
    loop = asyncio.get_running_loop()
    # 写入子块
    if child_docs:
        await loop.run_in_executor(
            None, conv_store.add_documents, child_docs
        )
    # 写入 PAIP 各部分
    if paip_docs:
        paip_ids = [doc.id for doc in paip_docs]
        await loop.run_in_executor(
            None, conv_store.add_documents, paip_docs
        )

    return base_id



