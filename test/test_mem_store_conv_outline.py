# test/test_mem_store_conv_outline.py
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# 将 src 加入系统路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mem_store_conv_outline import (
    DATA_DIR,
    COLLECTION_NAME,
    PAIPOutline,
    generate_paip_outline,
    store_conversation_outline,
)
from langchain_core.documents import Document


# ---------- Fixtures ----------
@pytest.fixture
def sample_conversation_doc():
    """构造一个包含咨询对话的 Document 对象"""
    text = """
咨询师：你好，今天想从哪儿开始？
来访者：我最近总是睡不好，脑子里反复想工作的事。
咨询师：听起来焦虑情绪比较明显，能具体说说工作上的压力吗？
来访者：项目截止日期快到了，我担心完不成，领导会失望。
咨询师：我们试试用认知行为的方法来看待这个想法。
"""
    return Document(page_content=text, metadata={"source": "session_001"})


@pytest.fixture
def mock_llm():
    """模拟 ChatDeepSeek，返回预设的 PAIPOutline"""
    with patch("mem_store_conv_outline.ChatDeepSeek") as mock_llm_class:
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        # 预设异步调用返回值
        fake_outline = PAIPOutline(
            problem="工作压力导致失眠，担心让领导失望",
            assessment="焦虑情绪明显，自动思维为'我不够好'",
            intervention="CBT 认知重构，挑战自动思维",
            plan="记录情绪日记，下周继续探索核心信念"
        )
        mock_structured.ainvoke = AsyncMock(return_value=fake_outline)
        yield mock_llm_class


@pytest.fixture
def mock_splitter():
    """模拟 ParentChildSemanticSplitter，返回预设子块"""
    with patch("mem_store_conv_outline.ParentChildSemanticSplitter") as mock_splitter_cls:
        mock_instance = MagicMock()
        mock_splitter_cls.return_value = mock_instance
        
        # 模拟 split_documents 异步方法返回两个子块
        async def mock_split(docs):
            return {
                "parent_chunks": [Document(page_content="父块")],
                "child_chunks": [
                    Document(page_content="子块1内容", metadata={}),
                    Document(page_content="子块2内容", metadata={}),
                ],
                "parent_map": {}
            }
        mock_instance.split_documents = mock_split
        yield mock_splitter_cls


@pytest.fixture
def mock_vectorstore():
    """模拟 Chroma 向量库"""
    with patch("mem_store_conv_outline.Chroma") as mock_chroma:
        mock_instance = MagicMock()
        mock_chroma.return_value = mock_instance
        mock_instance.add_documents = MagicMock()
        yield mock_chroma


@pytest.fixture
def mock_embeddings():
    """模拟 OllamaEmbeddings"""
    with patch("mem_store_conv_outline.OllamaEmbeddings") as mock_emb:
        mock_emb.return_value = MagicMock()
        yield mock_emb


# ---------- 测试 generate_paip_outline ----------
@pytest.mark.asyncio
async def test_generate_paip_outline_success(mock_llm):
    """测试正常生成 PAIP 摘要"""
    result = await generate_paip_outline("测试对话内容")
    assert isinstance(result, PAIPOutline)
    assert result.problem == "工作压力导致失眠，担心让领导失望"
    assert result.assessment == "焦虑情绪明显，自动思维为'我不够好'"
    assert result.intervention == "CBT 认知重构，挑战自动思维"
    assert result.plan == "记录情绪日记，下周继续探索核心信念"


@pytest.mark.asyncio
async def test_generate_paip_outline_fallback():
    """测试 LLM 调用异常时的降级处理"""
    with patch("mem_store_conv_outline.ChatDeepSeek") as mock_llm_class:
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.ainvoke = AsyncMock(side_effect=Exception("API 错误"))

        result = await generate_paip_outline("测试对话")
        assert result.problem == "生成失败"
        assert result.assessment == "生成失败"
        assert result.intervention == "生成失败"
        assert result.plan == "生成失败"


# ---------- 测试 store_conversation_outline ----------
@pytest.mark.asyncio
async def test_store_conversation_outline_success(
    sample_conversation_doc, mock_llm, mock_splitter, mock_embeddings, mock_vectorstore
):
    """测试完整存储流程，验证返回 base_id 及元数据写入"""
    base_id = await store_conversation_outline(sample_conversation_doc) #测试写于使用@tool装饰函数之前

    # 1. 验证返回的 ID 格式正确
    assert base_id.startswith("conv_")
    parts = base_id.split("_")
    assert len(parts) == 5  # conv + date + time + ms + seq

    # 2. 验证向量库 add_documents 被调用（至少两次：子块 + PAIP）
    mock_vectorstore.return_value.add_documents.assert_called()
    # 获取所有调用参数
    all_calls = mock_vectorstore.return_value.add_documents.call_args_list
    assert len(all_calls) >= 2  # 子块写入一次，PAIP 各部分写入一次（或多次）

    # 3. 检查子块文档的元数据
    child_call = all_calls[0]
    child_docs = child_call[0][0]  # 第一个位置参数是文档列表
    assert len(child_docs) == 2  # 模拟分块返回两个子块
    for doc in child_docs:
        assert doc.metadata["base_id"] == base_id
        assert doc.metadata["section"] == "raw"
        assert doc.metadata["text_type"] == "conversation"
        assert "date" in doc.metadata

    # 4. 检查 PAIP 文档的元数据
    paip_call = all_calls[1]
    paip_docs = paip_call[0][0]
    assert len(paip_docs) == 4  # problem, assessment, intervention, plan
    expected_sections = {"problem", "assessment", "intervention", "plan"}
    for doc in paip_docs:
        assert doc.metadata["base_id"] == base_id
        assert doc.metadata["section"] in expected_sections
        assert doc.metadata["text_type"] == "paip_summary"


@pytest.mark.asyncio
async def test_store_conversation_outline_empty_document():
    """测试空对话内容应抛出异常"""
    empty_doc = Document(page_content="   ", metadata={})
    with pytest.raises(ValueError, match="对话内容为空"):
        await store_conversation_outline(empty_doc)


@pytest.mark.asyncio
async def test_store_conversation_outline_metadata_source(
    sample_conversation_doc, mock_llm, mock_splitter, mock_embeddings, mock_vectorstore
):
    """测试原始 metadata 中的 source 字段被正确传递"""
    await store_conversation_outline(sample_conversation_doc)
    child_call = mock_vectorstore.return_value.add_documents.call_args_list[0]
    child_docs = child_call[0][0]
    for doc in child_docs:
        assert doc.metadata["source"] == "session_001"


# ---------- 集成测试标记（可选） ----------
@pytest.mark.integration
@pytest.mark.asyncio
async def test_store_conversation_outline_integration(sample_conversation_doc):
    """
    集成测试：真实调用 Ollama 和 DeepSeek API。
    需提前配置好环境变量及服务。
    """
    # 清理旧数据（可选）
    import shutil
    test_collection_path = DATA_DIR / COLLECTION_NAME
    if test_collection_path.exists():
        shutil.rmtree(test_collection_path)

    base_id = await store_conversation_outline(sample_conversation_doc)
    assert base_id.startswith("conv_")

    # 验证数据已写入：使用 Chroma 查询
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings

    embeddings = OllamaEmbeddings(model="qwen3-embedding:4b")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(DATA_DIR),
    )
    # 根据 base_id 获取所有关联文档
    results = vectorstore.get(where={"base_id": base_id})
    assert len(results["ids"]) >= 5  # 至少 2 子块 + 4 PAIP 块

# mock 模式
#pytest test/test_mem_store_conv_outline.py -v -m "not integration"

# integration
#pytest test/test_mem_store_conv_outline.py -v -m integration