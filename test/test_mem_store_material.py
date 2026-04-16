# test/test_mem_store_material.py
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# 将 src 目录加入系统路径，以便导入模块
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mem_store_material import (
    DATA_DIR,
    MaterialType,
    ParentChildSemanticSplitter,
    UniqueIDGenerator,
    infer_material_type,
    load_file,
    store_materials,
)
from langchain_core.documents import Document


# ---------- Fixtures ----------
@pytest.fixture
def sample_txt_file(tmp_path):
    """创建一个临时 txt 文件用于测试"""
    content = "这是第一段测试内容。\n\n这是第二段，包含更多文字以便分块。"
    file_path = tmp_path / "test_diary.txt"
    file_path.write_text(content, encoding="utf-8")
    return str(file_path)


@pytest.fixture
def mock_embeddings():
    """模拟嵌入模型，避免真实调用 Ollama"""
    with patch("mem_store_material.OllamaEmbeddings") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_vectorstore():
    """模拟 Chroma 向量库"""
    with patch("mem_store_material.Chroma") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        mock_instance.add_documents = MagicMock()
        yield mock_instance


@pytest.fixture
def mock_llm():
    """模拟 ChatDeepSeek，返回固定的类型推断结果"""
    with patch("mem_store_material.ChatDeepSeek") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        # 模拟 with_structured_output 链式调用
        structured_mock = MagicMock()
        mock_instance.with_structured_output.return_value = structured_mock
        # 模拟 invoke 返回一个具有 material_type 属性的对象
        fake_result = MagicMock()
        
        structured_mock.invoke.return_value = fake_result
        yield mock_instance


# ---------- 测试 load_file ----------
def test_load_file_txt(sample_txt_file):
    """测试加载 txt 文件"""
    docs = load_file(sample_txt_file)
    assert len(docs) == 1
    assert "第一段测试内容" in docs[0].page_content


def test_load_file_unsupported():
    """测试不支持的文件类型抛出异常"""
    with pytest.raises(ValueError, match="不支持的文件类型"):
        load_file("test.xyz")


# ---------- 测试 UniqueIDGenerator ----------
def test_unique_id_generator_uniqueness():
    """测试 ID 生成器在快速连续调用时的唯一性"""
    gen = UniqueIDGenerator()
    ids = [gen.generate(prefix="child") for _ in range(10)]
    # 所有 ID 应唯一
    assert len(ids) == len(set(ids))
    # ID 格式应为 child_YYYYMMDD_HHMMSS_mmm_sss
    for id_ in ids:
        assert id_.startswith("child_")
        parts = id_.split("_")
        assert len(parts) == 5  # child + date + time + ms + seq


def test_unique_id_generator_date_included():
    """测试 ID 中包含当天日期"""
    gen = UniqueIDGenerator()
    id_ = gen.generate()
    today_str = datetime.now().strftime("%Y%m%d")
    assert today_str in id_


# ---------- 测试 infer_material_type ----------



@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_infer_material_type_fallback():
    with patch("mem_store_material.ChatDeepSeek") as mock_llm_class:
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.side_effect = Exception("LLM error")
        
        result = await infer_material_type("任意文本")
        assert result == MaterialType.UNKNOWN


# ---------- 测试 ParentChildSemanticSplitter ----------
@pytest.mark.asyncio
async def test_split_documents_basic():
    """测试异步分块器的基本功能"""
    docs = [Document(page_content="段落一。段落二。段落三。")]
    splitter = ParentChildSemanticSplitter()
    
    # 注意：这里会真实调用 OllamaEmbeddings，若未启动 Ollama 会失败。
    # 在集成测试中可接受；单元测试中应 mock SemanticChunker。
    with patch.object(splitter.parent_splitter, "split_documents") as mock_parent, \
         patch.object(splitter.child_splitter, "split_documents") as mock_child:
        mock_parent.return_value = [Document(page_content="段落一。段落二。"), Document(page_content="段落三。")]
        mock_child.side_effect = [
            [Document(page_content="段落一。"), Document(page_content="段落二。")],
            [Document(page_content="段落三。")],
        ]
        result = await splitter.split_documents(docs)
    
    assert "parent_chunks" in result
    assert "child_chunks" in result
    assert len(result["child_chunks"]) == 3
    # 检查子块是否继承了 parent_id
    for child in result["child_chunks"]:
        assert "parent_id" in child.metadata


# ---------- 测试 store_materials 整体流程 ----------
@pytest.mark.asyncio
async def test_store_materials_success(
    sample_txt_file, mock_llm, mock_embeddings, mock_vectorstore
):
    """测试完整存储流程（mock 所有外部依赖）"""
    # 对 SemanticChunker 进行 mock，避免真实调用 Ollama
    with patch("mem_store_material.SemanticChunker") as mock_sem_chunker:
        # 模拟父块分割器返回两个父块
        mock_parent_splitter = MagicMock()
        mock_parent_splitter.split_documents.return_value = [
            Document(page_content="父块1内容"),
            Document(page_content="父块2内容"),
        ]
        # 模拟子块分割器为每个父块返回一个子块
        mock_child_splitter = MagicMock()
        mock_child_splitter.split_documents.side_effect = [
            [Document(page_content="子块1-1")],
            [Document(page_content="子块2-1")],
        ]
        # 让 SemanticChunker 的两个实例分别返回对应的 mock
        mock_sem_chunker.side_effect = [mock_parent_splitter, mock_child_splitter]

        # 执行存储
        ids = await store_materials(sample_txt_file)

    # 验证返回的 ID 数量与子块数量一致
    assert len(ids) == 2
    # 验证向量库的 add_documents 被调用，且传入了正确数量的文档
    mock_vectorstore.add_documents.assert_called_once()
    args, kwargs = mock_vectorstore.add_documents.call_args
    docs_passed = args[0]
    assert len(docs_passed) == 2
    # 检查文档元数据
    for doc in docs_passed:
        assert doc.metadata["date"] == datetime.now().strftime("%Y-%m-%d")
        
        assert doc.metadata["source_file"] == sample_txt_file
        assert "parent_id" in doc.metadata




# ---------- 集成测试（需要真实 Ollama 和 DeepSeek API） ----------
@pytest.mark.integration
@pytest.mark.asyncio
async def test_store_materials_integration(sample_txt_file):
    """集成测试：真实调用 Ollama 嵌入模型和 DeepSeek API"""
    # 可选清理：删除持久化目录下的所有 Chroma 数据（如需重置）
    import shutil
    chroma_persist_dir = DATA_DIR
    if chroma_persist_dir.exists():
        shutil.rmtree(chroma_persist_dir)
        chroma_persist_dir.mkdir(exist_ok=True)

    # 执行存储
    ids = await store_materials(sample_txt_file) #原代码基于没有@tool装饰时进行测试，加入@tool装饰后不可用
    assert isinstance(ids, list)
    assert len(ids) > 0

    # 通过 Chroma 查询验证数据已持久化
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings

    embeddings = OllamaEmbeddings(model="qwen3-embedding:4b")
    vectorstore = Chroma(
        collection_name="child_chunks",
        embedding_function=embeddings,
        persist_directory=str(DATA_DIR),
    )
    # 获取第一个 ID 对应的文档
    retrieved = vectorstore.get(ids=[ids[0]])
    assert retrieved["ids"] == [ids[0]]
    assert retrieved["documents"]  # 文档内容非空


# ---------- 运行说明 ----------
"""
运行所有单元测试（mock 版本，无需外部服务）：
    pytest test/test_mem_store_material.py -v

运行集成测试（需要真实服务）：
    pytest test/test_mem_store_material.py -v -m integration

若未安装 pytest-asyncio，请先执行：
    pip install pytest-asyncio
""" 