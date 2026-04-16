# test/test_mem_retrieve_material.py
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mem_retrieve_material import (
    material_filter,
    material_retrieve_step,
    MaterialResult,
    material_state,
    _build_child_where_clause,
    plan_node,
    semantic_search_children,
    metadata_filter_node,
    parent_lookup_node,
    children_lookup_node,
    build_material_graph,
)
from langchain_core.documents import Document


# ---------- Tests for _build_child_where_clause ----------
def test_build_child_where_clause_empty():
    flt = material_filter()
    where = _build_child_where_clause(flt)
    assert where == {}


def test_build_child_where_clause_text_type():
    flt = material_filter(text_type=["文章", "笔记"])
    where = _build_child_where_clause(flt)
    assert where == {"text_type": {"$in": ["文章", "笔记"]}}


def test_build_child_where_clause_date_range():
    flt = material_filter(date_start="2025-01-01", date_end="2025-03-15")
    where = _build_child_where_clause(flt)
    assert where == {"date": {"$gte": "2025-01-01", "$lte": "2025-03-15"}}


def test_build_child_where_clause_source_file():
    flt = material_filter(source_file="test.txt")
    where = _build_child_where_clause(flt)
    assert where == {"source_file": "test.txt"}


def test_build_child_where_clause_combined():
    flt = material_filter(text_type=["文章"], date_start="2025-01-01")
    where = _build_child_where_clause(flt)
    assert where["text_type"] == {"$in": ["文章"]}
    assert where["date"] == {"$gte": "2025-01-01"}


# ---------- Tests for semantic_search_children ----------
@pytest.mark.asyncio
async def test_semantic_search_children_returns_child_docs():
    """Test semantic search returns child documents."""
    mock_result = [
        (Document(page_content="child content", id="c1", metadata={"chunk_type": "child"}), 0.9),
        (Document(page_content="child 2", id="c2", metadata={"chunk_type": "child"}), 0.8),
    ]

    with patch("mem_retrieve_material.material_store") as mock_material, \
         patch("mem_retrieve_material.parent_store"):
        mock_material.similarity_search_with_score.return_value = mock_result

        state: material_state = {
            "query": "测试查询",
            "plan": [material_retrieve_step(step_id=1, mode="semantic_search", target="children", temp_query="测试")],
            "current_step_idx": 0,
            "results": [],
            "matched_child_ids": [],
            "matched_parent_ids": [],
        }

        result_state = await semantic_search_children(state)

        assert len(result_state["results"]) == 1
        assert len(result_state["results"][0].matched_children) == 2
        assert result_state["matched_child_ids"] == ["c1", "c2"]


@pytest.mark.asyncio
async def test_semantic_search_children_skips_wrong_mode():
    """Test semantic_search_children skips when mode is not semantic_search."""
    state: material_state = {
        "query": "测试",
        "plan": [material_retrieve_step(step_id=1, mode="metadata_filter", target="children")],
        "current_step_idx": 0,
        "results": [],
        "matched_child_ids": [],
        "matched_parent_ids": [],
    }

    result_state = await semantic_search_children(state)

    assert len(result_state["results"]) == 0


# ---------- Tests for metadata_filter_node ----------
@pytest.mark.asyncio
async def test_metadata_filter_node():
    """Test metadata filter on child store."""
    mock_store = MagicMock()
    mock_store.get.return_value = {
        "ids": ["c1", "c2"],
        "documents": ["doc1", "doc2"],
        "metadatas": [
            {"chunk_type": "child", "text_type": "文章"},
            {"chunk_type": "child", "text_type": "笔记"},
        ],
    }

    with patch("mem_retrieve_material.material_store", mock_store), \
         patch("mem_retrieve_material.parent_store"):
        state: material_state = {
            "query": "测试",
            "plan": [
                material_retrieve_step(
                    step_id=1,
                    mode="metadata_filter",
                    target="children",
                    filter=material_filter(text_type=["文章"]),
                )
            ],
            "current_step_idx": 0,
            "results": [],
            "matched_child_ids": [],
            "matched_parent_ids": [],
        }

        result_state = await metadata_filter_node(state)

        assert len(result_state["results"]) == 1
        assert len(result_state["results"][0].matched_children) == 2


# ---------- Tests for parent_lookup_node ----------
@pytest.mark.asyncio
async def test_parent_lookup_fetches_parent_docs():
    """Test parent_lookup fetches parent documents from parent_store."""
    mock_material = MagicMock()
    mock_parent = MagicMock()
    call_count = 0

    def material_get_side_effect(ids=None, where=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call: get children to find parent_ids
            return {
                "ids": ["c1", "c2"],
                "documents": ["child1", "child2"],
                "metadatas": [
                    {"chunk_type": "child", "parent_id": "p1"},
                    {"chunk_type": "child", "parent_id": "p1"},
                ],
            }
        return {"ids": [], "documents": [], "metadatas": []}

    def parent_get_side_effect(ids=None, where=None):
        return {
            "ids": ["p1"],
            "documents": ["parent content"],
            "metadatas": [{"chunk_type": "parent", "parent_id": "p1", "child_ids": ["c1", "c2"]}],
        }

    mock_material.get.side_effect = material_get_side_effect
    mock_parent.get.side_effect = parent_get_side_effect

    with patch("mem_retrieve_material.material_store", mock_material), \
         patch("mem_retrieve_material.parent_store", mock_parent):
        state: material_state = {
            "query": "测试",
            "plan": [material_retrieve_step(step_id=1, mode="parent_lookup", target="parents")],
            "current_step_idx": 0,
            "results": [
                MaterialResult(
                    step_id=1,
                    matched_children=[Document(page_content="c1", id="c1"), Document(page_content="c2", id="c2")],
                    parent_contexts=[],
                    mode="semantic_search",
                )
            ],
            "matched_child_ids": ["c1", "c2"],
            "matched_parent_ids": [],
        }

        result_state = await parent_lookup_node(state)

        assert "p1" in result_state["matched_parent_ids"]
        assert len(result_state["results"][-1].parent_contexts) == 1


@pytest.mark.asyncio
async def test_parent_lookup_no_children():
    """Test parent_lookup does nothing when no children matched."""
    with patch("mem_retrieve_material.material_store"), \
         patch("mem_retrieve_material.parent_store"):
        state: material_state = {
            "query": "测试",
            "plan": [material_retrieve_step(step_id=1, mode="parent_lookup", target="parents")],
            "current_step_idx": 0,
            "results": [],
            "matched_child_ids": [],  # No children
            "matched_parent_ids": [],
        }

        result_state = await parent_lookup_node(state)

        assert len(result_state["results"]) == 0


# ---------- Tests for children_lookup_node ----------
@pytest.mark.asyncio
async def test_children_lookup_fetches_child_docs():
    """Test children_lookup fetches children from parent child_ids."""
    mock_material = MagicMock()
    mock_parent = MagicMock()

    def parent_get_side_effect(ids=None, where=None):
        return {
            "ids": ["p1"],
            "documents": ["parent content"],
            "metadatas": [{"chunk_type": "parent", "child_ids": ["c1", "c2"]}],
        }

    def material_get_side_effect(ids=None, where=None):
        return {
            "ids": ["c1", "c2"],
            "documents": ["child1", "child2"],
            "metadatas": [
                {"chunk_type": "child", "parent_id": "p1"},
                {"chunk_type": "child", "parent_id": "p1"},
            ],
        }

    mock_parent.get.side_effect = parent_get_side_effect
    mock_material.get.side_effect = material_get_side_effect

    with patch("mem_retrieve_material.material_store", mock_material), \
         patch("mem_retrieve_material.parent_store", mock_parent):
        state: material_state = {
            "query": "测试",
            "plan": [material_retrieve_step(step_id=1, mode="children_lookup", target="children")],
            "current_step_idx": 0,
            "results": [],
            "matched_child_ids": [],
            "matched_parent_ids": ["p1"],
        }

        result_state = await children_lookup_node(state)

        assert len(result_state["results"]) == 1
        assert len(result_state["matched_child_ids"]) == 2


# ---------- Tests for plan_node ----------
@pytest.mark.asyncio
async def test_plan_node_fallback():
    """Test plan_node falls back to default plan on error."""
    with patch("mem_retrieve_material.ChatDeepSeek") as mock_llm:
        mock_instance = MagicMock()
        mock_llm.return_value = mock_instance
        structured_mock = MagicMock()
        mock_instance.with_structured_output.return_value = structured_mock
        structured_mock.ainvoke = AsyncMock(side_effect=Exception("LLM error"))

        state: material_state = {
            "query": "测试查询",
            "plan": [],
            "current_step_idx": 0,
            "results": [],
            "matched_child_ids": [],
            "matched_parent_ids": [],
        }

        result_state = await plan_node(state)

        assert len(result_state["plan"]) == 2
        assert result_state["plan"][0].mode == "semantic_search"
        assert result_state["plan"][1].mode == "parent_lookup"


# ---------- Tests for build_material_graph ----------
def test_build_material_graph_compiles():
    """Test that the graph builds without errors."""
    graph = build_material_graph()
    assert graph is not None
