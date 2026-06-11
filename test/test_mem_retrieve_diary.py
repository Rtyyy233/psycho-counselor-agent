# test/test_mem_retrieve_diary.py
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mem_retrieve_diary import (
    retrieve_filter,
    retrieve_step,
    RetrievalResult,
    agent_state,
    plan_node,
    metadata_filter_node,
    semantic_search_node,
    id_lookup_node,
    rerank_node,
    _build_where_clause,
    build_retrieve_graph,
    retrieve_diary,
)
from langchain_core.documents import Document


# ---------- Tests for _build_where_clause ----------
def test_build_where_clause_empty():
    """Test where clause with no filters."""
    flt = retrieve_filter()
    where = _build_where_clause(flt)
    assert where == {}


def test_build_where_clause_intensity():
    """Test where clause with intensity filter."""
    flt = retrieve_filter(intensity="强")
    where = _build_where_clause(flt)
    assert where == {"intensity": "强"}


def test_build_where_clause_date_range():
    """Test where clause with date range."""
    flt = retrieve_filter(date_start="25.01.01", date_end="25.03.15")
    where = _build_where_clause(flt)
    assert where == {"date": {"$gte": "25.01.01", "$lte": "25.03.15"}}


def test_build_where_clause_scene_type():
    """Test where clause with scene_type."""
    flt = retrieve_filter(scene_type="工作")
    where = _build_where_clause(flt)
    assert where == {"scene_type": "工作"}


def test_build_where_clause_event_type():
    """Test where clause with event_type."""
    flt = retrieve_filter(event_type="冲突")
    where = _build_where_clause(flt)
    assert where == {"event_type": "冲突"}


def test_build_where_clause_emotion():
    """Test where clause with emotion filter."""
    flt = retrieve_filter(emotion=["焦虑", "恐惧"])
    where = _build_where_clause(flt)
    assert where == {"emotion": {"$in": ["焦虑", "恐惧"]}}


def test_build_where_clause_combined():
    """Test where clause with multiple filters."""
    flt = retrieve_filter(
        intensity="强",
        date_start="25.01.01",
        scene_type="工作",
        emotion=["焦虑"],
    )
    where = _build_where_clause(flt)
    assert where["intensity"] == "强"
    assert where["date"] == {"$gte": "25.01.01"}
    assert where["scene_type"] == "工作"
    assert where["emotion"] == {"$in": ["焦虑"]}


# ---------- Tests for metadata_filter_node ----------
@pytest.mark.asyncio
async def test_metadata_filter_node_executes():
    """Test metadata_filter_node executes filter correctly."""
    mock_diary = MagicMock()
    mock_diary.get.return_value = {
        "ids": ["id1", "id2"],
        "documents": [
            Document(page_content="doc1", metadata={"emotion": ["焦虑"]}),
            Document(page_content="doc2", metadata={"emotion": ["悲伤"]}),
        ],
        "metadatas": [{"emotion": ["焦虑"]}, {"emotion": ["悲伤"]}],
    }

    with patch("mem_retrieve_diary.diary_annotation", mock_diary):
        sample_state: agent_state = {
            "query": "最近焦虑的日记",
            "retrieve_plan": [
                retrieve_step(
                    step_id=1,
                    target_collection="diary_annotation",
                    mode="metadata_filter",
                    filter=retrieve_filter(emotion=["焦虑"]),
                    input_source="query",
                )
            ],
            "current_step_idx": 0,
            "results": [],
            "previous_ids": [],
            "previous_texts": [],
        }

        result_state = await metadata_filter_node(sample_state)

        assert len(result_state["results"]) == 1
        assert result_state["results"][0].mode == "metadata_filter"
        assert result_state["results"][0].collection == "diary_annotation"
        assert len(result_state["results"][0].ids) == 2


@pytest.mark.asyncio
async def test_metadata_filter_node_skips_wrong_mode():
    """Test metadata_filter_node skips when mode is not metadata_filter."""
    with patch("mem_retrieve_diary.diary_annotation") as mock_diary:
        sample_state: agent_state = {
            "query": "测试",
            "retrieve_plan": [
                retrieve_step(
                    step_id=1,
                    target_collection="diary_annotation",
                    mode="semantic_search",  # Wrong mode
                    input_source="query",
                )
            ],
            "current_step_idx": 0,
            "results": [],
            "previous_ids": [],
            "previous_texts": [],
        }

        result_state = await metadata_filter_node(sample_state)

        assert len(result_state["results"]) == 0


# ---------- Tests for semantic_search_node ----------
@pytest.mark.asyncio
async def test_semantic_search_node_annotation():
    """Test semantic_search_node searches diary_annotation."""
    mock_docs = [
        (Document(page_content="result1", id="r1"), 0.9),
        (Document(page_content="result2", id="r2"), 0.8),
    ]

    with patch("mem_retrieve_diary.diary_annotation") as mock_diary:
        mock_diary.similarity_search_with_score.return_value = mock_docs

        sample_state: agent_state = {
            "query": "焦虑情绪",
            "retrieve_plan": [
                retrieve_step(
                    step_id=1,
                    target_collection="diary_annotation",
                    mode="semantic_search",
                    temp_query="焦虑",
                    input_source="query",
                )
            ],
            "current_step_idx": 0,
            "results": [],
            "previous_ids": [],
            "previous_texts": [],
        }

        result_state = await semantic_search_node(sample_state)

        assert len(result_state["results"]) == 1
        assert result_state["results"][0].mode == "semantic_search"
        assert result_state["results"][0].collection == "diary_annotation"
        mock_diary.similarity_search_with_score.assert_called_once_with("焦虑", k=10)


@pytest.mark.asyncio
async def test_semantic_search_node_original():
    """Test semantic_search_node searches original_diary."""
    mock_docs = [
        (Document(page_content="original1", id="o1"), 0.85),
    ]

    with patch("mem_retrieve_diary.original_diary") as mock_orig:
        mock_orig.similarity_search_with_score.return_value = mock_docs

        sample_state: agent_state = {
            "query": "工作压力",
            "retrieve_plan": [
                retrieve_step(
                    step_id=1,
                    target_collection="original_diary",
                    mode="semantic_search",
                    temp_query="工作",
                    input_source="query",
                )
            ],
            "current_step_idx": 0,
            "results": [],
            "previous_ids": [],
            "previous_texts": [],
        }

        result_state = await semantic_search_node(sample_state)

        assert len(result_state["results"]) == 1
        assert result_state["results"][0].collection == "original_diary"


# ---------- Tests for id_lookup_node ----------
@pytest.mark.asyncio
async def test_id_lookup_node():
    """Test id_lookup_node fetches by ID."""
    with patch("mem_retrieve_diary.diary_annotation") as mock_diary:
        mock_diary.get.return_value = {
            "documents": ["annotated1", "annotated2"],
            "metadatas": [{"emotion": ["焦虑"]}, {"emotion": ["悲伤"]}],
            "ids": ["id1", "id2"]
        }

        sample_state: agent_state = {
            "query": "测试",
            "retrieve_plan": [
                retrieve_step(
                    step_id=1,
                    target_collection="diary_annotation",
                    mode="id_lookup",
                    input_source="previous_step",
                )
            ],
            "current_step_idx": 0,
            "results": [],
            "previous_ids": ["id1", "id2"],
            "previous_texts": [],
        }

        result_state = await id_lookup_node(sample_state)

        assert len(result_state["results"]) == 1
        assert result_state["results"][0].mode == "id_lookup"
        mock_diary.get.assert_called_once_with(ids=["id1", "id2"])


@pytest.mark.asyncio
async def test_id_lookup_node_no_ids():
    """Test id_lookup_node does nothing when no IDs available."""
    with patch("mem_retrieve_diary.diary_annotation") as mock_diary:
        sample_state: agent_state = {
            "query": "测试",
            "retrieve_plan": [
                retrieve_step(
                    step_id=1,
                    target_collection="diary_annotation",
                    mode="id_lookup",
                    input_source="previous_step",
                )
            ],
            "current_step_idx": 0,
            "results": [],
            "previous_ids": [],
            "previous_texts": [],
        }

        result_state = await id_lookup_node(sample_state)

        assert len(result_state["results"]) == 0
        mock_diary.get.assert_not_called()


# ---------- Tests for rerank_node ----------
@pytest.mark.asyncio
async def test_rerank_node():
    """Test rerank_node reorders results by metadata scoring."""
    with patch("mem_retrieve_diary.diary_annotation") as mock_diary:
        mock_diary.get.return_value = {
            "documents": ["doc_焦虑", "doc_平静"],
            "metadatas": [
                {"emotion": ["焦虑"], "intensity": "强", "event_type": "冲突"},
                {"emotion": ["平静"], "intensity": "弱", "event_type": "日常"},
            ],
            "ids": ["id1", "id2"]
        }

        sample_state: agent_state = {
            "query": "焦虑情绪",  # Should boost "焦虑" entries
            "retrieve_plan": [
                retrieve_step(
                    step_id=1,
                    target_collection="diary_annotation",
                    mode="rerank",
                    input_source="previous_step",
                )
            ],
            "current_step_idx": 0,
            "results": [],
            "previous_ids": ["id1", "id2"],
            "previous_texts": [],
        }

        result_state = await rerank_node(sample_state)

        assert len(result_state["results"]) == 1
        assert result_state["results"][0].mode == "rerank"
        # First doc should be the one matching "焦虑"
        assert result_state["results"][0].documents[0].page_content == "doc_焦虑"


@pytest.mark.asyncio
async def test_rerank_node_no_ids():
    """Test rerank_node does nothing when no previous IDs."""
    with patch("mem_retrieve_diary.diary_annotation") as mock_diary:
        sample_state: agent_state = {
            "query": "测试",
            "retrieve_plan": [
                retrieve_step(
                    step_id=1,
                    target_collection="diary_annotation",
                    mode="rerank",
                    input_source="previous_step",
                )
            ],
            "current_step_idx": 0,
            "results": [],
            "previous_ids": [],
            "previous_texts": [],
        }

        result_state = await rerank_node(sample_state)

        assert len(result_state["results"]) == 0


# ---------- Tests for plan_node ----------
@pytest.mark.asyncio
async def test_plan_node_fallback():
    """Test plan_node falls back to semantic search on error."""
    with patch("mem_retrieve_diary.ChatDeepSeek") as mock_llm:
        mock_instance = MagicMock()
        mock_llm.return_value = mock_instance
        structured_mock = MagicMock()
        mock_instance.with_structured_output.return_value = structured_mock
        structured_mock.ainvoke = AsyncMock(side_effect=Exception("LLM error"))

        sample_state: agent_state = {
            "query": "最近焦虑的日记",
            "retrieve_plan": [],
            "current_step_idx": 0,
            "results": [],
            "previous_ids": [],
            "previous_texts": [],
        }

        result_state = await plan_node(sample_state)

        assert len(result_state["retrieve_plan"]) == 1
        assert result_state["retrieve_plan"][0].mode == "semantic_search"


# ---------- Tests for build_retrieve_graph ----------
def test_build_retrieve_graph_compiles():
    """Test that the graph builds without errors."""
    graph = build_retrieve_graph()
    assert graph is not None
