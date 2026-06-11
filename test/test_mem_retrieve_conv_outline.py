# test/test_mem_retrieve_conv_outline.py
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mem_retrieve_conv_outline import (
    conv_filter,
    conv_retrieve_step,
    ConvRetrievalResult,
    PAIPResult,
    conv_state,
    plan_node,
    semantic_search_node,
    metadata_filter_node,
    paip_outline_lookup_node,
    _build_where_clause,
    build_conv_retrieve_graph,
    retrieve_conv_outline,
)
from langchain_core.documents import Document


# ---------- Tests for _build_where_clause ----------
def test_build_where_clause_empty():
    """Test where clause with no filters."""
    flt = conv_filter()
    where = _build_where_clause(flt)
    assert where == {}


def test_build_where_clause_section():
    """Test where clause with section filter."""
    flt = conv_filter(section="problem")
    where = _build_where_clause(flt)
    assert where == {"section": "problem"}


def test_build_where_clause_text_type():
    """Test where clause with text_type filter."""
    flt = conv_filter(text_type="paip_summary")
    where = _build_where_clause(flt)
    assert where == {"text_type": "paip_summary"}


def test_build_where_clause_date_range():
    """Test where clause with date range."""
    flt = conv_filter(date_start="2025-01-01", date_end="2025-03-15")
    where = _build_where_clause(flt)
    assert where == {"date": {"$gte": "2025-01-01", "$lte": "2025-03-15"}}


def test_build_where_clause_source_file():
    """Test where clause with source_file filter."""
    flt = conv_filter(source_file="counseling_001.txt")
    where = _build_where_clause(flt)
    assert where == {"source": "counseling_001.txt"}


def test_build_where_clause_combined():
    """Test where clause with multiple filters."""
    flt = conv_filter(
        section="problem",
        text_type="paip_summary",
        date_start="2025-01-01",
    )
    where = _build_where_clause(flt)
    assert where["section"] == "problem"
    assert where["text_type"] == "paip_summary"
    assert where["date"] == {"$gte": "2025-01-01"}


# ---------- Tests for semantic_search_node ----------
@pytest.mark.asyncio
async def test_semantic_search_node_executes():
    """Test semantic_search_node performs similarity search."""
    mock_docs = [
        (Document(page_content="来访者谈到了焦虑问题", id="conv_id1", metadata={"base_id": "conv_20250101_001"}), 0.9),
        (Document(page_content="本次咨询聚焦抑郁情绪", id="conv_id2", metadata={"base_id": "conv_20250102_001"}), 0.85),
    ]

    with patch("mem_retrieve_conv_outline.conv_store") as mock_store:
        mock_store.similarity_search_with_score.return_value = mock_docs

        sample_state: conv_state = {
            "query": "焦虑情绪",
            "plan": [
                conv_retrieve_step(
                    step_id=1,
                    mode="semantic_search",
                    temp_query="焦虑",
                )
            ],
            "current_step_idx": 0,
            "results": [],
            "matched_base_ids": [],
            "all_sections_for_base": {},
        }

        result_state = await semantic_search_node(sample_state)

        assert len(result_state["results"]) == 1
        assert result_state["results"][0].mode == "semantic_search"
        mock_store.similarity_search_with_score.assert_called_once_with("焦虑", k=20)


@pytest.mark.asyncio
async def test_semantic_search_node_skips_wrong_mode():
    """Test semantic_search_node skips when mode is not semantic_search."""
    with patch("mem_retrieve_conv_outline.conv_store") as mock_store:
        sample_state: conv_state = {
            "query": "测试",
            "plan": [
                conv_retrieve_step(
                    step_id=1,
                    mode="metadata_filter",
                )
            ],
            "current_step_idx": 0,
            "results": [],
            "matched_base_ids": [],
            "all_sections_for_base": {},
        }

        result_state = await semantic_search_node(sample_state)

        assert len(result_state["results"]) == 0
        mock_store.similarity_search_with_score.assert_not_called()


# ---------- Tests for metadata_filter_node ----------
@pytest.mark.asyncio
async def test_metadata_filter_node_executes():
    """Test metadata_filter_node executes filter correctly."""
    mock_store = MagicMock()
    mock_store.get.return_value = {
        "ids": ["conv_id1", "conv_id2"],
        "documents": [
            "problem content 1",
            "problem content 2",
        ],
        "metadatas": [
            {"base_id": "conv_20250101_001", "section": "problem"},
            {"base_id": "conv_20250102_001", "section": "problem"},
        ],
    }

    with patch("mem_retrieve_conv_outline.conv_store", mock_store):
        sample_state: conv_state = {
            "query": "测试",
            "plan": [
                conv_retrieve_step(
                    step_id=1,
                    mode="metadata_filter",
                    filter=conv_filter(section="problem"),
                )
            ],
            "current_step_idx": 0,
            "results": [],
            "matched_base_ids": [],
            "all_sections_for_base": {},
        }

        result_state = await metadata_filter_node(sample_state)

        assert len(result_state["results"]) == 1
        assert result_state["results"][0].mode == "metadata_filter"
        assert len(result_state["results"][0].matched_docs) == 2


@pytest.mark.asyncio
async def test_metadata_filter_node_skips_without_filter():
    """Test metadata_filter_node skips when no filter provided."""
    with patch("mem_retrieve_conv_outline.conv_store") as mock_store:
        sample_state: conv_state = {
            "query": "测试",
            "plan": [
                conv_retrieve_step(
                    step_id=1,
                    mode="metadata_filter",
                    filter=None,
                )
            ],
            "current_step_idx": 0,
            "results": [],
            "matched_base_ids": [],
            "all_sections_for_base": {},
        }

        result_state = await metadata_filter_node(sample_state)

        assert len(result_state["results"]) == 0


# ---------- Tests for paip_outline_lookup_node ----------
@pytest.mark.asyncio
async def test_paip_outline_lookup_node():
    """Test paip_outline_lookup_node reconstructs PAIP outlines."""
    mock_store = MagicMock()
    mock_store.get.return_value = {
        "ids": ["conv_20250101_001_problem", "conv_20250101_001_assessment"],
        "documents": ["来访者主诉焦虑", "焦虑程度中等"],
        "metadatas": [
            {"base_id": "conv_20250101_001", "section": "problem"},
            {"base_id": "conv_20250101_001", "section": "assessment"},
        ],
    }

    with patch("mem_retrieve_conv_outline.conv_store", mock_store):
        sample_state: conv_state = {
            "query": "焦虑",
            "plan": [
                conv_retrieve_step(
                    step_id=1,
                    mode="paip_outline_lookup",
                )
            ],
            "current_step_idx": 0,
            "results": [
                ConvRetrievalResult(
                    step_id=1,
                    matched_docs=[],
                    paip_outlines=[],
                    mode="semantic_search",
                )
            ],
            "matched_base_ids": ["conv_20250101_001"],
            "all_sections_for_base": {},
        }

        result_state = await paip_outline_lookup_node(sample_state)

        assert len(result_state["results"]) == 1
        assert result_state["results"][0].mode == "paip_outline_lookup"
        assert len(result_state["results"][0].paip_outlines) == 2


@pytest.mark.asyncio
async def test_paip_outline_lookup_node_no_base_ids():
    """Test paip_outline_lookup_node does nothing when no base_ids."""
    with patch("mem_retrieve_conv_outline.conv_store") as mock_store:
        sample_state: conv_state = {
            "query": "测试",
            "plan": [
                conv_retrieve_step(
                    step_id=1,
                    mode="paip_outline_lookup",
                )
            ],
            "current_step_idx": 0,
            "results": [],
            "matched_base_ids": [],
            "all_sections_for_base": {},
        }

        result_state = await paip_outline_lookup_node(sample_state)

        assert len(result_state["results"]) == 0
        mock_store.get.assert_not_called()


@pytest.mark.asyncio
async def test_paip_outline_lookup_node_skips_wrong_mode():
    """Test paip_outline_lookup_node skips when mode is not paip_outline_lookup."""
    with patch("mem_retrieve_conv_outline.conv_store") as mock_store:
        sample_state: conv_state = {
            "query": "测试",
            "plan": [
                conv_retrieve_step(
                    step_id=1,
                    mode="semantic_search",
                )
            ],
            "current_step_idx": 0,
            "results": [],
            "matched_base_ids": [],
            "all_sections_for_base": {},
        }

        result_state = await paip_outline_lookup_node(sample_state)

        assert len(result_state["results"]) == 0


# ---------- Tests for plan_node ----------
@pytest.mark.asyncio
async def test_plan_node_fallback():
    """Test plan_node falls back to semantic search on error."""
    with patch("mem_retrieve_conv_outline.ChatDeepSeek") as mock_llm:
        mock_instance = MagicMock()
        mock_llm.return_value = mock_instance
        structured_mock = MagicMock()
        mock_instance.with_structured_output.return_value = structured_mock
        structured_mock.ainvoke = AsyncMock(side_effect=Exception("LLM error"))

        sample_state: conv_state = {
            "query": "焦虑相关的咨询记录",
            "plan": [],
            "current_step_idx": 0,
            "results": [],
            "matched_base_ids": [],
            "all_sections_for_base": {},
        }

        result_state = await plan_node(sample_state)

        assert len(result_state["plan"]) == 2
        assert result_state["plan"][0].mode == "semantic_search"
        assert result_state["plan"][1].mode == "paip_outline_lookup"


# ---------- Tests for build_conv_retrieve_graph ----------
def test_build_conv_retrieve_graph_compiles():
    """Test that the graph builds without errors."""
    graph = build_conv_retrieve_graph()
    assert graph is not None
