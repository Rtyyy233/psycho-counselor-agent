from mem_collections import conv_store
from operator import add
from typing import TypedDict, List, Literal, Optional, Annotated
import asyncio
from langchain_deepseek import ChatDeepSeek
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool


class conv_filter(BaseModel):
    """Filter for conversation outline search."""

    section: Optional[
        Literal["problem", "assessment", "intervention", "plan", "raw"]
    ] = Field(None, description="PAIP 部分或原文")
    text_type: Optional[Literal["paip_summary", "conversation"]] = Field(
        None, description="内容类型"
    )
    date_start: Optional[str] = Field(None, description="开始日期 YYYY-MM-DD")
    date_end: Optional[str] = Field(None, description="结束日期 YYYY-MM-DD")
    source_file: Optional[str] = Field(None, description="来源文件")


class conv_retrieve_step(BaseModel):
    step_id: int = Field(description="步骤编号，从1开始")
    mode: Literal["semantic_search", "metadata_filter", "paip_outline_lookup"]
    filter: Optional[conv_filter] = Field(None, description="过滤条件")
    temp_query: Optional[str] = Field(None, description="语义搜索查询语句")


class PAIPResult(BaseModel):
    """Single PAIP section result."""

    section: str
    content: str
    base_id: str
    metadata: dict


class ConvRetrievalResult(BaseModel):
    """Result from conversation outline retrieval."""

    step_id: int
    matched_docs: List[Document]
    paip_outlines: List[PAIPResult]  # Reconstructed PAIP outlines for matched base_ids
    mode: str


class conv_state(TypedDict):
    query: str
    plan: List[conv_retrieve_step]
    current_step_idx: int
    results: List[ConvRetrievalResult]
    matched_base_ids: Annotated[List[str], add]
    all_sections_for_base: dict  # base_id -> {problem, assessment, intervention, plan}


def _build_where_clause(flt: conv_filter) -> dict:
    """Build Chroma where clause from conv_filter."""
    where = {}
    if flt.section:
        where["section"] = flt.section
    if flt.text_type:
        where["text_type"] = flt.text_type
    if flt.date_start:
        where.setdefault("date", {})["$gte"] = flt.date_start
    if flt.date_end:
        where.setdefault("date", {})["$lte"] = flt.date_end
    if flt.source_file:
        where["source"] = flt.source_file
    return where


async def semantic_search_node(state: conv_state) -> conv_state:
    """
    Semantic search on conv_outline collection.
    Searches both raw conversation chunks and PAIP summaries.
    """
    step = state["plan"][state["current_step_idx"]]
    if step.mode != "semantic_search":
        return state

    query = step.temp_query or state["query"]

    def _sync():
        return conv_store.similarity_search_with_score(query, k=20)

    results = await asyncio.get_running_loop().run_in_executor(None, _sync)

    docs = [r[0] for r in results]
    base_ids = set()
    for doc in docs:
        if doc.metadata.get("base_id"):
            base_ids.add(doc.metadata["base_id"])

    state["matched_base_ids"].extend(list(base_ids))

    result = ConvRetrievalResult(
        step_id=step.step_id,
        matched_docs=docs,
        paip_outlines=[],
        mode="semantic_search",
    )
    state["results"].append(result)
    return state


async def metadata_filter_node(state: conv_state) -> conv_state:
    """
    Filter by metadata on conv_outline.
    Can filter by section, text_type, date range, source.
    """
    step = state["plan"][state["current_step_idx"]]
    if step.mode != "metadata_filter" or step.filter is None:
        return state

    where = _build_where_clause(step.filter)

    def _sync():
        return conv_store.get(where=where)

    result_data = await asyncio.get_running_loop().run_in_executor(None, _sync)

    docs = [
        Document(page_content=pc, metadata=m, id=i)
        for pc, m, i in zip(
            result_data.get("documents", []),
            result_data.get("metadatas", []),
            result_data.get("ids", []),
        )
    ]

    base_ids = set()
    for doc in docs:
        if doc.metadata.get("base_id"):
            base_ids.add(doc.metadata["base_id"])

    state["matched_base_ids"].extend(list(base_ids))

    result = ConvRetrievalResult(
        step_id=step.step_id,
        matched_docs=docs,
        paip_outlines=[],
        mode="metadata_filter",
    )
    state["results"].append(result)
    return state


async def paip_outline_lookup_node(state: conv_state) -> conv_state:
    """
    Given base_ids, reconstruct full PAIP outlines.
    Fetches all 4 PAIP sections for each matched base_id.
    """
    step = state["plan"][state["current_step_idx"]]
    if step.mode != "paip_outline_lookup":
        return state

    base_ids = state["matched_base_ids"]
    if not base_ids:
        return state

    # Fetch all PAIP sections for these base_ids
    section_names = ["problem", "assessment", "intervention", "plan"]

    def _sync():
        where = {"text_type": "paip_summary", "base_id": {"$in": base_ids}}
        return conv_store.get(where=where)

    paip_data = await asyncio.get_running_loop().run_in_executor(None, _sync)

    # Organize by base_id
    outlines_by_base: dict = {bid: {} for bid in base_ids}
    for doc_id, pc, metadata in zip(
        paip_data.get("ids", []),
        paip_data.get("documents", []),
        paip_data.get("metadatas", []),
    ):
        bid = metadata.get("base_id")
        section = metadata.get("section")
        if bid and section in section_names:
            outlines_by_base[bid][section] = pc

    # Build PAIPResult list
    paip_results = []
    for bid, sections in outlines_by_base.items():
        for sec_name, sec_content in sections.items():
            paip_results.append(
                PAIPResult(
                    section=sec_name,
                    content=sec_content,
                    base_id=bid,
                    metadata=sections,
                )
            )

    # Update last result with PAIP outlines
    if state["results"]:
        state["results"][-1].paip_outlines = paip_results
        state["results"][-1].mode = "paip_outline_lookup"
    else:
        # No previous result, create one
        result = ConvRetrievalResult(
            step_id=step.step_id,
            matched_docs=[],
            paip_outlines=paip_results,
            mode="paip_outline_lookup",
        )
        state["results"].append(result)

    # Store for access
    state["all_sections_for_base"] = outlines_by_base
    return state


async def plan_node(state: conv_state) -> conv_state:
    """
    Plan the retrieval strategy based on query.
    Default: semantic search → paip_outline_lookup (get full PAIP structure).
    """
    plan_model = ChatDeepSeek(model="deepseek-chat", temperature=0.3)
    planner = plan_model.with_structured_output(conv_retrieve_step, mode="json")

    system_prompt = """你是一个咨询对话摘要检索规划专家。制定检索计划。

可用模式：
1. semantic_search - 语义搜索（支持原始对话和PAIP摘要）
2. metadata_filter - 元数据过滤（按section/text_type/date/source）
3. paip_outline_lookup - 根据base_id获取完整PAIP大纲

默认策略（推荐）：
1. semantic_search（语义搜索相关对话或摘要）
2. paip_outline_lookup（获取完整PAIP结构）

请生成1-2步的检索计划。"""

    try:
        steps = await planner.ainvoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"查询: {state['query']}"},
            ]
        )
        if isinstance(steps, conv_retrieve_step):
            plan = [steps]
        elif isinstance(steps, list):
            plan = steps
        else:
            plan = [
                conv_retrieve_step(**s) if isinstance(s, dict) else s  # type:ignore
                for s in steps
            ]
    except Exception:
        # Default: semantic search then paip lookup
        plan = [
            conv_retrieve_step(
                step_id=1,
                mode="semantic_search",
                temp_query=state["query"],
            ),  # type:ignore
            conv_retrieve_step(
                step_id=2,
                mode="paip_outline_lookup",
            ),  # type:ignore
        ]

    for i, step in enumerate(plan):
        step.step_id = i + 1  # type:ignore

    state["plan"] = plan  # type:ignore
    state["current_step_idx"] = 0
    return state


def route_dispatch(
    state: conv_state,
) -> Literal[
    "semantic_search_node",
    "metadata_filter_node",
    "paip_outline_lookup_node",
    "__end__",
]:
    """Route to appropriate node based on step mode."""
    if state["current_step_idx"] >= len(state["plan"]):
        print(
            f"DEBUG route_dispatch: current_step_idx {state['current_step_idx']} >= plan length {len(state['plan'])}, returning '__end__'"
        )
        return "__end__"
    step = state["plan"][state["current_step_idx"]]
    result = {
        "semantic_search": "semantic_search_node",
        "metadata_filter": "metadata_filter_node",
        "paip_outline_lookup": "paip_outline_lookup_node",
    }.get(step.mode, "__end__")  # type:ignore
    print(f"DEBUG route_dispatch: step.mode={step.mode}, returning '{result}'")
    return result


def first_route(
    state: conv_state,
) -> Literal[
    "semantic_search_node",
    "metadata_filter_node",
    "paip_outline_lookup_node",
    "__end__",
]:
    """Route from planner to first execution node."""
    return route_dispatch(state)


def after_execution(
    state: conv_state,
) -> Literal[
    "semantic_search_node",
    "metadata_filter_node",
    "paip_outline_lookup_node",
    "__end__",
]:
    """Advance to next step or end."""
    state["current_step_idx"] += 1
    print(
        f"DEBUG after_execution: incremented current_step_idx to {state['current_step_idx']}, plan length={len(state['plan'])}"
    )
    if state["current_step_idx"] >= len(state["plan"]):
        print(f"DEBUG after_execution: returning '__end__'")
        return "__end__"
    result = route_dispatch(state)
    print(f"DEBUG after_execution: returning '{result}'")
    return result


def build_conv_retrieve_graph():
    """Build LangGraph state machine for conversation outline retrieval."""
    graph = StateGraph(conv_state)

    graph.add_node("planner", plan_node)
    graph.add_node("semantic_search_node", semantic_search_node)
    graph.add_node("metadata_filter_node", metadata_filter_node)
    graph.add_node("paip_outline_lookup_node", paip_outline_lookup_node)

    graph.set_entry_point("planner")
    graph.add_conditional_edges("planner", first_route)
    graph.add_conditional_edges("semantic_search_node", after_execution)
    graph.add_conditional_edges("metadata_filter_node", after_execution)
    graph.add_conditional_edges("paip_outline_lookup_node", after_execution)

    return graph.compile()


_conv_graph = None


def get_conv_retrieve_graph():
    global _conv_graph
    # Force rebuild for debugging - remove after fix is confirmed
    _conv_graph = None
    if _conv_graph is None:
        print("DEBUG: Building new conversation retrieval graph")
        _conv_graph = build_conv_retrieve_graph()
    return _conv_graph


@tool
async def retrieve_conv_outline(query: str) -> List[ConvRetrievalResult]:
    """
    Main entry point for conversation outline retrieval.

    Returns:
        List of ConvRetrievalResult containing matched docs and reconstructed PAIP outlines.
    """
    graph = get_conv_retrieve_graph()

    initial_state: conv_state = {
        "query": query,
        "plan": [],
        "current_step_idx": 0,
        "results": [],
        "matched_base_ids": [],
        "all_sections_for_base": {},
    }

    final_state = await graph.ainvoke(initial_state)
    return final_state["results"]
