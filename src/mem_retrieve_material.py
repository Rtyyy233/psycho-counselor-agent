import os
from mem_integration import material_store, parent_store
from operator import add
from typing import TypedDict, List, Literal, Optional, Annotated
import asyncio
import logging
from langchain_deepseek import ChatDeepSeek
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from config import LLM_MODEL

logger = logging.getLogger(__name__)


class material_filter(BaseModel):
    """Filter for material search."""
    text_type: Optional[List[str]] = Field(None, description="材料类型")
    date_start: Optional[str] = Field(None, description="开始日期 YYYY-MM-DD")
    date_end: Optional[str] = Field(None, description="结束日期 YYYY-MM-DD")
    source_file: Optional[str] = Field(None, description="来源文件")


class material_retrieve_step(BaseModel):
    step_id: int = Field(description="步骤编号，从1开始")
    mode: Literal["semantic_search", "metadata_filter", "parent_lookup", "children_lookup"]
    target: Literal["children", "parents", "both"] = Field(description="检索目标")
    filter: Optional[material_filter] = Field(None, description="过滤条件")
    temp_query: Optional[str] = Field(None, description="语义搜索查询语句")


class MaterialResult(BaseModel):
    """Result from material retrieval."""
    step_id: int
    matched_children: List[Document]
    parent_contexts: List[Document]  # Full parent documents for matched children
    mode: str


class material_state(TypedDict):
    query: str
    plan: List[material_retrieve_step]
    current_step_idx: int
    results: List[MaterialResult]
    matched_child_ids: Annotated[List[str], add]
    matched_parent_ids: Annotated[List[str], add]


def _build_child_where_clause(flt: material_filter) -> dict:
    """Build Chroma where clause for child collection."""
    where = {}
    if flt.text_type:
        where["text_type"] = {"$in": flt.text_type}
    if flt.date_start:
        where.setdefault("date", {})["$gte"] = flt.date_start
    if flt.date_end:
        where.setdefault("date", {})["$lte"] = flt.date_end
    if flt.source_file:
        where["source_file"] = flt.source_file
    return where


async def semantic_search_children(state: material_state) -> material_state:
    """
    Semantic search on child chunks (small chunks for precision).
    Searches material_store (child_chunks collection).
    """
    step = state["plan"][state["current_step_idx"]]
    if step.mode != "semantic_search":
        return state

    query = step.temp_query or state["query"]

    def _sync():
        return material_store.similarity_search_with_score(query, k=20)

    results = await asyncio.get_running_loop().run_in_executor(None, _sync)

    child_docs = [r[0] for r in results]
    state["matched_child_ids"].extend([d.id for d in child_docs]) #type:ignore

    result = MaterialResult(
        step_id=step.step_id,
        matched_children=child_docs,
        parent_contexts=[],
        mode="semantic_search",
    )
    state["results"].append(result)
    state["current_step_idx"] += 1
    return state


async def metadata_filter_node(state: material_state) -> material_state:
    """
    Filter by metadata on child chunks.
    Searches material_store (child_chunks collection).
    """
    step = state["plan"][state["current_step_idx"]]
    if step.mode != "metadata_filter" or step.filter is None:
        return state

    where = _build_child_where_clause(step.filter)

    def _sync():
        return material_store.get(where=where)

    result_data = await asyncio.get_running_loop().run_in_executor(None, _sync)

    docs = [
        Document(page_content=pc, metadata=m, id=i)
        for pc, m, i in zip(
            result_data.get("documents", []),
            result_data.get("metadatas", []),
            result_data.get("ids", []),
        )
    ]

    state["matched_child_ids"].extend([d.id for d in docs]) #type:ignore

    result = MaterialResult(
        step_id=step.step_id,
        matched_children=docs,
        parent_contexts=[],
        mode="metadata_filter",
    )
    state["results"].append(result)
    state["current_step_idx"] += 1
    return state


async def parent_lookup_node(state: material_state) -> material_state:
    """
    Small-to-big: Given matched child IDs, fetch their parent documents (big context).
    1. Get parent_ids from matched children (material_store)
    2. Fetch parent documents (parent_store)
    """
    step = state["plan"][state["current_step_idx"]]
    if step.mode != "parent_lookup":
        return state

    child_ids = state["matched_child_ids"]
    if not child_ids:
        return state

    # Step 1: Get parent_ids from children
    def _sync_get_children():
        return material_store.get(ids=child_ids)

    children_data = await asyncio.get_running_loop().run_in_executor(None, _sync_get_children)

    # Collect unique parent IDs
    parent_ids = set()
    for metadata in children_data.get("metadatas", []):
        if metadata.get("parent_id"):
            parent_ids.add(metadata["parent_id"])

    if not parent_ids:
        return state

    # Step 2: Fetch parent documents from parent_store
    def _sync_get_parents():
        return parent_store.get(ids=list(parent_ids))

    parents_data = await asyncio.get_running_loop().run_in_executor(None, _sync_get_parents)

    parent_docs = [
        Document(page_content=pc, metadata=m, id=i)
        for pc, m, i in zip(
            parents_data.get("documents", []),
            parents_data.get("metadatas", []),
            parents_data.get("ids", []),
        )
    ]

    state["matched_parent_ids"].extend([d.id for d in parent_docs]) #type:ignore

    # Update last result with parent contexts
    if state["results"]:
        state["results"][-1].parent_contexts = parent_docs

    state["current_step_idx"] += 1
    return state


async def children_lookup_node(state: material_state) -> material_state:
    """
    Given parent IDs, fetch child chunks for detailed view.
    1. Get child_ids from matched parents (parent_store)
    2. Fetch child documents (material_store)
    """
    step = state["plan"][state["current_step_idx"]]
    if step.mode != "children_lookup":
        return state

    parent_ids = state["matched_parent_ids"]
    if not parent_ids:
        return state

    # Step 1: Get child_ids from parents (stored in parent metadata)
    def _sync_get_parents():
        return parent_store.get(ids=list(parent_ids))

    parents_data = await asyncio.get_running_loop().run_in_executor(None, _sync_get_parents)

    # Collect all child IDs from parent documents
    all_child_ids = set()
    for metadata in parents_data.get("metadatas", []):
        child_ids = metadata.get("child_ids", [])
        if isinstance(child_ids, list):
            all_child_ids.update(child_ids)

    if not all_child_ids:
        return state

    # Step 2: Fetch child documents from material_store
    def _sync_get_children():
        return material_store.get(ids=list(all_child_ids))

    children_data = await asyncio.get_running_loop().run_in_executor(None, _sync_get_children)

    child_docs = [
        Document(page_content=pc, metadata=m, id=i)
        for pc, m, i in zip(
            children_data.get("documents", []),
            children_data.get("metadatas", []),
            children_data.get("ids", []),
        )
    ]

    state["matched_child_ids"].extend([d.id for d in child_docs]) #type:ignore

    result = MaterialResult(
        step_id=step.step_id,
        matched_children=child_docs,
        parent_contexts=[],
        mode="children_lookup",
    )
    state["results"].append(result)
    state["current_step_idx"] += 1
    return state


async def plan_node(state: material_state) -> material_state:
    """
    Plan the retrieval strategy based on query.
    Default: semantic search children → parent lookup (small-to-big).
    """
    plan_model = ChatDeepSeek(model=LLM_MODEL, temperature=0.3)
    planner = plan_model.with_structured_output(material_retrieve_step, mode="json")

    system_prompt = """你是一个材料检索规划专家。制定检索计划。

可用模式：
1. semantic_search - 语义搜索子块（精准匹配）
2. metadata_filter - 元数据过滤
3. parent_lookup - 根据子块查找父块（small-to-big核心）
4. children_lookup - 根据父块查找子块

默认策略（推荐）：
1. semantic_search children
2. parent_lookup (获取父块上下文)

请生成1-2步的检索计划。"""

    try:
        steps = await planner.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"查询: {state['query']}"},
        ])
        if isinstance(steps, material_retrieve_step):
            plan = [steps]
        elif isinstance(steps, list):
            plan = steps
        else:
            plan = [material_retrieve_step(**s) if isinstance(s, dict) else s for s in steps] #type:ignore
    except Exception:
        # Default: semantic search on children then parent lookup
        plan = [
            material_retrieve_step(
                step_id=1,
                mode="semantic_search",
                target="children",
                temp_query=state["query"],
            ),#type:ignore
            material_retrieve_step(
                step_id=2,
                mode="parent_lookup",
                target="parents",
            ),#type:ignore
        ]

    # Limit maximum steps to prevent infinite loops
    MAX_STEPS = 5
    if len(plan) > MAX_STEPS:
        logger.warning(f"Plan has {len(plan)} steps, limiting to {MAX_STEPS}")
        plan = plan[:MAX_STEPS]
    
    for i, step in enumerate(plan):
        step.step_id = i + 1 #type:ignore

    state["plan"] = plan #type:ignore
    state["current_step_idx"] = 0
    logger.debug(f"Generated plan with {len(plan)} steps for query: {state['query'][:50]}...")
    return state


def route_dispatch_node(state: material_state) -> material_state:
    """Route dispatch node - returns state unchanged."""
    return state

def route_dispatch(state: material_state) -> Literal["semantic_search_children", "metadata_filter_node", "parent_lookup_node", "children_lookup_node", "__end__"]:
    """Route to appropriate node based on step mode."""
    if state["current_step_idx"] >= len(state["plan"]):
        return "__end__"
    step = state["plan"][state["current_step_idx"]]
    return {
        "semantic_search": "semantic_search_children",
        "metadata_filter": "metadata_filter_node",
        "parent_lookup": "parent_lookup_node",
        "children_lookup": "children_lookup_node",
    }.get(step.mode, "__end__") #type:ignore


def first_route(
    state: material_state,
) -> Literal["route_dispatch", "__end__"]:
    """Route from planner to first execution node."""
    if not state["plan"]:
        logger.debug("first_route: plan is empty, returning '__end__'")
        return "__end__"
    logger.debug(f"first_route: plan has {len(state['plan'])} steps, returning 'route_dispatch'")
    return "route_dispatch"


def after_execution(state: material_state) -> Literal["route_dispatch", "__end__"]:
    """Route to next step or end after node execution."""
    if "_execution_loop_count" not in state:
        state["_execution_loop_count"] = 0
    state["_execution_loop_count"] += 1

    MAX_LOOPS = 20
    if state["_execution_loop_count"] > MAX_LOOPS:
        logger.error(f"Infinite loop detected! current_step_idx={state['current_step_idx']}, plan_len={len(state['plan'])}, forcing end")
        return "__end__"

    if state["current_step_idx"] >= len(state["plan"]):
        logger.debug(f"after_execution: reached end of plan, returning '__end__'")
        return "__end__"

    logger.debug(f"after_execution: current_step_idx={state['current_step_idx']}, routing to next step")
    return "route_dispatch"


def build_material_graph():
    """Build LangGraph state machine for material retrieval."""
    graph = StateGraph(material_state)

    graph.add_node("planner", plan_node)
    graph.add_node("route_dispatch", route_dispatch_node)
    graph.add_node("semantic_search_children", semantic_search_children)
    graph.add_node("metadata_filter_node", metadata_filter_node)
    graph.add_node("parent_lookup_node", parent_lookup_node)
    graph.add_node("children_lookup_node", children_lookup_node)

    graph.set_entry_point("planner")
    graph.add_conditional_edges("planner", first_route)
    graph.add_conditional_edges("route_dispatch", route_dispatch)
    graph.add_conditional_edges("semantic_search_children", after_execution)
    graph.add_conditional_edges("metadata_filter_node", after_execution)
    graph.add_conditional_edges("parent_lookup_node", after_execution)
    graph.add_conditional_edges("children_lookup_node", after_execution)

    return graph.compile()


_material_graph = None


def get_material_graph():
    global _material_graph
    if _material_graph is None:
        _material_graph = build_material_graph()
    return _material_graph


async def retrieve_materials(query: str) -> List[MaterialResult]:
    """
    search for non-diary materials stored in database

    Returns:
        List of MaterialResult containing matched children and their parent contexts.
    """
    graph = get_material_graph()

    initial_state: material_state = {
        "query": query,
        "plan": [],
        "current_step_idx": 0,
        "results": [],
        "matched_child_ids": [],
        "matched_parent_ids": [],
    }

    final_state = await graph.ainvoke(initial_state)
    return final_state["results"]
