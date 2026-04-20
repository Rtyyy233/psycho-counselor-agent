from mem_integration import original_diary, diary_annotation
from mem_store_diary import EmotionType
from typing import TypedDict, List, Literal, Optional, Annotated
import asyncio
from langchain_deepseek import ChatDeepSeek
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool


class retrieve_filter(BaseModel):
    intensity: Optional[Literal["潜意识", "弱", "中", "强", "极强"]] = Field(
        None, description="情绪强度"
    )
    date_start: Optional[str] = Field(
        None,
        description="检索时间段的开始日期，必须严格按照如'25.03.15'的格式标注",
    )
    date_end: Optional[str] = Field(
        None,
        description="检索时间段的结束日期，必须严格按照如'25.03.15'的格式标注",
    )
    scene_type: Optional[
        Literal["工作", "家庭", "亲密关系", "社交", "独处", "学习", "其他"]
    ] = Field(None, description="场景类型")
    event_type: Optional[
        Literal[
            "创伤",
            "积极",
            "重大转折",
            "日常",
            "冲突",
            "成就",
            "失落",
            "反思",
            "情绪抒发",
        ]
    ] = Field(None, description="事件类型")
    emotion: Optional[List[str]] = Field(
        None,
        description="当前情绪名称，可包含一到三个情绪。常见情绪：喜悦、悲伤、焦虑、愤怒、恐惧、惊讶、厌恶、平静、疲惫、孤独、羞耻、内疚、希望、迷茫、满足、害羞、安全感、兴奋、失望、感激、爱、恨、嫉妒、自豪、自卑、好奇、无聊、放松、紧张、困惑等",
    )


class retrieve_step(BaseModel):
    step_id: int = Field(description="步骤编号，从1开始")
    target_collection: Literal["original_diary", "diary_annotation"] = Field(
        description="查询的目标数据库"
    )
    mode: Literal["metadata_filter", "semantic_search", "id_lookup", "rerank"]
    filter: Optional[retrieve_filter] = Field(
        None, description="metadata_filter模式的过滤参数"
    )
    temp_query: Optional[str] = Field(None, description="semantic_search模式的查询语句")
    input_source: Literal["query", "previous_step"] = Field(
        description="依赖上一步结果还是原始查询"
    )


class RetrievalResult(BaseModel):
    step_id: int
    ids: List[str]
    documents: List[Document]
    mode: str
    collection: str


class agent_state(TypedDict):
    query: str
    retrieve_plan: List[retrieve_step]
    current_step_idx: int  # 0-based index into retrieve_plan
    results: List[RetrievalResult]
    previous_ids: List[str]
    previous_texts: List[str]


async def plan_node(state: agent_state) -> agent_state:
    """
    Decompose user query into multi-step retrieval plan.
    """
    plan_model = ChatDeepSeek(model="deepseek-chat", temperature=0.3)
    planner = plan_model.with_structured_output(retrieve_step, mode="json")

    system_prompt = """你是一个日记检索规划专家。根据用户查询，制定多步检索计划。

可用搜索模式：
1. metadata_filter - 基于结构化元数据过滤（intensity, date_start, date_end, scene_type, event_type, emotion）
2. semantic_search - 语义相似性搜索
3. id_lookup - 根据ID查找文档
4. rerank - 结果重排序

策略模式：
- A: Multi-Facet Filter - 多维度同时过滤
- B: Temporal Aggregation - 时间范围过滤后聚合
- C: Cross-Entry Pattern - 多ID获取模式分析
- D: Filter → Semantic - 先过滤再语义搜索
- E: Semantic → Filter - 先语义搜索再过滤
- F: Deep Dive - 过滤后获取原文

请制定1-3步的检索计划。"""

    try:
        steps = await planner.ainvoke(
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"用户查询: {state['query']}\n\n制定检索计划。",
                },
            ]
        )
        if isinstance(steps, retrieve_step):
            plan = [steps]
        elif isinstance(steps, list):
            plan = steps
        else:
            plan = [retrieve_step(**s) if isinstance(s, dict) else s for s in steps]  # type:ignore
    except Exception:
        # Fallback: simple semantic search on annotation
        plan = [
            retrieve_step(
                step_id=1,
                target_collection="diary_annotation",
                mode="semantic_search",
                temp_query=state["query"],
                input_source="query",
                filter=None,
            )
        ]

    # Ensure step_ids are sequential
    for i, step in enumerate(plan):
        step.step_id = i + 1  # type:ignore

    state["retrieve_plan"] = plan  # type:ignore
    state["current_step_idx"] = 0
    return state


def _build_where_clause(flt: retrieve_filter) -> dict:
    """Build Chroma where clause from retrieve_filter."""
    where = {}
    if flt.intensity:
        where["intensity"] = flt.intensity
    if flt.date_start:
        where["date"] = {"$gte": flt.date_start}
    if flt.date_end:
        where.setdefault("date", {})["$lte"] = flt.date_end
    if flt.scene_type:
        where["scene_type"] = flt.scene_type
    if flt.event_type:
        where["event_type"] = flt.event_type
    if flt.emotion:
        where["emotion"] = {"$in": flt.emotion}
    return where


async def metadata_filter_node(state: agent_state) -> agent_state:
    """Execute metadata filter on diary_annotation."""
    step = state["retrieve_plan"][state["current_step_idx"]]
    if step.mode != "metadata_filter" or step.filter is None:
        return state

    where = _build_where_clause(step.filter)

    def _sync():
        return diary_annotation.get(where=where)

    result_data = await asyncio.get_running_loop().run_in_executor(None, _sync)
    ids_list = list(result_data.get("ids", [])) if result_data.get("ids") else []
    docs = result_data.get("documents", [])

    result = RetrievalResult(
        step_id=step.step_id,
        ids=ids_list,
        documents=list(docs),
        mode="metadata_filter",
        collection="diary_annotation",
    )
    state["results"].append(result)
    state["previous_ids"] = ids_list
    state["previous_texts"] = [d.page_content for d in docs]
    return state


async def semantic_search_node(state: agent_state) -> agent_state:
    """Execute semantic similarity search."""
    step = state["retrieve_plan"][state["current_step_idx"]]
    if step.mode != "semantic_search":
        return state

    collection = (
        original_diary
        if step.target_collection == "original_diary"
        else diary_annotation
    )
    query_text = step.temp_query if step.input_source == "query" else state["query"]

    def _sync():
        return collection.similarity_search_with_score(query_text, k=10)  # type:ignore

    results = await asyncio.get_running_loop().run_in_executor(None, _sync)
    docs = [r[0] for r in results]
    ids_list = [d.id for d in docs if d.id]

    result = RetrievalResult(
        step_id=step.step_id,
        ids=ids_list,
        documents=docs,
        mode="semantic_search",
        collection=step.target_collection,
    )
    state["results"].append(result)
    state["previous_ids"] = ids_list
    state["previous_texts"] = [d.page_content for d in docs]
    return state


async def id_lookup_node(state: agent_state) -> agent_state:
    """Execute ID-based lookup."""
    step = state["retrieve_plan"][state["current_step_idx"]]
    if step.mode != "id_lookup":
        return state

    collection = (
        original_diary
        if step.target_collection == "original_diary"
        else diary_annotation
    )
    ids_to_fetch = state["previous_ids"] if step.input_source == "previous_step" else []
    if not ids_to_fetch:
        return state

    def _sync():
        return collection.get(ids=ids_to_fetch)

    docs_data = await asyncio.get_running_loop().run_in_executor(None, _sync)
    docs = [
        Document(page_content=pc, metadata=m, id=i)
        for pc, m, i in zip(
            docs_data.get("documents", []),
            docs_data.get("metadatas", [{}]),
            docs_data.get("ids", []),
        )
    ]

    result = RetrievalResult(
        step_id=step.step_id,
        ids=list(ids_to_fetch),
        documents=docs,
        mode="id_lookup",
        collection=step.target_collection,
    )
    state["results"].append(result)
    state["previous_ids"] = list(ids_to_fetch)
    state["previous_texts"] = [d.page_content for d in docs]
    return state


async def rerank_node(state: agent_state) -> agent_state:  # ai logic waits for check
    """Rerank results using annotation metadata."""
    step = state["retrieve_plan"][state["current_step_idx"]]
    if step.mode != "rerank":
        return state

    prev_ids = state["previous_ids"]
    if not prev_ids:
        return state

    def _sync():
        return diary_annotation.get(ids=prev_ids)

    docs_data = await asyncio.get_running_loop().run_in_executor(None, _sync)

    query_lower = state["query"].lower()
    intensity_scores = {"潜意识": 1, "弱": 2, "中": 3, "强": 4, "极强": 5}

    scored = []
    for doc, metadata in zip(
        docs_data.get("documents", []),
        docs_data.get("metadatas", []),
    ):
        score = 0
        emotions = metadata.get("emotion", [])
        if isinstance(emotions, list):
            for e in emotions:
                if e and e.lower() in query_lower:
                    score += 3
        intensity = metadata.get("intensity", "")
        score += intensity_scores.get(intensity, 0)
        event_type = metadata.get("event_type", "")
        if event_type and event_type.lower() in query_lower:
            score += 2
        scored.append((score, doc, metadata))

    scored.sort(key=lambda x: x[0], reverse=True)

    reranked_docs = [
        Document(page_content=doc, metadata=metadata, id=prev_ids[i])
        for i, (_, doc, metadata) in enumerate(scored)
    ]

    result = RetrievalResult(
        step_id=step.step_id,
        ids=prev_ids,
        documents=reranked_docs,
        mode="rerank",
        collection="diary_annotation",
    )
    state["results"].append(result)
    state["previous_ids"] = prev_ids
    state["previous_texts"] = [d.page_content for d in reranked_docs]
    return state


def route_dispatch_node(state: agent_state) -> agent_state:
    """Route dispatch node - returns state unchanged."""
    return state

def route_dispatch(state: agent_state) -> Literal["metadata_filter_node", "semantic_search_node", "id_lookup_node", "rerank_node", "__end__"]:
    """Route to appropriate node based on current step mode."""
    if state["current_step_idx"] >= len(state["retrieve_plan"]):
        return "__end__"
    step = state["retrieve_plan"][state["current_step_idx"]]
    return {
        "metadata_filter": "metadata_filter_node",
        "semantic_search": "semantic_search_node",
        "id_lookup": "id_lookup_node",
        "rerank": "rerank_node",
    }.get(step.mode, "__end__")  # type:ignore


def first_route(
    state: agent_state,
) -> Literal[
    "metadata_filter_node",
    "semantic_search_node",
    "id_lookup_node",
    "rerank_node",
    "__end__",
]:
    """Route from planner to first execution node."""
    return route_dispatch(state)


def after_execution(
    state: agent_state,
) -> Literal[
    "metadata_filter_node",
    "semantic_search_node",
    "id_lookup_node",
    "rerank_node",
    "__end__",
]:
    """After any execution node, advance to next step or end."""
    state["current_step_idx"] += 1
    if state["current_step_idx"] >= len(state["retrieve_plan"]):
        return "__end__"
    return route_dispatch(state)


def build_retrieve_graph():
    """Build LangGraph state machine for diary retrieval."""
    graph = StateGraph(agent_state)

    graph.add_node("planner", plan_node)
    graph.add_node("route_dispatch", route_dispatch_node)
    graph.add_node("metadata_filter_node", metadata_filter_node)
    graph.add_node("semantic_search_node", semantic_search_node)
    graph.add_node("id_lookup_node", id_lookup_node)
    graph.add_node("rerank_node", rerank_node)

    graph.set_entry_point("planner")
    graph.add_conditional_edges("planner", first_route)
    graph.add_conditional_edges("metadata_filter_node", after_execution)
    graph.add_conditional_edges("semantic_search_node", after_execution)
    graph.add_conditional_edges("id_lookup_node", after_execution)
    graph.add_conditional_edges("rerank_node", after_execution)

    return graph.compile()


_graph = None


def get_retrieve_graph():
    global _graph
    # Force rebuild for debugging - remove after fix is confirmed
    _graph = None
    if _graph is None:
        print("DEBUG: Building new diary retrieval graph")
        _graph = build_retrieve_graph()
    return _graph


async def retrieve_diary(query: str) -> List[RetrievalResult]:
    """
    依据日记检索所需的信息
    """
    graph = get_retrieve_graph()

    initial_state: agent_state = {
        "query": query,
        "retrieve_plan": [],
        "current_step_idx": 0,
        "results": [],
        "previous_ids": [],
        "previous_texts": [],
    }

    final_state = await graph.ainvoke(initial_state)
    return final_state["results"]
