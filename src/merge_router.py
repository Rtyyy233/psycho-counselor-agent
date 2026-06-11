"""
检索查询合并路由

判断来自chatter和supervisor的两个检索查询是否需要合并为一次检索执行。
"""

import logging
from typing import Optional

from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field
from config import LLM_MODEL

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """你是一个检索查询合并判断器。两个查询都用于从心理咨询记忆库中检索信息。

你需要判断它们是否应该合并为一个查询：
- 如果两个查询在语义上相近、询问的是同一方面的信息 → 合并
- 如果两个查询指向完全不同的主题或领域 → 独立

合并时，请生成一个同时覆盖两个查询需求的合并查询语句。"""


class RouterDecision(BaseModel):
    """路由决策结果"""
    should_merge: bool = Field(description="两个查询是否应该合并")
    merged_query: str = Field(description="合并后的查询语句，仅在should_merge为true时填写")
    reason: str = Field(description="判断理由")


async def route_queries(
    chatter_query: Optional[str] = None,
    supervisor_query: Optional[str] = None,
    context: str = "",
) -> RouterDecision:
    """
    判断来自chatter和supervisor的两个检索查询是否需要合并执行。

    如果只有一个查询非空，直接返回该查询（无需LLM调用）。
    如果两个查询都存在，调用LLM判断语义相关性后决定合并或独立。

    Args:
        chatter_query: 来自chatter（咨询师）的检索查询
        supervisor_query: 来自supervisor（督导）的检索查询
        context: 当前对话上下文文本，辅助路由判断

    Returns:
        RouterDecision: 包含是否合并、合并后查询（或空字符串）、判断理由
    """
    # 只有一个非空查询时，直接返回
    if chatter_query and not supervisor_query:
        return RouterDecision(
            should_merge=False,
            merged_query=chatter_query,
            reason="仅chatter提供了查询，无需合并",
        )

    if supervisor_query and not chatter_query:
        return RouterDecision(
            should_merge=False,
            merged_query=supervisor_query,
            reason="仅supervisor提供了查询，无需合并",
        )

    if not chatter_query and not supervisor_query:
        return RouterDecision(
            should_merge=False,
            merged_query="",
            reason="两个查询均为空，无需合并",
        )

    # 两个查询都存在，调用LLM判断
    router_llm = ChatDeepSeek(
        model=LLM_MODEL,
        temperature=0.1,
    )
    structured_llm = router_llm.with_structured_output(RouterDecision)

    user_prompt = f"""当前对话上下文：
{context}

查询1（来自咨询师chatter）：
{chatter_query}

查询2（来自督导supervisor）：
{supervisor_query}

请判断这两个查询是否需要合并。"""

    logger.info(
        "路由判断：chatter_query=%s, supervisor_query=%s",
        chatter_query,
        supervisor_query,
    )

    result = await structured_llm.ainvoke([
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ])

    logger.info(
        "路由结果：should_merge=%s, reason=%s",
        result.should_merge,
        result.reason,
    )

    return result
