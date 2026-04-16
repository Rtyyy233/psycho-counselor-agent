"""
统一工具调用接口

提供标准化的工具调用和响应提取，处理@tool装饰函数返回的各种响应类型。
"""

import asyncio
import re
from typing import Any, Callable, Optional
import logging

logger = logging.getLogger(__name__)


async def call_tool_async(
    tool: Callable,
    args: dict,
    timeout: float = 60.0,
    max_retries: int = 1,
    retry_delay: float = 1.0,
) -> str:
    """
    统一调用异步工具并提取结果文本。

    Args:
        tool: @tool装饰的异步工具函数
        args: 传递给工具的参数字典
        timeout: 超时时间（秒）
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）

    Returns:
        提取的工具响应文本

    Raises:
        asyncio.TimeoutError: 超时
        Exception: 工具执行失败
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            # Get tool name for logging (handle StructuredTool objects)
            tool_name = getattr(tool, "__name__", getattr(tool, "name", str(tool)))
            logger.debug(
                f"调用工具 {tool_name}, 参数: {args}, 尝试 {attempt + 1}/{max_retries + 1}"
            )

            # 调用工具
            raw_result = await asyncio.wait_for(tool.ainvoke(args), timeout=timeout)

            # 提取结果
            result = extract_tool_response(raw_result)
            logger.debug(
                f"工具 {tool_name} 调用成功, 结果: {result[:200] if len(result) > 200 else result}"
            )
            return result

        except asyncio.TimeoutError:
            tool_name = getattr(tool, "__name__", getattr(tool, "name", str(tool)))
            last_exception = asyncio.TimeoutError(
                f"工具 {tool_name} 调用超时 ({timeout}秒)"
            )
            logger.warning(f"工具调用超时, 尝试 {attempt + 1}/{max_retries + 1}")

        except Exception as e:
            last_exception = e
            logger.exception(f"工具调用失败: {e}, 尝试 {attempt + 1}/{max_retries + 1}")

        # 如果不是最后一次尝试，等待后重试（指数退避）
        if attempt < max_retries:
            # 指数退避：每次重试延迟加倍，最大不超过10秒
            delay = retry_delay * (2**attempt)
            max_delay = 10.0
            actual_delay = min(delay, max_delay)
            logger.debug(f"重试延迟: {actual_delay:.2f}秒")
            await asyncio.sleep(actual_delay)

    # 所有重试都失败
    raise last_exception if last_exception else Exception("工具调用失败")


def extract_tool_response(response: Any) -> str:
    """
    从各种LangChain/LangGraph响应类型中提取文本内容。

    处理以下类型:
    - 字符串
    - AIMessage, ToolMessage, HumanMessage (有content属性)
    - AgentFinish (有output属性)
    - 字典 (包含content/output/text/result/messages等键)
    - 其他对象 (尝试转换为字符串)

    Args:
        response: 工具返回的响应

    Returns:
        提取的文本内容
    """
    # 如果是字符串，直接返回
    if isinstance(response, str):
        return response

    # 检查content属性 (AIMessage, HumanMessage等)
    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, str):
            return content
        else:
            return str(content)

    # 检查output属性 (AgentFinish, 某些工具响应)
    if hasattr(response, "output"):
        return str(response.output)

    # 检查text属性 (某些消息类型)
    if hasattr(response, "text"):
        return str(response.text)

    # 检查result属性 (某些工具响应)
    if hasattr(response, "result"):
        return str(response.result)

    # 检查字典类型
    if isinstance(response, dict):
        # 常见键名
        for key in ["content", "output", "text", "result", "response", "message"]:
            if key in response:
                value = response[key]
                if isinstance(value, str):
                    return value
                else:
                    return str(value)

        # 特殊处理: messages列表 (LangGraph常见)
        if "messages" in response and isinstance(response["messages"], list):
            # 查找最后一个有内容的message
            for msg in reversed(response["messages"]):
                if hasattr(msg, "content"):
                    content = msg.content
                    return str(content) if not isinstance(content, str) else content
                elif isinstance(msg, dict) and "content" in msg:
                    return str(msg["content"])
                elif isinstance(msg, dict) and "text" in msg:
                    return str(msg["text"])

            # 如果没有找到内容，返回第一个message的字符串表示
            if response["messages"]:
                return str(response["messages"][0])

        # 检查嵌套结构
        for key in ["output", "result", "response", "return_value", "data"]:
            if key in response:
                return extract_tool_response(response[key])

        # 单键字典
        if len(response) == 1:
            return extract_tool_response(list(response.values())[0])

    # 最后手段: 转换为字符串
    result = str(response)
    logger.debug(f"extract_tool_response最后手段转换: {result[:200]}")
    return result


def detect_file_type(content: str, filename: str) -> str:
    """
    检测文件类型（日记/材料/对话大纲）

    Args:
        content: 文件内容
        filename: 文件名

    Returns:
        "diary", "material", "conversation_outline" 或 "unknown"
    """
    content_lower = content.lower()
    filename_lower = filename.lower()

    # 1. 检查文件名模式
    if any(
        pattern in filename_lower for pattern in ["diary", "journal", "日志", "日记"]
    ):
        return "diary"
    if any(
        pattern in filename_lower
        for pattern in ["material", "资料", "学习", "阅读", "article"]
    ):
        return "material"
    if any(
        pattern in filename_lower
        for pattern in ["conversation", "dialogue", "对话", "session", "咨询"]
    ):
        return "conversation_outline"
    if any(pattern in filename_lower for pattern in ["outline", "大纲", "paip"]):
        return "conversation_outline"

    # 2. 检查内容模式
    # 日记特征: 日期模式，第一人称叙述，情感词汇
    date_patterns = [
        r"\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}",  # 2024-12-01, 2024/12/01
        r"\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}",  # 01-12-2024, 01/12/24
        r"\d{1,2}月\d{1,2}日",  # 12月1日
    ]

    import re

    for pattern in date_patterns:
        if re.search(pattern, content):
            return "diary"

    # 情感词汇（中文）
    emotion_words = [
        "开心",
        "难过",
        "悲伤",
        "生气",
        "愤怒",
        "焦虑",
        "担心",
        "害怕",
        "恐惧",
        "高兴",
        "快乐",
        "幸福",
        "失望",
        "沮丧",
        "兴奋",
        "紧张",
        "放松",
    ]
    if any(word in content for word in emotion_words):
        return "diary"

    # 材料特征: 学术词汇，标题结构，参考文献
    material_indicators = [
        "参考文献",
        "引用",
        "章节",
        "摘要",
        "引言",
        "结论",
        "图",
        "表",
        "research",
        "study",
        "analysis",
        "method",
        "result",
    ]
    if any(indicator in content_lower for indicator in material_indicators):
        return "material"

    # 对话大纲特征: 问题描述，评估，干预，计划
    conversation_indicators = [
        "问题描述",
        "评估",
        "干预",
        "计划",
        "目标",
        "进展",
        "problem",
        "assessment",
        "intervention",
        "plan",
        "goal",
    ]
    if any(indicator in content for indicator in conversation_indicators):
        return "conversation_outline"

    # 3. 默认基于扩展名
    if filename_lower.endswith((".txt", ".md")):
        # TXT和MD文件更可能是日记
        return "diary"
    elif filename_lower.endswith((".pdf", ".docx")):
        # PDF和DOCX更可能是材料
        return "material"

    return "unknown"
