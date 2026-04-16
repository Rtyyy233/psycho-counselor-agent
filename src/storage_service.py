"""
存储服务层

提供统一的文件存储接口，封装文件类型检测和工具调用。
替代memory_manager代理，提供更可靠直接的存储操作。
"""

import asyncio
import os
import re
from typing import Tuple, Optional
import logging

from tool_utils import call_tool_async
from config import (
    STORAGE_TIMEOUT,
    MAX_RETRIES,
    DIARY_KEYWORDS,
    CONVERSATION_KEYWORDS,
    MATERIAL_KEYWORDS,
    DATE_PATTERNS,
)

logger = logging.getLogger(__name__)


async def store_file(file_path: str, filename: str = None) -> str:
    """
    存储文件到向量数据库，自动检测文件类型并调用相应工具。

    Args:
        file_path: 文件路径
        filename: 原始文件名（用于类型检测，可选）

    Returns:
        存储结果消息

    Raises:
        ValueError: 文件类型检测失败或存储失败
        asyncio.TimeoutError: 存储超时
        Exception: 其他错误
    """
    # 如果未提供文件名，从文件路径提取
    if filename is None:
        filename = os.path.basename(file_path)

    # 检测文件类型
    file_type = detect_file_type_from_content(file_path, filename)
    logger.info(f"检测到文件类型: {file_type}, 文件: {filename}")

    # 根据文件类型调用相应工具
    if file_type == "diary":
        from mem_store_diary import store_diary

        result = await call_tool_async(
            store_diary,
            {"file_path": file_path},
            timeout=STORAGE_TIMEOUT,
            max_retries=MAX_RETRIES,
        )
        return f"日记存储成功: {result}"

    elif file_type == "conversation_outline":
        from mem_store_conv_outline import store_conversation_outline

        result = await call_tool_async(
            store_conversation_outline,
            {"file_path": file_path},
            timeout=STORAGE_TIMEOUT,
            max_retries=MAX_RETRIES,
        )
        return f"对话大纲存储成功: {result}"

    elif file_type == "material":
        from mem_store_material import store_materials

        result = await call_tool_async(
            store_materials,
            {"file_path": file_path},
            timeout=STORAGE_TIMEOUT,
            max_retries=MAX_RETRIES,
        )
        return f"材料存储成功: {result}"

    else:
        raise ValueError(f"无法识别的文件类型: {file_type}")


def detect_file_type_from_content(file_path: str, filename: str) -> str:
    """
    基于文件内容和文件名检测文件类型。

    Returns:
        "diary", "material", "conversation_outline" 或 "unknown"
    """
    # 首先检查文件名关键词
    filename_lower = filename.lower()
    if any(keyword in filename_lower for keyword in DIARY_KEYWORDS):
        return "diary"
    if any(keyword in filename_lower for keyword in CONVERSATION_KEYWORDS):
        return "conversation_outline"
    if any(keyword in filename_lower for keyword in MATERIAL_KEYWORDS):
        return "material"

    # 读取文件内容样本（前10KB）
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content_sample = f.read(10240)  # 10KB
    except UnicodeDecodeError:
        # 尝试二进制读取
        with open(file_path, "rb") as f:
            content_bytes = f.read(10240)
            content_sample = content_bytes.decode("utf-8", errors="ignore")

    content_lower = content_sample.lower()

    # 检查日期模式（日记特征）
    for pattern in DATE_PATTERNS:
        if re.search(pattern, content_sample):
            return "diary"

    # 检查关键词
    if any(keyword in content_sample for keyword in DIARY_KEYWORDS):
        return "diary"

    if any(keyword in content_sample for keyword in CONVERSATION_KEYWORDS):
        return "conversation_outline"

    if any(keyword in content_lower for keyword in MATERIAL_KEYWORDS):
        return "material"

    # 检查对话大纲特定模式（PAIP结构）
    paip_sections = ["问题描述", "评估", "干预", "计划"]
    if any(section in content_sample for section in paip_sections):
        return "conversation_outline"

    # 默认视为材料
    return "material"


def get_storage_tool_for_type(file_type: str):
    """
    根据文件类型获取对应的存储工具。

    Returns:
        工具函数
    """
    if file_type == "diary":
        from mem_store_diary import store_diary

        return store_diary
    elif file_type == "conversation_outline":
        from mem_store_conv_outline import store_conversation_outline

        return store_conversation_outline
    elif file_type == "material":
        from mem_store_material import store_materials

        return store_materials
    else:
        raise ValueError(f"不支持的文件类型: {file_type}")
