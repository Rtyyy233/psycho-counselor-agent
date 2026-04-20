#!/usr/bin/env python3
"""
测试工具调用
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加src到路径
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from tool_utils import call_tool_async
from mem_store_diary import store_diary
import tempfile
import logging

# 设置日志级别为DEBUG
logging.basicConfig(level=logging.DEBUG)


async def test_store_diary():
    """测试store_diary工具调用"""
    print(f"store_diary 类型: {type(store_diary)}")
    print(f"store_diary 属性: {dir(store_diary)[:20]}")

    # 创建一个测试文件
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("""2024-12-01
今天心情很好，和朋友去了公园。
感觉轻松愉快。
""")
        temp_path = f.name

    try:
        print(f"测试文件: {temp_path}")
        print(f"文件存在: {os.path.exists(temp_path)}")

        # 调用工具
        result = await call_tool_async(
            store_diary, {"file_path": temp_path}, timeout=30.0, max_retries=1
        )

        print(f"工具调用结果: {result}")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # 清理
        if os.path.exists(temp_path):
            os.unlink(temp_path)


async def main():
    print("=== 测试工具调用 ===")
    await test_store_diary()


if __name__ == "__main__":
    asyncio.run(main())
