#!/usr/bin/env python3
"""
测试文件上传API
"""

import asyncio
import aiohttp
import tempfile
import os
import sys
from pathlib import Path


async def test_upload():
    # 创建测试文件
    content = """25.12.21
今天心情很好，和朋友去了公园。
感觉轻松愉快。
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        filepath = f.name

    try:
        print(f"测试文件: {filepath}")

        # 准备表单数据
        data = aiohttp.FormData()
        data.add_field(
            "file",
            open(filepath, "rb"),
            filename="test_diary.txt",
            content_type="text/plain",
        )

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8000/api/upload", data=data
            ) as resp:
                print(f"状态码: {resp.status}")
                result = await resp.json()
                print(f"响应: {result}")

    except Exception as e:
        print(f"上传测试失败: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if os.path.exists(filepath):
            os.unlink(filepath)


async def main():
    print("=== 测试文件上传 ===")
    await test_upload()


if __name__ == "__main__":
    asyncio.run(main())
