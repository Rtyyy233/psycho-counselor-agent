#!/usr/bin/env python3
"""
调试上传功能，查看详细错误
"""

import asyncio
import aiohttp
import tempfile
import os
import json


async def test_debug():
    content = """2024-12-25
今天是我最害羞的一天。
感到安全感和兴奋。
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        filepath = f.name

    try:
        print("=== 调试文件上传 ===")

        # 准备表单数据
        data = aiohttp.FormData()
        data.add_field(
            "file",
            open(filepath, "rb"),
            filename="test_debug.txt",
            content_type="text/plain",
        )

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8000/api/upload", data=data
            ) as resp:
                print(f"状态码: {resp.status}")
                if resp.status == 200:
                    result = await resp.json()
                    print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
                else:
                    error_text = await resp.text()
                    print(f"错误响应: {error_text}")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if os.path.exists(filepath):
            os.unlink(filepath)


async def test_direct_model():
    print("\n=== 直接测试模型 ===")
    try:
        sys.path.insert(0, "src")
        from mem_store_diary import EmotionalState, DiaryChunk

        # 测试创建实例
        es = EmotionalState(emotion=["害羞", "安全感", "兴奋"])
        print(f"成功创建EmotionalState: {es}")
        print(f"EmotionalState model_fields: {EmotionalState.model_fields}")

        # 查看emotion字段的annotation
        emotion_field = EmotionalState.model_fields.get("emotion")
        if emotion_field:
            print(f"emotion字段annotation: {emotion_field.annotation}")
            print(f"emotion字段default: {emotion_field.default}")

    except Exception as e:
        print(f"直接测试失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    import sys

    asyncio.run(test_debug())
    asyncio.run(test_direct_model())
