#!/usr/bin/env python3
"""
测试扩展情绪词汇的文件上传
包含：害羞、安全感、兴奋、失望、感激等新增情绪类型
"""

import asyncio
import aiohttp
import tempfile
import os
import sys
from pathlib import Path


async def test_extended_emotions():
    # 创建测试文件，包含多种扩展情绪词汇
    content = """2024-12-25
今天是我最害羞的一天，在会议上发言时脸都红了。
但同时也感到一种深深的安全感，因为同事们都支持我。
对这次机会感到兴奋，虽然结果有些失望。
最后，心中充满感激之情。

12月26日
复杂的一天，情绪波动很大。
早上感到孤独和迷茫，下午转为好奇和放松。
晚上又变得紧张和困惑。
今天真是情绪过山车。

2025.01.01
新年第一天，感到自豪和满足。
虽然有点疲惫，但充满希望。
爱我的家人，恨自己有时候的拖延。
有点嫉妒别人的成就，但很快转为平静。
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        filepath = f.name

    try:
        print(f"测试文件（扩展情绪）: {filepath}")
        print("文件内容预览：")
        print(content[:200] + "...")

        # 准备表单数据
        data = aiohttp.FormData()
        data.add_field(
            "file",
            open(filepath, "rb"),
            filename="test_extended_emotions.txt",
            content_type="text/plain",
        )

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8000/api/upload", data=data
            ) as resp:
                print(f"状态码: {resp.status}")
                if resp.status == 200:
                    result = await resp.json()
                    print(f"上传成功!")
                    print(f"存储状态: {result.get('storage_result', 'N/A')}")
                    print(f"分析状态: {'analysis' in result}")

                    # 检查响应中是否包含情绪分析
                    analysis = result.get("analysis", "")
                    if analysis:
                        # 检查是否包含我们提到的情绪词汇
                        emotions_to_check = [
                            "害羞",
                            "安全感",
                            "兴奋",
                            "失望",
                            "感激",
                            "孤独",
                            "迷茫",
                            "好奇",
                            "放松",
                            "紧张",
                            "困惑",
                            "自豪",
                            "满足",
                            "疲惫",
                            "希望",
                            "爱",
                            "恨",
                            "嫉妒",
                            "平静",
                        ]
                        found_emotions = [
                            emotion
                            for emotion in emotions_to_check
                            if emotion in analysis
                        ]
                        print(f"分析中包含的情绪词汇: {found_emotions}")
                else:
                    print(f"错误响应: {await resp.text()}")

    except Exception as e:
        print(f"上传测试失败: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if os.path.exists(filepath):
            os.unlink(filepath)


async def main():
    print("=== 测试扩展情绪词汇的文件上传 ===")
    await test_extended_emotions()


if __name__ == "__main__":
    asyncio.run(main())
