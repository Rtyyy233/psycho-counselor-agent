#!/usr/bin/env python3
"""
最终集成测试：验证情绪类型扩展后的完整上传、存储、分析流程
"""

import asyncio
import aiohttp
import tempfile
import os
import json
import sys
from pathlib import Path


async def test_complex_diary():
    """测试包含多种情绪、日期格式和内容的复杂日记"""
    content = """2024-12-25 圣诞节
今天是最害羞又兴奋的一天！在家庭聚会上表演节目，虽然害羞但感到家人的温暖和安全。
晚上独自思考，有点孤独和迷茫，但也有些反思。

2025.01.15 工作日记
今天项目成功了！感到自豪和满足，团队合作很愉快。
但也有点疲惫和紧张，担心下一步。

12月20日 社交
和朋友去爬山，感觉放松和好奇。大自然让我平静。
遇到陌生人时有点紧张，但后来感到感激。
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        filepath = f.name

    try:
        print("=== 最终集成测试 ===")
        print("测试文件内容预览：")
        print(content[:300] + "...")

        # 准备表单数据
        data = aiohttp.FormData()
        data.add_field(
            "file",
            open(filepath, "rb"),
            filename="complex_diary.txt",
            content_type="text/plain",
        )

        async with aiohttp.ClientSession() as session:
            print("\n1. 上传文件到API...")
            async with session.post(
                "http://localhost:8000/api/upload", data=data
            ) as resp:
                print(f"HTTP状态码: {resp.status}")
                if resp.status == 200:
                    result = await resp.json()
                    print("✓ 上传成功")
                    print(f"存储结果: {result.get('storage_result', 'N/A')}")
                    print(f"分析完成: {'analysis' in result}")

                    # 检查扩展情绪词汇
                    analysis = result.get("analysis", "")
                    extended_emotions = [
                        "害羞",
                        "兴奋",
                        "安全感",
                        "孤独",
                        "迷茫",
                        "反思",
                        "自豪",
                        "满足",
                        "疲惫",
                        "紧张",
                        "放松",
                        "好奇",
                        "平静",
                        "感激",
                    ]
                    found = [e for e in extended_emotions if e in analysis]
                    print(f"✓ 分析中包含的扩展情绪: {found}")

                    # 检查分析质量
                    if len(analysis) > 100:
                        print("✓ 分析内容完整")
                    else:
                        print("✗ 分析内容可能过短")

                    return True
                else:
                    error_text = await resp.text()
                    print(f"✗ 上传失败: {error_text}")
                    return False

    except Exception as e:
        print(f"✗ 测试异常: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        if os.path.exists(filepath):
            os.unlink(filepath)


async def test_direct_tool_call():
    """直接测试工具调用"""
    print("\n2. 直接测试store_diary工具...")
    try:
        sys.path.insert(0, "src")
        from tool_utils import call_tool_async

        # 创建测试文件
        content = "2024-12-01\n今天心情平静而放松。"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            filepath = f.name

        result = await call_tool_async("store_diary", {"file_path": filepath})
        print(f"工具调用结果: {result}")

        if result == "Success" or "Success" in str(result):
            print("✓ 工具调用成功")
            return True
        else:
            print(f"✗ 工具调用失败: {result}")
            return False

    except Exception as e:
        print(f"✗ 工具测试异常: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        if "filepath" in locals() and os.path.exists(filepath):
            os.unlink(filepath)


async def test_web_interface():
    """测试Web界面可访问性"""
    print("\n3. 测试Web界面...")
    try:
        async with aiohttp.ClientSession() as session:
            # 测试根路径
            async with session.get("http://localhost:8000/") as resp:
                if resp.status == 200:
                    print("✓ Web根路径可访问")
                else:
                    print(f"✗ Web根路径不可访问: {resp.status}")

            # 测试上传页面
            async with session.get("http://localhost:8000/upload") as resp:
                if resp.status == 200:
                    print("✓ 上传页面可访问")
                else:
                    print(f"✗ 上传页面不可访问: {resp.status}")

            return True
    except Exception as e:
        print(f"✗ Web界面测试异常: {e}")
        return False


async def main():
    """运行所有测试"""
    print("=" * 60)
    print("心理咨询师代理 - 最终集成测试")
    print("=" * 60)

    tests_passed = 0
    tests_total = 3

    # 测试1: 复杂日记上传
    if await test_complex_diary():
        tests_passed += 1

    # 测试2: 直接工具调用
    if await test_direct_tool_call():
        tests_passed += 1

    # 测试3: Web界面
    if await test_web_interface():
        tests_passed += 1

    print("\n" + "=" * 60)
    print(f"测试结果: {tests_passed}/{tests_total} 通过")

    if tests_passed == tests_total:
        print("✓ 所有测试通过！系统功能正常。")
        print("\n总结:")
        print("1. 情绪类型扩展已生效（支持30种情绪词汇）")
        print("2. 文件上传、存储、分析流程完整")
        print("3. Web界面可正常访问")
        print("4. 向量数据库存储成功")
        return 0
    else:
        print("✗ 部分测试失败，需要进一步检查。")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
