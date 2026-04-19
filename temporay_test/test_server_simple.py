#!/usr/bin/env python3
"""
简单测试服务器功能 - 使用ASCII字符避免编码问题
"""

import asyncio
import aiohttp
import tempfile
import os


async def test_upload_simple():
    content = "2024-12-01\n今天心情平静。"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        filepath = f.name

    try:
        print("Testing file upload...")

        data = aiohttp.FormData()
        data.add_field(
            "file",
            open(filepath, "rb"),
            filename="test_simple.txt",
            content_type="text/plain",
        )

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8000/api/upload", data=data
            ) as resp:
                print(f"HTTP Status: {resp.status}")
                if resp.status == 200:
                    result = await resp.json()
                    print(f"Upload successful")
                    print(f"Storage result: {result.get('storage_result', 'N/A')}")
                    print(f"Has analysis: {'analysis' in result}")
                    return True
                else:
                    error_text = await resp.text()
                    print(f"Upload failed: {error_text}")
                    return False

    except Exception as e:
        print(f"Test error: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        if os.path.exists(filepath):
            os.unlink(filepath)


async def test_web_pages():
    print("\nTesting web pages...")
    try:
        async with aiohttp.ClientSession() as session:
            # Test root
            async with session.get("http://localhost:8000/") as resp:
                print(f"Root page: {resp.status}")

            # Test upload page
            async with session.get("http://localhost:8000/upload") as resp:
                print(f"Upload page: {resp.status}")

            return True
    except Exception as e:
        print(f"Web page test error: {e}")
        return False


async def main():
    print("=== Counselor Agent Server Test ===")

    # Test 1: Upload
    upload_ok = await test_upload_simple()

    # Test 2: Web pages
    pages_ok = await test_web_pages()

    print("\n=== Results ===")
    print(f"Upload test: {'PASS' if upload_ok else 'FAIL'}")
    print(f"Web pages test: {'PASS' if pages_ok else 'FAIL'}")

    if upload_ok and pages_ok:
        print("\nAll tests passed! Server is working correctly.")
        return 0
    else:
        print("\nSome tests failed.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
