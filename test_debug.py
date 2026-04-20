#!/usr/bin/env python3
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_chatter():
    """测试 chatter 是否能够初始化并响应"""
    print("测试 chatter 初始化...")
    try:
        from chatter import chatter
        print("[OK] chatter 导入成功")
        
        # 尝试一个简单的调用
        print("测试 chatter.ainvoke...")
        # 注意：这需要 DEEPSEEK_API_KEY 环境变量
        response = await chatter.ainvoke({
            "messages": [{"role": "user", "content": "Hello"}]
        })
        print(f"[OK] chatter 响应成功")
        return True
    except Exception as e:
        print(f"[ERROR] chatter 测试失败: {e}")
        return False

async def test_shared_context():
    """测试 SharedContext 基本功能"""
    print("\n测试 SharedContext...")
    try:
        from SharedContext import SharedContext
        ctx = SharedContext()
        await ctx.add_message("user", "test message")
        messages = await ctx.get_recent_messages(5)
        print(f"[OK] SharedContext 工作正常，消息数: {len(messages)}")
        return True
    except Exception as e:
        print(f"[ERROR] SharedContext 测试失败: {e}")
        return False

async def test_analysist():
    """测试 analysist 初始化"""
    print("\n测试 analysist...")
    try:
        from analysist import analysist
        print("[OK] analysist 导入成功")
        return True
    except Exception as e:
        print(f"[ERROR] analysist 测试失败: {e}")
        return False

async def test_supervisor():
    """测试 supervisor 初始化"""
    print("\n测试 supervisor...")
    try:
        from supervisor import supervisor
        print("[OK] supervisor 导入成功")
        return True
    except Exception as e:
        print(f"[ERROR] supervisor 测试失败: {e}")
        return False

async def main():
    print("=== 心理咨询系统调试 ===\n")
    
    # 检查环境变量
    print("检查环境变量...")
    if "DEEPSEEK_API_KEY" in os.environ:
        print(f"[OK] DEEPSEEK_API_KEY 已设置")
    else:
        print("[ERROR] DEEPSEEK_API_KEY 未设置！这是必需的。")
        print("  请在 .env 文件中添加: DEEPSEEK_API_KEY=your_key_here")
    
    # 运行测试
    results = []
    results.append(await test_shared_context())
    results.append(await test_analysist())
    results.append(await test_supervisor())
    results.append(await test_chatter())
    
    print("\n=== 测试总结 ===")
    for i, (name, result) in enumerate([
        ("SharedContext", results[0]),
        ("analysist", results[1]),
        ("supervisor", results[2]),
        ("chatter", results[3])
    ]):
        status = "[OK]" if result else "[ERROR]"
        print(f"{status} {name}")
    
    if all(results):
        print("\n所有组件初始化正常。")
    else:
        print("\n某些组件初始化失败。请检查以上错误信息。")

if __name__ == "__main__":
    asyncio.run(main())