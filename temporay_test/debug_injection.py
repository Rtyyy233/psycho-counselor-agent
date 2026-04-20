#!/usr/bin/env python3
"""
Debug script for Analyst injection failure.
Tests the complete flow from user message to injection.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

os.environ["LANGCHAIN_TRACING_V2"] = "false"

from top_module import SharedContext, analyst_observer
from analysist import call_analysist
from chatter import call_chatter
from langchain_deepseek import ChatDeepSeek

async def test_analyst_direct():
    """Test Analyst directly without observer."""
    print("=== 测试Analyst直接调用 ===")
    
    ctx = SharedContext()
    await ctx.add_message("user", "我感到焦虑不安")
    await ctx.add_message("assistant", "我在这里倾听你")
    
    print("调用call_analysist...")
    result = await call_analysist(ctx)
    
    if result:
        print(f"[OK] Analyst返回结果 ({len(result)} 字符)")
        print(f"前200字符: {result[:200]}")
        
        # 测试设置injection
        await ctx.safe_set_analyst(result)
        print(f"[OK] Injection设置: {ctx.analyst_injection is not None}")
        if ctx.analyst_injection:
            print(f"  内容长度: {len(ctx.analyst_injection.content)}")
    else:
        print("[FAIL] Analyst返回None")
    
    return result is not None

async def test_analyst_observer():
    """Test Analyst observer task."""
    print("\n=== 测试Analyst Observer ===")
    
    ctx = SharedContext()
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0.5)
    
    # 启动observer任务
    print("启动analyst_observer任务...")
    observer_task = asyncio.create_task(analyst_observer(ctx, llm))
    
    # 添加消息
    await ctx.add_message("user", "我感到焦虑不安")
    await ctx.add_message("assistant", "我在这里倾听你")
    
    # 等待一小会儿确保observer在运行
    await asyncio.sleep(0.5)
    
    # 触发analyst
    print("触发analyst_trigger...")
    ctx.on_analyst_trigger.set()
    
    # 等待injection被设置
    print("等待injection (最多10秒)...")
    for i in range(20):
        await asyncio.sleep(0.5)
        if ctx.analyst_injection:
            print(f"[OK] 第{i*0.5}秒后检测到injection")
            print(f"  内容长度: {len(ctx.analyst_injection.content)}")
            break
    
    if not ctx.analyst_injection:
        print("[FAIL] 10秒后仍未检测到injection")
        
        # 检查observer是否还在运行
        if observer_task.done():
            try:
                await observer_task
            except Exception as e:
                print(f"Observer任务异常: {e}")
        else:
            print("Observer任务仍在运行但未设置injection")
    
    # 取消observer任务
    observer_task.cancel()
    try:
        await observer_task
    except asyncio.CancelledError:
        pass
    
    return ctx.analyst_injection is not None

async def test_chatter_with_injection():
    """Test Chatter with pre-set injection."""
    print("\n=== 测试Chatter with Injection ===")
    
    ctx = SharedContext()
    await ctx.add_message("user", "测试消息")
    await ctx.add_message("assistant", "测试回复")
    
    # 手动设置injection
    test_injection = "测试分析内容：用户表现出焦虑情绪，建议采用共情倾听。"
    await ctx.safe_set_analyst(test_injection)
    
    print(f"设置injection后: {ctx.analyst_injection is not None}")
    print(f"Injection内容: {ctx.analyst_injection.content[:100]}...")
    
    # 调用chatter
    print("调用call_chatter...")
    response = await call_chatter(ctx)
    
    print(f"Chatter响应长度: {len(response)}")
    print(f"响应前150字符: {response[:150]}")
    
    # 检查injection是否被清除（chatter应该清除injection）
    print(f"调用后injection状态: {ctx.analyst_injection}")
    
    return len(response) > 0

async def test_full_flow():
    """Test the complete flow: observer trigger -> analyst -> chatter."""
    print("\n=== 测试完整流程 ===")
    
    ctx = SharedContext()
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0.5)
    
    # 启动observer
    observer_task = asyncio.create_task(analyst_observer(ctx, llm))
    
    # 模拟handle_user_message流程
    # 1. 添加用户消息
    await ctx.add_message("user", "我感到非常焦虑，睡不好觉")
    
    # 2. 触发observer（非阻塞）
    ctx.on_analyst_trigger.set()
    
    # 3. 立即调用chatter（模拟实际流程）
    print("立即调用chatter（模拟实际时序）...")
    chatter_task = asyncio.create_task(call_chatter(ctx))
    
    # 等待chatter完成
    response = await chatter_task
    print(f"Chatter响应长度: {len(response)}")
    print(f"响应前100字符: {response[:100]}")
    
    # 等待一段时间让analyst完成
    print("等待2秒让analyst完成...")
    await asyncio.sleep(2)
    
    print(f"Analyst injection状态: {ctx.analyst_injection}")
    if ctx.analyst_injection:
        print(f"Injection内容长度: {len(ctx.analyst_injection.content)}")
        print(f"前100字符: {ctx.analyst_injection.content[:100]}")
    
    # 取消observer
    observer_task.cancel()
    try:
        await observer_task
    except asyncio.CancelledError:
        pass
    
    return ctx.analyst_injection is not None

async def main():
    """Run only observer test."""
    print("开始Analyst Observer测试...")
    
    # 只运行Observer测试
    success = await test_analyst_observer()
    
    if success:
        print("\n[OK] Observer测试通过")
    else:
        print("\n[FAIL] Observer测试失败")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)