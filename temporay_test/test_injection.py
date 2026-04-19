#!/usr/bin/env python3
"""
测试分析结果注入到chatter的问题
"""
import asyncio
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from top_module import PsychologicalCounselor, SharedContext, ChatMessage
from chatter import call_chatter
from analysist import call_analysist


async def test_without_observers():
    """测试没有观察者的情况"""
    print("=== 测试1: 没有观察者任务 ===")
    
    # 创建共享上下文
    ctx = SharedContext()
    
    # 添加用户消息
    await ctx.add_message("user", "我今天感觉特别焦虑，睡不着觉。")
    
    # 设置分析器触发器（但没有观察者监听）
    ctx.on_analyst_trigger.set()
    
    # 立即调用chatter
    response = await call_chatter(ctx)
    
    print(f"Chatter响应: {response}")
    print(f"分析器注入: {ctx.analyst_injection}")
    print(f"监督器注入: {ctx.supervisor_injection}")
    
    if ctx.analyst_injection:
        print("✅ 分析结果被注入")
    else:
        print("❌ 分析结果没有被注入（因为没有观察者任务）")
    
    return ctx.analyst_injection is not None


async def test_with_observers():
    """测试有观察者的情况（模拟top_module的方式）"""
    print("\n=== 测试2: 有观察者任务（模拟）===")
    
    # 创建完整的心理咨询师实例
    counselor = PsychologicalCounselor()
    
    # 注意：我们需要手动启动观察者任务
    from top_module import analyst_observer, supervisor_observer
    
    # 启动观察者任务
    analyst_task = asyncio.create_task(analyst_observer(counselor.ctx, counselor.llm))
    supervisor_task = asyncio.create_task(supervisor_observer(counselor.ctx, counselor.llm))
    
    try:
        # 模拟用户消息
        await counselor.ctx.add_message("user", "我今天感觉特别焦虑，睡不着觉。")
        
        # 设置触发器
        counselor.ctx.on_analyst_trigger.set()
        counselor.ctx.on_supervisor_trigger.set()
        
        # 给观察者一点时间运行
        await asyncio.sleep(0.5)
        
        # 调用chatter
        response = await call_chatter(counselor.ctx)
        
        print(f"Chatter响应: {response}")
        print(f"分析器注入: {counselor.ctx.analyst_injection}")
        print(f"监督器注入: {counselor.ctx.supervisor_injection}")
        
        if counselor.ctx.analyst_injection:
            print("✅ 分析结果被注入")
            print(f"分析内容: {counselor.ctx.analyst_injection.content[:100]}...")
        else:
            print("❌ 分析结果没有被注入")
            
        return counselor.ctx.analyst_injection is not None
        
    finally:
        # 清理任务
        analyst_task.cancel()
        supervisor_task.cancel()
        try:
            await analyst_task
            await supervisor_task
        except asyncio.CancelledError:
            pass


async def test_direct_analyst_call():
    """测试直接调用分析器"""
    print("\n=== 测试3: 直接调用分析器 ===")
    
    ctx = SharedContext()
    
    # 添加用户消息
    await ctx.add_message("user", "我今天感觉特别焦虑，睡不着觉。")
    
    # 直接调用分析器
    analysis = await call_analysist(ctx)
    
    print(f"分析结果: {analysis}")
    
    if analysis:
        # 手动设置注入
        await ctx.safe_set_analyst(analysis, priority="high")
        
        # 调用chatter
        response = await call_chatter(ctx)
        
        print(f"Chatter响应: {response}")
        print(f"分析器注入: {ctx.analyst_injection}")
        
        if ctx.analyst_injection:
            print("✅ 分析结果被注入（手动设置）")
        else:
            print("❌ 分析结果没有被注入")
    else:
        print("❌ 分析器没有返回结果")
    
    return analysis is not None and ctx.analyst_injection is not None


async def test_web_socket_simulation():
    """模拟WebSocket处理流程"""
    print("\n=== 测试4: 模拟WebSocket处理流程 ===")
    
    # 创建上下文（模拟websocket_chat函数）
    ctx = SharedContext()
    
    # 添加用户消息（模拟handle_user_message函数）
    user_input = "我今天感觉特别焦虑，睡不着觉。"
    await ctx.add_message("user", user_input)
    
    # 设置触发器（模拟第228-229行）
    ctx.on_analyst_trigger.set()
    ctx.on_supervisor_trigger.set()
    
    # 关键问题：没有观察者任务监听这些触发器！
    print("问题: 设置了触发器但没有启动观察者任务")
    print("当前WebSocket代码中的流程:")
    print("1. 设置触发器 ctx.on_analyst_trigger.set()")
    print("2. 立即调用 call_chatter(ctx)")
    print("3. 但没有启动 analyst_observer 和 supervisor_observer 任务")
    
    # 立即调用chatter（模拟第234行）
    response = await call_chatter(ctx)
    
    print(f"Chatter响应: {response}")
    print(f"分析器注入: {ctx.analyst_injection}")
    print(f"监督器注入: {ctx.supervisor_injection}")
    
    if ctx.analyst_injection:
        print("✅ 分析结果被注入")
    else:
        print("❌ 分析结果没有被注入（因为观察者任务没有运行）")
    
    return ctx.analyst_injection is not None


async def main():
    print("测试分析结果注入问题")
    print("=" * 60)
    
    results = []
    
    # 运行测试
    results.append(("没有观察者", await test_without_observers()))
    results.append(("有观察者", await test_with_observers()))
    results.append(("直接调用分析器", await test_direct_analyst_call()))
    results.append(("模拟WebSocket流程", await test_web_socket_simulation()))
    
    print("\n" + "=" * 60)
    print("测试总结:")
    for name, passed in results:
        status = "通过" if passed else "失败"
        print(f"{name}: {status}")
    
    print("\n结论:")
    print("WebSocket处理中缺少观察者任务启动代码")
    print("需要在websocket_chat函数中启动analyst_observer和supervisor_observer任务")


if __name__ == "__main__":
    asyncio.run(main())