#!/usr/bin/env python3
"""
Test script to verify WebSocket observer tasks and injection mechanism.
"""

import asyncio
import sys
from pathlib import Path
import io
import sys

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

# Set LangSmith environment variables before imports
import os
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "counselor-agent-demo")

from top_module import SharedContext, analyst_observer, supervisor_observer
from langchain_deepseek import ChatDeepSeek

async def test_observer_tasks():
    """Test that observer tasks can be started and process triggers."""
    print("Testing observer tasks...")
    
    # Create shared context
    ctx = SharedContext()
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0.5)
    
    # Start observer tasks
    analyst_task = asyncio.create_task(analyst_observer(ctx, llm))
    supervisor_task = asyncio.create_task(supervisor_observer(ctx, llm))
    
    # Add some messages to context
    await ctx.add_message("user", "我今天感到有些焦虑")
    await ctx.add_message("assistant", "我理解你的感受，能具体说说是什么让你感到焦虑吗？")
    
    # Set triggers
    print("Setting analyst trigger...")
    ctx.on_analyst_trigger.set()
    
    # Wait a bit for observer to process
    await asyncio.sleep(2)
    
    # Check if injection was created
    if ctx.analyst_injection:
        print(f"[OK] Analyst injection created: {ctx.analyst_injection.content[:100]}...")
    else:
        print("[ERROR] No analyst injection created")
    
    # Set supervisor trigger
    print("Setting supervisor trigger...")
    ctx.on_supervisor_trigger.set()
    
    # Wait a bit for observer to process
    await asyncio.sleep(2)
    
    # Check if injection was created
    if ctx.supervisor_injection:
        print(f"[OK] Supervisor injection created: {ctx.supervisor_injection.content[:100]}...")
    else:
        print("[ERROR] No supervisor injection created")
    
    # Cancel tasks
    analyst_task.cancel()
    supervisor_task.cancel()
    
    try:
        await analyst_task
    except asyncio.CancelledError:
        pass
    
    try:
        await supervisor_task
    except asyncio.CancelledError:
        pass
    
    print("Test completed!")

async def test_chatter_injection():
    """Test that chatter correctly uses injections."""
    print("\nTesting chatter injection...")
    
    from chatter import call_chatter
    
    # Create shared context
    ctx = SharedContext()
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0.5)
    
    # Add conversation
    await ctx.add_message("user", "我最近工作压力很大")
    await ctx.add_message("assistant", "工作压力确实会让人感到疲惫，你从事什么工作呢？")
    await ctx.add_message("user", "我是一名软件工程师，最近项目deadline很紧")
    
    # Manually set an analyst injection (simulating what observer would do)
    await ctx.safe_set_analyst("根据用户的日记记录，用户在过去一周多次提到工作压力和睡眠不足。建议关注工作与生活的平衡，并提供放松技巧。", priority="high")
    
    print(f"Analyst injection set: {ctx.analyst_injection.content}")
    
    # Call chatter with injection
    response = await call_chatter(ctx)
    
    print(f"[OK] Chatter response: {response[:200]}...")
    
    # Check if response acknowledges the injection
    if "压力" in response or "工作" in response or "放松" in response:
        print("[OK] Response appears to incorporate injection")
    else:
        print("[WARN] Response may not be using injection effectively")

async def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing WebSocket Observer Fixes")
    print("=" * 60)
    
    await test_observer_tasks()
    await test_chatter_injection()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())