#!/usr/bin/env python3
"""
Test the fixed analyst module.
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

from top_module import SharedContext
from analysist import call_analysist

async def test_analyst_with_retrieval():
    """Test if analyst can produce analysis with retrieval failures."""
    print("Testing analyst with retrieval...")
    
    ctx = SharedContext()
    
    # Add conversation context
    await ctx.add_message("user", "我感到焦虑不安")
    await ctx.add_message("assistant", "我理解你的感受，能具体说说是什么让你感到焦虑吗？")
    await ctx.add_message("user", "工作压力很大，deadline很近")
    
    # Call analyst directly
    print("Calling call_analysist...")
    analysis = await call_analysist(ctx)
    
    if analysis:
        print(f"✓ Analysis produced ({len(analysis)} chars):")
        print(f"  {analysis[:200]}...")
        return True
    else:
        print("✗ Analysis returned None")
        return False

async def test_injection_flow():
    """Test full injection flow."""
    print("\nTesting full injection flow...")
    
    from top_module import analyst_observer
    from langchain_deepseek import ChatDeepSeek
    
    ctx = SharedContext()
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0.5)
    
    # Add conversation
    await ctx.add_message("user", "我感到焦虑不安")
    await ctx.add_message("assistant", "我理解你的感受，能具体说说是什么让你感到焦虑吗？")
    
    # Create analyst task
    analyst_task = asyncio.create_task(analyst_observer(ctx, llm))
    
    # Set trigger
    print("Setting analyst trigger...")
    ctx.on_analyst_trigger.set()
    
    # Wait longer for analyst to complete (retrieval might be slow)
    print("Waiting for analyst (10 seconds)...")
    for i in range(10):
        await asyncio.sleep(1)
        if ctx.analyst_injection:
            print(f"Analyst injection created after {i+1} seconds")
            break
    
    # Check result
    if ctx.analyst_injection:
        print(f"✓ Analyst injection created: {ctx.analyst_injection.content[:200]}...")
        
        # Test that chatter can use it
        from chatter import call_chatter
        response = await call_chatter(ctx)
        print(f"✓ Chatter response with injection: {response[:200]}...")
        
        # Check if injection was cleared
        if ctx.analyst_injection is None:
            print("✓ Injection cleared after use")
        else:
            print("⚠ Injection NOT cleared after use")
    else:
        print("✗ No analyst injection created")
        # Check task status
        if analyst_task.done():
            try:
                await analyst_task
            except Exception as e:
                print(f"Analyst task error: {e}")
    
    # Cancel task
    analyst_task.cancel()
    try:
        await analyst_task
    except asyncio.CancelledError:
        pass
    
    return ctx.analyst_injection is not None

async def main():
    """Run tests."""
    print("=" * 60)
    print("Analyst Fix Test")
    print("=" * 60)
    
    # Test 1: Direct analyst call
    analyst_ok = await test_analyst_with_retrieval()
    
    # Test 2: Full injection flow
    injection_ok = await test_injection_flow()
    
    print("\n" + "=" * 60)
    print("Results:")
    print(f"  Analyst produces analysis: {'PASS' if analyst_ok else 'FAIL'}")
    print(f"  Injection flow works: {'PASS' if injection_ok else 'FAIL'}")
    print("=" * 60)
    
    return analyst_ok and injection_ok

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)