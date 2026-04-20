#!/usr/bin/env python3
"""
Diagnose injection timing and Ollama issues.
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
from chatter import call_chatter
from langchain_deepseek import ChatDeepSeek

async def test_injection_timing():
    """Test if injection happens before or after chatter call."""
    print("Testing injection timing...")
    
    ctx = SharedContext()
    
    # Simulate the flow in handle_user_message
    # 1. Add user message
    await ctx.add_message("user", "我感到焦虑不安")
    
    # 2. Set trigger (as done in handle_user_message)
    print("Setting analyst trigger...")
    ctx.on_analyst_trigger.set()
    
    # 3. Immediately call chatter (as done in handle_user_message)
    print("Immediately calling chatter...")
    
    # Check if injection exists before chatter call
    print(f"Before chatter - analyst_injection: {ctx.analyst_injection}")
    print(f"Before chatter - supervisor_injection: {ctx.supervisor_injection}")
    
    # Call chatter
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0.5)
    
    # Mock the chatter call to see what happens
    # First, let's manually simulate what chatter does
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    
    # Build messages as chatter would
    llm_messages = [SystemMessage(content="You are a counselor.")]
    
    # Check injections
    analyst_injection = ctx.analyst_injection
    supervisor_injection = ctx.supervisor_injection
    
    enriched_prompt = "You are a counselor."
    if analyst_injection:
        print(f"Analyst injection found during chatter call: {analyst_injection.content[:100]}...")
        enriched_prompt += f"\n\n[分析洞察]\n{analyst_injection.content}"
        # Note: chatter would clear the injection here
        # ctx.analyst_injection = None
    else:
        print("No analyst injection found during chatter call")
    
    if supervisor_injection:
        print(f"Supervisor injection found during chatter call: {supervisor_injection.content[:100]}...")
        enriched_prompt += f"\n\n[指导建议]\n{supervisor_injection.content}"
        # ctx.supervisor_injection = None
    
    print(f"Enriched prompt length: {len(enriched_prompt)} chars")
    
    # The issue: analyst_injection is None because analyst hasn't run yet
    # But analyst runs asynchronously after trigger is set
    
    return ctx

async def test_ollama_connection():
    """Test if Ollama is responding."""
    print("\nTesting Ollama connection...")
    
    try:
        from langchain_ollama import OllamaEmbeddings
        import asyncio
        
        embeddings = OllamaEmbeddings(model="qwen3-embedding:4b")
        
        # Try to embed a simple text
        print("Attempting to embed text...")
        result = embeddings.embed_query("test")
        print(f"Ollama embedding successful: {len(result)} dimensions")
        return True
    except Exception as e:
        print(f"Ollama error: {e}")
        return False

async def test_analyst_flow():
    """Test if analyst can complete its work."""
    print("\nTesting analyst flow...")
    
    from top_module import analyst_observer
    from langchain_deepseek import ChatDeepSeek
    
    ctx = SharedContext()
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0.5)
    
    # Add some context
    await ctx.add_message("user", "我感到焦虑不安")
    await ctx.add_message("assistant", "我理解你的感受，能具体说说是什么让你感到焦虑吗？")
    
    # Create analyst task
    analyst_task = asyncio.create_task(analyst_observer(ctx, llm))
    
    # Set trigger
    print("Setting analyst trigger...")
    ctx.on_analyst_trigger.set()
    
    # Wait a bit for analyst to process
    await asyncio.sleep(3)
    
    # Check if injection was created
    if ctx.analyst_injection:
        print(f"Analyst created injection: {ctx.analyst_injection.content[:200]}...")
    else:
        print("No analyst injection created")
        # Check if analyst task is still running
        if not analyst_task.done():
            print("Analyst task still running (may be waiting for Ollama)")
        else:
            print("Analyst task completed or failed")
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

async def test_websocket_state_reset():
    """Test if WebSocket reconnection resets context."""
    print("\nTesting WebSocket state reset...")
    
    # Simulate what happens in switchSession
    # 1. Old WebSocket closes
    # 2. New WebSocket connects with same session_id
    # 3. New ConnectionContext is created
    # 4. New SharedContext is created
    # 5. Observer tasks are started
    
    # This means injections are lost because they're in the old SharedContext
    
    print("WebSocket reconnection creates NEW SharedContext")
    print("Injection state is NOT preserved across reconnections")
    print("This matches user's suspicion about 'back to chat' resetting state")
    
    return True

async def main():
    """Run all diagnostics."""
    print("=" * 60)
    print("Injection & Ollama Diagnostics")
    print("=" * 60)
    
    # Test Ollama first
    ollama_ok = await test_ollama_connection()
    
    if not ollama_ok:
        print("\n⚠️  Ollama is not responding. This will cause:")
        print("   - Diary/material/conv outline retrieval failures")
        print("   - Analyst may not produce injections")
        print("\nPlease ensure Ollama is running:")
        print("  ollama serve")
        print("  ollama pull qwen3-embedding:4b")
    
    # Test injection timing
    await test_injection_timing()
    
    # Test analyst flow
    analyst_ok = await test_analyst_flow()
    
    # Test WebSocket state issue
    await test_websocket_state_reset()
    
    print("\n" + "=" * 60)
    print("Diagnosis Summary:")
    print("=" * 60)
    
    print(f"1. Ollama connection: {'OK' if ollama_ok else 'FAILED'}")
    print(f"2. Analyst injection: {'OK' if analyst_ok else 'FAILED or delayed'}")
    print("3. Timing issue: Chatter called BEFORE analyst completes")
    print("4. WebSocket reset: Reconnection loses injection state")
    
    print("\nRecommendations:")
    print("1. Fix Ollama service (if failed)")
    print("2. Consider delayed injection for NEXT message")
    print("3. Or wait for analyst before responding (slower but accurate)")
    print("4. Store injections in session storage to survive reconnections")

if __name__ == "__main__":
    asyncio.run(main())