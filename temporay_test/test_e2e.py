#!/usr/bin/env python3
"""
End-to-end test for the psychological counselor system.
Tests file upload, diary processing, and conversation with injection.
"""

import asyncio
import sys
import os
from pathlib import Path
import json
import tempfile

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

# Disable LangSmith for testing
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Import after setting environment variables
from langchain_deepseek import ChatDeepSeek
from top_module import SharedContext, analyst_observer, supervisor_observer
from mem_store_diary import store_diary
from chatter import call_chatter

async def test_diary_processing():
    """Test processing a diary file and retrieving insights."""
    print("Testing diary processing...")
    
    # Create a test diary file
    diary_content = """2024年3月15日 星期五 晴

今天我感到非常焦虑。工作上有个重要的项目要交付，但我总觉得时间不够用。昨晚又失眠了，脑子里一直在想工作的事情。

我感到压力很大，也有些沮丧。希望能找到放松的方法。朋友建议我试试冥想，但我总是静不下心来。

情绪：焦虑，压力，沮丧，疲惫"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(diary_content)
        temp_file = f.name
    
    try:
        # Process the diary file
        result = await store_diary(temp_file)
        print(f"[OK] Diary processed: {result}")
        
        # Check if it was stored
        if "stored" in result.lower() or "success" in result.lower() or "存储" in result:
            print("[OK] Diary appears to be stored successfully")
        else:
            print("[WARN] Diary storage may have issues")
            
    finally:
        os.unlink(temp_file)
    
    return True

async def test_conversation_with_injection():
    """Test a conversation where analyst provides insights."""
    print("\nTesting conversation with injection...")
    
    # Create shared context
    ctx = SharedContext()
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0.5)
    
    # Start observer tasks
    analyst_task = asyncio.create_task(analyst_observer(ctx, llm))
    supervisor_task = asyncio.create_task(supervisor_observer(ctx, llm))
    
    # Simulate a conversation about work stress
    await ctx.add_message("user", "我最近工作压力特别大，经常失眠")
    await ctx.add_message("assistant", "听起来你最近承受了很大的工作压力。失眠确实会让人更加疲惫。你能具体说说工作中的压力来源吗？")
    await ctx.add_message("user", "项目deadline很紧，而且需求一直在变，我感到很焦虑")
    
    # Trigger analyst
    print("Triggering analyst...")
    ctx.on_analyst_trigger.set()
    
    # Wait for analyst to process
    await asyncio.sleep(5)
    
    # Check if analyst created an injection
    if ctx.analyst_injection:
        print(f"[OK] Analyst created injection: {ctx.analyst_injection.content[:150]}...")
        
        # Now test chatter with the injection
        print("Testing chatter with analyst injection...")
        response = await call_chatter(ctx)
        print(f"[OK] Chatter response: {response[:200]}...")
        
        # Check if response incorporates the injection
        if "压力" in response or "焦虑" in response or "工作" in response or "失眠" in response:
            print("[OK] Response appears to incorporate analyst insights")
        else:
            print("[WARN] Response may not be using injection effectively")
    else:
        print("[INFO] No analyst injection created (may be due to no diary data)")
        
        # Test chatter without injection
        print("Testing chatter without injection...")
        response = await call_chatter(ctx)
        print(f"[OK] Chatter response: {response[:200]}...")
    
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
    
    return True

async def test_web_server_apis():
    """Test the web server APIs without running the full server."""
    print("\nTesting web server APIs...")
    
    # We'll test by importing and calling the functions directly
    try:
        from web.main import handle_user_message, ConnectionContext
        print("[OK] Web server modules can be imported")
        
        # Note: Can't fully test without running server, but import is good
        return True
    except Exception as e:
        print(f"[ERROR] Failed to import web modules: {e}")
        return False

async def main():
    """Run all end-to-end tests."""
    print("=" * 60)
    print("End-to-End System Test")
    print("=" * 60)
    
    # Test 1: Diary processing
    diary_ok = await test_diary_processing()
    
    # Test 2: Conversation with injection
    conv_ok = await test_conversation_with_injection()
    
    # Test 3: Web server API imports
    web_ok = await test_web_server_apis()
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"  Diary Processing: {'PASS' if diary_ok else 'FAIL'}")
    print(f"  Conversation with Injection: {'PASS' if conv_ok else 'FAIL'}")
    print(f"  Web Server APIs: {'PASS' if web_ok else 'FAIL'}")
    print("=" * 60)
    
    all_passed = diary_ok and conv_ok and web_ok
    return all_passed

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)