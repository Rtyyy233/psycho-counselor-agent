import asyncio
import time
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import SystemMessage, HumanMessage

async def test():
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0.2)
    
    # Test 1: Simple prompt
    print("Test 1: Simple prompt")
    start = time.time()
    try:
        response = await asyncio.wait_for(
            llm.ainvoke("Say hello briefly."),
            timeout=3.0
        )
        elapsed = time.time() - start
        print(f"  Response in {elapsed:.2f}s: {response.content[:50] if hasattr(response, 'content') else str(response)[:50]}")
    except asyncio.TimeoutError:
        elapsed = time.time() - start
        print(f"  Timeout after {elapsed:.2f}s")
    except Exception as e:
        elapsed = time.time() - start
        print(f"  Error after {elapsed:.2f}s: {e}")
    
    # Test 2: Longer prompt (like synthesis)
    print("\nTest 2: Longer prompt")
    prompt = """基于以下相关记忆，分析用户的查询并提供专业心理分析：

用户查询：我感到很不安

相关记忆片段：
[日记] 用户提到最近工作压力大，睡眠不好。

请提供：
1. 对用户当前情绪状态和认知模式的分析
2. 记忆片段中发现的模式和关联
3. 建议的下一步探索方向

请用中文回复，保持专业且富有洞察力。"""
    
    start = time.time()
    try:
        response = await asyncio.wait_for(
            llm.ainvoke([
                SystemMessage(content="你是一位专业的心理分析专家。根据提供的记忆片段进行分析。"),
                HumanMessage(content=prompt),
            ]),
            timeout=5.0
        )
        elapsed = time.time() - start
        print(f"  Response in {elapsed:.2f}s, length: {len(response.content) if hasattr(response, 'content') else len(str(response))}")
    except asyncio.TimeoutError:
        elapsed = time.time() - start
        print(f"  Timeout after {elapsed:.2f}s")
    except Exception as e:
        elapsed = time.time() - start
        print(f"  Error after {elapsed:.2f}s: {e}")

if __name__ == "__main__":
    asyncio.run(test())