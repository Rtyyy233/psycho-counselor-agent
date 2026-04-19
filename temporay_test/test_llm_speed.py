import asyncio
import time
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import SystemMessage, HumanMessage

async def test_llm_speed():
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0.2)
    
    print("Testing LLM response time...")
    start = time.time()
    
    try:
        response = await asyncio.wait_for(
            llm.ainvoke([
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="Say hello briefly.")
            ]),
            timeout=10.0
        )
        elapsed = time.time() - start
        print(f"LLM responded in {elapsed:.2f} seconds")
        print(f"Response: {response.content if hasattr(response, 'content') else response}")
    except asyncio.TimeoutError:
        elapsed = time.time() - start
        print(f"LLM timed out after {elapsed:.2f} seconds")
    except Exception as e:
        elapsed = time.time() - start
        print(f"LLM error after {elapsed:.2f} seconds: {e}")

if __name__ == "__main__":
    asyncio.run(test_llm_speed())