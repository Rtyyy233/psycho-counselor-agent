import asyncio
import time

try:
    from langchain_ollama import ChatOllama
    print("ChatOllama import successful")
    
    async def test():
        llm = ChatOllama(model="qwen3.5:latest", temperature=0.2)
        print("Testing ChatOllama with qwen3.5:latest...")
        start = time.time()
        try:
            response = await asyncio.wait_for(
                llm.ainvoke("Say hello briefly."),
                timeout=5.0
            )
            elapsed = time.time() - start
            print(f"Response in {elapsed:.2f}s: {response}")
        except asyncio.TimeoutError:
            elapsed = time.time() - start
            print(f"Timeout after {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - start
            print(f"Error after {elapsed:.2f}s: {e}")
    
    asyncio.run(test())
except ImportError as e:
    print(f"Import error: {e}")