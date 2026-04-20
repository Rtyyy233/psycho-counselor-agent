import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set environment variables for tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = "test-tracing"
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

print("Testing LangSmith tracing with simple LLM call...")
print(f"LANGCHAIN_TRACING_V2: {os.environ.get('LANGCHAIN_TRACING_V2')}")
print(f"LANGCHAIN_PROJECT: {os.environ.get('LANGCHAIN_PROJECT')}")
print(f"LANGCHAIN_ENDPOINT: {os.environ.get('LANGCHAIN_ENDPOINT')}")

# Import after setting env vars
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage

async def test_tracing():
    try:
        # Create LLM instance
        llm = ChatDeepSeek(model="deepseek-chat", temperature=0.1)
        
        print("\nMaking LLM call (this should generate a trace)...")
        
        # Make a simple call
        response = await llm.ainvoke([
            HumanMessage(content="Say 'Hello, tracing test!'")
        ])
        
        print(f"LLM response: {response.content}")
        print("\nIf tracing is working, you should see this run in LangSmith:")
        print("1. Go to https://smith.langchain.com")
        print("2. Select project 'test-tracing'")
        print("3. Look for a trace with 'ChatDeepSeek'")
        
        # Try to check if tracing was enabled
        try:
            from langsmith.run_helpers import traceable
            print("\ntraceable decorator is available")
        except ImportError:
            print("\ntraceable not available")
            
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()

# Also test with ChatOllama as alternative
async def test_ollama_tracing():
    try:
        from langchain_ollama import ChatOllama
        print("\n\nTesting with ChatOllama...")
        
        llm = ChatOllama(model="qwen3.5:4b", temperature=0.1)
        response = await llm.ainvoke("Say 'Hello from Ollama tracing test!'")
        print(f"Ollama response: {response.content}")
    except ImportError:
        print("\nChatOllama not available")
    except Exception as e:
        print(f"Ollama test error: {e}")

if __name__ == "__main__":
    asyncio.run(test_tracing())
    # asyncio.run(test_ollama_tracing())