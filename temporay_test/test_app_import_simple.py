import os
import sys
from pathlib import Path

# Simulate the app startup
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Load environment
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

# Set LangSmith env vars as in main.py
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "main")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

print("Environment set up. Testing imports...")

# Try to import key modules
try:
    from langchain_deepseek import ChatDeepSeek
    print("OK: ChatDeepSeek imported")
    
    # Try creating an instance
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0.5)
    print("OK: ChatDeepSeek instance created")
    
except Exception as e:
    print(f"ERROR: ChatDeepSeek error: {e}")
    import traceback
    traceback.print_exc()

try:
    from langchain_ollama import OllamaEmbeddings
    print("OK: OllamaEmbeddings imported")
except Exception as e:
    print(f"ERROR: OllamaEmbeddings error: {e}")

try:
    from top_module import SharedContext, analyst_observer, supervisor_observer
    print("OK: top_module imports successful")
except Exception as e:
    print(f"ERROR: top_module error: {e}")
    import traceback
    traceback.print_exc()

try:
    from analysist import call_analysist
    print("OK: analysist imported")
except Exception as e:
    print(f"ERROR: analysist error: {e}")

try:
    from chatter import call_chatter
    print("OK: chatter imported")
except Exception as e:
    print(f"ERROR: chatter error: {e}")

print("\nChecking LangSmith configuration...")
print(f"LANGCHAIN_TRACING_V2: {os.environ.get('LANGCHAIN_TRACING_V2')}")
print(f"LANGCHAIN_PROJECT: {os.environ.get('LANGCHAIN_PROJECT')}")
print(f"LANGCHAIN_API_KEY present: {'LANGCHAIN_API_KEY' in os.environ}")

# Test if langsmith client can be created
try:
    import langsmith
    client = langsmith.Client()
    print("OK: LangSmith client created")
    
    # Try to list runs (should work if tracing is enabled)
    try:
        runs = list(client.list_runs(limit=1))
        print(f"OK: Can list runs (found {len(runs)} runs)")
    except Exception as e:
        print(f"WARNING: Cannot list runs (may be normal): {e}")
        
except Exception as e:
    print(f"ERROR: LangSmith client error: {e}")
    import traceback
    traceback.print_exc()