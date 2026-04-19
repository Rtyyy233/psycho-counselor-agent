import os
from dotenv import load_dotenv
import langchain

load_dotenv()

print("Testing LangSmith with simpler imports...")
print(f"LangChain version: {langchain.__version__}")

# Try different import paths
try:
    from langsmith import Client
    print("Successfully imported langsmith.Client")
    
    # Test connection
    client = Client()
    print("LangSmith client created")
    
    # Try to get current user
    try:
        user = client.get_current_user()
        print(f"Connected to LangSmith as: {user.username}")
    except Exception as e:
        print(f"Error getting user (but client created): {e}")
        
except ImportError as e:
    print(f"langsmith import error: {e}")
    print("Trying langchain_core.tracers...")
    try:
        from langchain_core.tracers.langchain import LangChainTracer
        print("Successfully imported LangChainTracer from langchain_core")
    except ImportError as e2:
        print(f"langchain_core import error: {e2}")

# Try to enable tracing via environment
print("\nSetting environment variables...")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "default")

# Test if langchain will use tracing
try:
    from langchain.globals import set_tracer
    from langchain.callbacks.tracers import LangChainTracer
    
    tracer = LangChainTracer()
    set_tracer(tracer)
    print("Tracer set via langchain.globals")
except Exception as e:
    print(f"Error setting tracer: {e}")
    print("Trying alternative import...")
    try:
        import langchain.schema
        print("langchain.schema imported")
    except Exception as e2:
        print(f"Alternative import failed: {e2}")