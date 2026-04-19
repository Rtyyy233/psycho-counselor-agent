import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("Testing LangSmith configuration...")
print(f"LANGCHAIN_TRACING_V2: {os.getenv('LANGCHAIN_TRACING_V2')}")
print(f"LANGCHAIN_API_KEY present: {'LANGCHAIN_API_KEY' in os.environ}")
print(f"LANGCHAIN_PROJECT: {os.getenv('LANGCHAIN_PROJECT')}")

# Try to import langchain and test tracing
try:
    import langchain
    print(f"LangChain version: {langchain.__version__}")
    
    # Check if tracing is enabled
    from langchain.callbacks.tracers.langchain import LangChainTracer
    print("LangChainTracer import successful")
    
    # Try to create a simple tracer
    try:
        tracer = LangChainTracer()
        print("Tracer created successfully")
    except Exception as e:
        print(f"Error creating tracer: {e}")
        
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"General error: {e}")

# Check if there are any environment variable issues
print("\nFull environment check:")
for key in ['LANGCHAIN_TRACING_V2', 'LANGCHAIN_API_KEY', 'LANGCHAIN_PROJECT', 'LANGCHAIN_ENDPOINT']:
    value = os.getenv(key)
    if value:
        print(f"{key}: {value[:20]}..." if len(str(value)) > 20 else f"{key}: {value}")
    else:
        print(f"{key}: NOT SET")