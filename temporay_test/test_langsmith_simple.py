import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("LANGCHAIN_API_KEY")
print(f"API Key: {api_key[:15]}..." if api_key else "No API key")

if not api_key:
    print("ERROR: LANGCHAIN_API_KEY not set")
    exit(1)

# Try different API endpoints
endpoints = [
    "https://api.smith.langchain.com",
    "https://api.langchain.com",
    "https://smith.langchain.com/api",
]

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

for base_url in endpoints:
    print(f"\nTrying endpoint: {base_url}")
    
    # Test simple health endpoint
    try:
        response = requests.get(
            f"{base_url}/health",
            headers=headers,
            timeout=5
        )
        print(f"  /health: {response.status_code} - {response.text[:100]}")
    except Exception as e:
        print(f"  /health: Error - {e}")
    
    # Test API version endpoint
    try:
        response = requests.get(
            f"{base_url}/api/v1/version",
            headers=headers,
            timeout=5
        )
        print(f"  /api/v1/version: {response.status_code} - {response.text[:100]}")
    except Exception as e:
        print(f"  /api/v1/version: Error - {e}")
    
    # Test projects endpoint (most common)
    try:
        response = requests.get(
            f"{base_url}/api/v1/projects",
            headers=headers,
            timeout=5
        )
        print(f"  /api/v1/projects: {response.status_code}")
        if response.status_code == 200:
            print(f"    Success! Found {len(response.json())} projects")
            break
    except Exception as e:
        print(f"  /api/v1/projects: Error - {e}")

# Check environment
print("\nEnvironment variables:")
for key in ['LANGCHAIN_TRACING_V2', 'LANGCHAIN_API_KEY', 'LANGCHAIN_PROJECT', 'LANGCHAIN_ENDPOINT']:
    val = os.getenv(key)
    if val:
        print(f"  {key}: {'*' * 10 if 'KEY' in key else val}")
    else:
        print(f"  {key}: NOT SET")

# Try to import langsmith and check version
try:
    import langsmith
    print(f"\nLangSmith version: {langsmith.__version__}")
    
    # Check client creation
    client = langsmith.Client()
    print("LangSmith client created")
    
    # Try a simple operation
    try:
        # In newer versions, this might work
        result = client.list_projects(limit=1)
        print(f"List projects successful, found {len(list(result))} projects")
    except Exception as e:
        print(f"list_projects error (might be expected): {e}")
        
except Exception as e:
    print(f"\nLangSmith import/init error: {e}")