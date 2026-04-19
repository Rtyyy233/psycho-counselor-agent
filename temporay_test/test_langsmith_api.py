import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("LANGCHAIN_API_KEY")
print(f"API Key: {api_key[:15]}..." if api_key else "No API key")

if not api_key:
    print("ERROR: LANGCHAIN_API_KEY not set")
    exit(1)

# LangSmith API endpoints
base_url = "https://api.smith.langchain.com"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

print("\nTesting LangSmith API connection...")

# Test 1: Check if API key is valid
try:
    # Try to get projects
    response = requests.get(
        f"{base_url}/api/v1/projects",
        headers=headers,
        timeout=10
    )
    
    print(f"Status code: {response.status_code}")
    
    if response.status_code == 200:
        print("✓ API key is valid")
        projects = response.json()
        print(f"Found {len(projects)} projects")
        for project in projects[:3]:
            print(f"  - {project.get('name', 'Unnamed')} (ID: {project.get('id', 'N/A')})")
    elif response.status_code == 401:
        print("✗ API key is invalid or expired")
        print(f"Response: {response.text[:200]}")
    else:
        print(f"Unexpected response: {response.status_code}")
        print(f"Response: {response.text[:200]}")
        
except requests.exceptions.ConnectionError:
    print("✗ Connection failed - check network or API endpoint")
except requests.exceptions.Timeout:
    print("✗ Request timeout")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: Check tracing configuration
print("\nChecking tracing configuration...")
print(f"LANGCHAIN_TRACING_V2: {os.getenv('LANGCHAIN_TRACING_V2')}")
print(f"LANGCHAIN_PROJECT: {os.getenv('LANGCHAIN_PROJECT')}")

# Check if the project exists
project_name = os.getenv("LANGCHAIN_PROJECT", "main")
try:
    response = requests.get(
        f"{base_url}/api/v1/projects",
        headers=headers,
        params={"name": project_name},
        timeout=10
    )
    
    if response.status_code == 200:
        projects = response.json()
        if projects:
            print(f"✓ Project '{project_name}' exists")
        else:
            print(f"⚠ Project '{project_name}' not found - will be created on first trace")
    else:
        print(f"⚠ Could not verify project (status: {response.status_code})")
        
except Exception as e:
    print(f"⚠ Could not check project: {e}")