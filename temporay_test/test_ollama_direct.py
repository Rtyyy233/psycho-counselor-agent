import requests
import json
import time

def test_ollama_chat():
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen3.5:4b",
        "prompt": "Say hello briefly.",
        "stream": False
    }
    
    print(f"Testing Ollama chat API with model: {payload['model']}")
    start = time.time()
    try:
        response = requests.post(url, json=payload, timeout=10)
        elapsed = time.time() - start
        print(f"Response in {elapsed:.2f}s, status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {result.get('response', 'No response')[:100]}")
        else:
            print(f"Error: {response.text[:200]}")
    except requests.exceptions.Timeout:
        elapsed = time.time() - start
        print(f"Request timeout after {elapsed:.2f}s")
    except Exception as e:
        elapsed = time.time() - start
        print(f"Error after {elapsed:.2f}s: {e}")

if __name__ == "__main__":
    test_ollama_chat()