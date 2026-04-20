import asyncio
import websockets
import json
import time

async def test_websocket():
    uri = "ws://localhost:8000/ws"
    
    print(f"Connecting to {uri}")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected")
            
            # Send initial message to start session
            message = {
                "type": "message",
                "content": "Hello, I feel anxious today.",
                "session_id": "test-session-123"
            }
            
            print(f"Sending: {message}")
            await websocket.send(json.dumps(message))
            
            # Wait for response
            print("Waiting for response...")
            response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            print(f"Received: {response[:500]}")
            
            # Parse response
            data = json.loads(response)
            if data.get("type") == "message":
                print(f"\nChatter response: {data.get('content', '')[:200]}...")
                # Check if response seems to include analysis
                response_text = data.get('content', '')
                if "分析" in response_text or "建议" in response_text or "情绪" in response_text:
                    print("✓ Response appears to include analysis (contains analysis keywords)")
                else:
                    print("✗ Response may not include analysis")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())