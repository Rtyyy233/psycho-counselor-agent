"""
Psychological Counselor Web Application

FastAPI backend with WebSocket support for real-time chat.
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import asdict

# LangSmith integration
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

load_dotenv(project_root / ".env")

# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_PROJECT"] = "counselor-agent-demo"

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    File,
    UploadFile,
)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel
from langchain_deepseek import ChatDeepSeek
import tempfile
import shutil
import os

from session_manager import SessionManager, get_session_manager, ChatMessage
from top_module import SharedContext, ChatMessage as TopModuleChatMessage

# ========== Configuration ==========

WEB_HOST = os.getenv("WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.getenv("WEB_PORT", "8000"))

# ========== FastAPI App ==========

app = FastAPI(title="Psychological Counselor")

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ========== Request/Response Models ==========


class CreateSessionRequest(BaseModel):
    title: Optional[str] = None


class UpdateTitleRequest(BaseModel):
    title: str


class WebSocketMessage(BaseModel):
    type: str  # "message", "ping"
    content: Optional[str] = None
    session_id: Optional[str] = None


# ========== Session Context per WebSocket ==========


class ConnectionContext:
    """Holds state for each WebSocket connection."""

    def __init__(self, websocket: WebSocket, session_id: str):
        self.websocket = websocket
        self.session_id = session_id
        self.ctx: Optional[SharedContext] = None
        self.llm: Optional[ChatDeepSeek] = None


# ========== Helper Functions ==========


def top_module_chatmessage_to_dict(msg: TopModuleChatMessage) -> dict:
    """Convert TopModuleChatMessage to dict."""
    return {
        "role": msg.role,
        "content": msg.content,
        "timestamp": msg.timestamp.isoformat()
        if hasattr(msg.timestamp, "isoformat")
        else str(msg.timestamp),
    }


def extract_content_from_response(response) -> str:
    """
    Extract text content from various LangChain/LangGraph response types.
    Handles AIMessage, ToolMessage, AgentFinish, and other common response types.
    """
    # If it's already a string, return it
    if isinstance(response, str):
        return response

    # Check for content attribute (AIMessage, HumanMessage, etc.)
    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, str):
            return content
        else:
            return str(content)

    # Check for output attribute (AgentFinish, some tool responses)
    if hasattr(response, "output"):
        return str(response.output)

    # Check for text attribute (some message types)
    if hasattr(response, "text"):
        return str(response.text)

    # Check for result attribute (some tool responses)
    if hasattr(response, "result"):
        return str(response.result)

    # Check if it's a dict with common keys
    if isinstance(response, dict):
        # Debug logging
        print(f"DEBUG extract_content_from_response dict keys: {list(response.keys())}")

        for key in ["content", "output", "text", "result", "response", "message"]:
            if key in response:
                print(
                    f"DEBUG found key '{key}': {response[key][:100] if isinstance(response[key], str) else type(response[key])}"
                )
                return str(response[key])

        # Special case: messages list with AIMessage
        if "messages" in response and isinstance(response["messages"], list):
            print(
                f"DEBUG processing messages list, length: {len(response['messages'])}"
            )
            # Get the last message content
            for msg in reversed(response["messages"]):
                if hasattr(msg, "content"):
                    content = msg.content
                    print(
                        f"DEBUG found AIMessage with content: {content[:100] if isinstance(content, str) else type(content)}"
                    )
                    return str(content)
                elif isinstance(msg, dict) and "content" in msg:
                    print(
                        f"DEBUG found dict message with content: {msg['content'][:100]}"
                    )
                    return str(msg["content"])
                elif isinstance(msg, dict) and "text" in msg:
                    return str(msg["text"])

            # If no content found in messages, return first message as string
            if response["messages"]:
                return str(response["messages"][0])

        # Check for nested structures that might contain the actual result
        for key in ["output", "result", "response", "return_value", "data"]:
            if key in response:
                nested = response[key]
                print(f"DEBUG found nested key '{key}', type: {type(nested)}")
                return extract_content_from_response(nested)

        # If dict has only one key, maybe it's wrapped
        if len(response) == 1:
            single_value = list(response.values())[0]
            print(f"DEBUG single key dict, extracting from: {type(single_value)}")
            return extract_content_from_response(single_value)

    # Last resort: convert to string
    result = str(response)
    print(f"DEBUG extract_content_from_response last resort: {result[:200]}")
    return result


async def send_json(websocket: WebSocket, data: dict):
    """Send JSON data via WebSocket."""
    await websocket.send_json(data)


async def handle_user_message(
    ctx: ConnectionContext, content: str, manager: SessionManager
):
    """Process a user message through the counselor pipeline."""

    # 1. Add to session storage
    session = await manager.add_message(ctx.session_id, "user", content)
    if not session:
        return

    # Generate title from first message if needed
    if len(session.messages) == 1:
        title = await manager.generate_title(content)
        session = await manager.update_title(ctx.session_id, title)

    # 2. Add to context for processing
    ctx.ctx.messages.append(TopModuleChatMessage(role="user", content=content))

    # 3. Quick acknowledgment
    await send_json(
        ctx.websocket, {"type": "indicator", "agent": "chatter", "status": "typing"}
    )

    # 4. Trigger observers (non-blocking from user perspective)
    ctx.ctx.on_analyst_trigger.set()
    ctx.ctx.on_supervisor_trigger.set()

    # 5. Generate response
    from chatter import call_chatter

    response = await call_chatter(ctx.ctx)

    # 6. Add response to context and session
    ctx.ctx.messages.append(TopModuleChatMessage(role="assistant", content=response))
    await manager.add_message(ctx.session_id, "assistant", response)

    # 7. Send response to user
    await send_json(
        ctx.websocket, {"type": "message", "content": response, "sender": "assistant"}
    )

    # 8. Send context stats
    msg_count = len(ctx.ctx.messages)
    token_estimate = sum(len(m.content) for m in ctx.ctx.messages) // 4
    await send_json(
        ctx.websocket,
        {"type": "context_stats", "messages": msg_count, "tokens": token_estimate},
    )

    # 9. Check if summarization is needed
    from conversation_manager import ConversationManager

    conv_manager = ConversationManager(ctx.ctx)
    if conv_manager.should_summarize():
        summary = await conv_manager.generate_summary(ctx.ctx.messages)
        await conv_manager.store_segment(ctx.ctx.messages, summary)
        await conv_manager.reset_context_with_summary(summary)

        await send_json(
            ctx.websocket,
            {
                "type": "context_stats",
                "messages": len(ctx.ctx.messages),
                "tokens": conv_manager.estimate_tokens(ctx.ctx.messages),
                "summarized": True,
                "summary": summary.main_topic,
            },
        )


# ========== WebSocket Endpoint ==========


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket, session_id: str = None):
    """WebSocket endpoint for real-time chat."""
    await websocket.accept()

    manager = get_session_manager()

    # Get or create session
    if session_id:
        session = await manager.get_session(session_id)
        if not session:
            session = await manager.create_session()
    else:
        session = await manager.create_session()

    # Initialize context
    ctx = ConnectionContext(websocket, session.id)
    ctx.llm = ChatDeepSeek(model="deepseek-chat", temperature=0.5)
    ctx.ctx = SharedContext()

    # Load existing messages into context
    for msg in session.messages:
        ctx.ctx.messages.append(
            TopModuleChatMessage(role=msg.role, content=msg.content)
        )

    # Send session info
    await send_json(
        websocket,
        {
            "type": "session_info",
            "session_id": session.id,
            "title": session.title,
            "messages": [top_module_chatmessage_to_dict(m) for m in session.messages],
        },
    )

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "message":
                content = data.get("content", "").strip()
                if content:
                    await handle_user_message(ctx, content, manager)

            elif msg_type == "ping":
                await send_json(websocket, {"type": "pong"})

    except WebSocketDisconnect:
        pass


# ========== HTTP Endpoints ==========


@app.get("/", response_class=FileResponse)
async def root():
    """Serve the main HTML page."""
    index_path = Path(__file__).parent / "static" / "index.html"
    if index_path.exists():
        return index_path
    return HTMLResponse("<h1>Psychological Counselor</h1><p>index.html not found</p>")


@app.get("/upload", response_class=FileResponse)
async def upload_page():
    """Serve the upload page."""
    upload_path = Path(__file__).parent / "static" / "upload.html"
    if upload_path.exists():
        return upload_path
    return HTMLResponse("<h1>Upload Memory</h1><p>upload.html not found</p>")


@app.get("/api/sessions")
async def list_sessions():
    """List all sessions."""
    manager = get_session_manager()
    sessions = await manager.list_sessions()
    return [
        {
            "id": s.id,
            "title": s.title,
            "message_count": len(s.messages),
            "created_at": s.created_at,
            "updated_at": s.updated_at,
        }
        for s in sessions
    ]


@app.post("/api/sessions")
async def create_session(request: CreateSessionRequest):
    """Create a new session."""
    manager = get_session_manager()
    title = request.title or "New Conversation"
    session = await manager.create_session(title)
    return {"id": session.id, "title": session.title, "created_at": session.created_at}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get a specific session."""
    manager = get_session_manager()
    session = await manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "id": session.id,
        "title": session.title,
        "messages": [
            asdict(m) if isinstance(m, ChatMessage) else m for m in session.messages
        ],
        "created_at": session.created_at,
        "updated_at": session.updated_at,
    }


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    manager = get_session_manager()
    deleted = await manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")

    # Ensure there's always at least one session
    sessions = await manager.list_sessions()
    if not sessions:
        await manager.create_session("New Conversation")

    return {"status": "deleted"}


@app.patch("/api/sessions/{session_id}/title")
async def update_session_title(session_id: str, request: UpdateTitleRequest):
    """Update session title."""
    manager = get_session_manager()
    session = await manager.update_title(session_id, request.title)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"id": session.id, "title": session.title}


@app.post("/api/sessions/{session_id}/clear")
async def clear_session_messages(session_id: str):
    """Clear all messages in a session."""
    manager = get_session_manager()
    session = await manager.clear_messages(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"id": session.id, "cleared": True}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file (diary, material, conversation) to memory storage."""
    # Save uploaded file to temp location
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    try:
        # Write uploaded content
        with open(temp_path, "wb") as f:
            content = await file.read()
            # Validate file size (max 10MB)
            max_file_size = 10 * 1024 * 1024  # 10MB
            if len(content) > max_file_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"文件大小超过限制（最大10MB），当前文件大小：{len(content) / 1024 / 1024:.2f}MB",
                )
            f.write(content)

        # First ensure file exists and is readable
        if not os.path.exists(temp_path):
            raise HTTPException(status_code=400, detail=f"File not found: {temp_path}")

        print(
            f"Attempting to store file: {temp_path}, exists: {os.path.exists(temp_path)}"
        )

        storage_result = None
        analysis_result = None

        try:
            # Determine file type based on content and call appropriate storage function
            import asyncio

            # First read a sample of the file to determine type
            file_content_sample = ""
            try:
                with open(temp_path, "rb") as f:
                    raw_content = f.read(
                        5000
                    )  # Read first 5000 bytes for type detection

                # Try different encodings
                for encoding in ["utf-8", "gbk", "latin-1"]:
                    try:
                        file_content_sample = raw_content.decode(encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    file_content_sample = raw_content.decode("utf-8", errors="ignore")
            except Exception as e:
                print(f"Error reading file sample: {e}")
                file_content_sample = ""

            # Determine storage type based on content patterns
            import re

            # Check for diary patterns (date patterns, personal reflection)
            date_pattern = (
                r"\d{4}[-./]\d{1,2}[-./]\d{1,2}|\d{1,2}[-./]\d{1,2}[-./]\d{4}"
            )
            diary_keywords = [
                "今天",
                "昨天",
                "早上",
                "晚上",
                "心情",
                "感觉",
                "日记",
                "记录",
                "我",
                "我的",
                "自己",
                "今天",
                "明日",
                "本周",
                "最近",
                "心事",
                "情绪",
                "想法",
            ]

            is_diary = False
            if re.search(date_pattern, file_content_sample):
                is_diary = True
            elif any(keyword in file_content_sample for keyword in diary_keywords):
                is_diary = True

            # Check for conversation outline patterns (PAIP format, counseling terms)
            conv_keywords = [
                "问题",
                "评估",
                "干预",
                "计划",
                "咨询",
                "对话",
                "session",
                "client",
                "患者",
                "来访者",
                "咨询师",
                "therapy",
                "counseling",
                "treatment",
            ]
            # Also check for structured sections (e.g., "问题:", "评估:", etc.)
            section_pattern = r"(问题|评估|干预|计划)[:：]"
            has_section = bool(re.search(section_pattern, file_content_sample))

            is_conversation = has_section or any(
                keyword in file_content_sample for keyword in conv_keywords
            )

            # Decide which storage function to use
            if is_diary:
                print(f"Detected as diary, storing with store_diary")
                from mem_store_diary import store_diary

                print(f"DEBUG: Calling store_diary.ainvoke with file: {temp_path}")
                raw_storage_result = await asyncio.wait_for(
                    store_diary.ainvoke({"file_path": temp_path}), timeout=60.0
                )
            elif is_conversation:
                print(
                    f"Detected as conversation outline, storing with store_conversation_outline"
                )
                from mem_store_conv_outline import store_conversation_outline

                print(
                    f"DEBUG: Calling store_conversation_outline.ainvoke with file: {temp_path}"
                )
                raw_storage_result = await asyncio.wait_for(
                    store_conversation_outline.ainvoke({"file_path": temp_path}),
                    timeout=60.0,
                )
                from mem_store_conv_outline import store_conversation_outline

                print(
                    f"DEBUG: Calling store_conversation_outline with file: {temp_path}"
                )
                raw_storage_result = await asyncio.wait_for(
                    store_conversation_outline(file_path=temp_path),
                    timeout=60.0,
                )
            else:
                print(f"Detected as material, storing with store_materials")
                from mem_store_material import store_materials

                print(f"DEBUG: Calling store_materials.ainvoke with file: {temp_path}")
                raw_storage_result = await asyncio.wait_for(
                    store_materials.ainvoke({"file_path": temp_path}), timeout=60.0
                )
                print(f"DEBUG: raw_storage_result type: {type(raw_storage_result)}")
                print(
                    f"DEBUG: raw_storage_result repr: {repr(raw_storage_result)[:500]}"
                )

            # Extract content from storage response
            storage_result = extract_content_from_response(raw_storage_result)
            print(f"Storage result: {storage_result}")

        except asyncio.TimeoutError:
            error_msg = "存储操作超时（60秒），请稍后重试或检查文件大小"
            print(error_msg)
            storage_result = error_msg
        except Exception as e:
            print(f"Storage error: {e}")
            storage_result = f"存储失败: {str(e)}"

        # Trigger analysis of the newly stored content
        analysis_result = None
        try:
            # Read file content for analysis (with size limit)
            max_file_size = 1024 * 1024  # 1MB limit
            file_size = os.path.getsize(temp_path)

            if file_size > max_file_size:
                analysis_result = "文件过大，跳过分析（超过1MB限制）"
            else:
                # Try to read as text file
                try:
                    with open(temp_path, "rb") as f:
                        raw_content = f.read()

                    # Try UTF-8, fallback to other encodings
                    for encoding in ["utf-8", "gbk", "latin-1"]:
                        try:
                            file_content = raw_content.decode(encoding)
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        file_content = raw_content.decode("utf-8", errors="ignore")

                    # Use LLM directly to analyze the file content (skip agent to avoid graph errors)
                    try:
                        from langchain_deepseek import ChatDeepSeek
                        from langchain_core.messages import SystemMessage, HumanMessage

                        # Limit content length for analysis
                        content_preview = file_content[:2000]
                        if len(file_content) > 2000:
                            content_preview += (
                                f"...（共{len(file_content)}字符，已截断）"
                            )

                        analysis_prompt = f"""请分析以下新存储的文件内容：

文件内容：
{content_preview}

请从心理分析角度分析：
1. 文件中表达的主要情绪和主题
2. 潜在的认知和行为模式
3. 与现有记忆的可能关联
4. 建议的探索方向"""

                        # Create a simple LLM instance for analysis
                        analysis_llm = ChatDeepSeek(
                            model="deepseek-chat", temperature=0.2
                        )

                        analysis_response = await asyncio.wait_for(
                            analysis_llm.ainvoke(
                                [
                                    SystemMessage(
                                        content="你是一位专业的心理分析专家。请分析新存储的文件内容。"
                                    ),
                                    HumanMessage(content=analysis_prompt),
                                ]
                            ),
                            timeout=30.0,  # 30秒超时
                        )

                        analysis_result = extract_content_from_response(
                            analysis_response
                        )

                    except asyncio.TimeoutError:
                        analysis_result = "分析超时（30秒），跳过分析"
                        print(analysis_result)
                    except Exception as e:
                        # Catch LLM-specific errors
                        error_msg = str(e)
                        if "semantic_search_node" in error_msg:
                            analysis_result = (
                                "分析过程中出现LangGraph配置错误，跳过分析"
                            )
                        elif (
                            "Connection" in error_msg or "timeout" in error_msg.lower()
                        ):
                            analysis_result = "连接到分析服务失败，跳过分析"
                        else:
                            analysis_result = f"分析错误: {error_msg[:100]}"
                        print(f"LLM analysis error: {e}")

                except asyncio.TimeoutError:
                    analysis_result = "分析超时（30秒），跳过分析"
                    print(analysis_result)
                except Exception as e:
                    analysis_result = f"文件读取错误: {str(e)}"

        except Exception as e:
            print(f"Analysis error: {e}")
            analysis_result = f"分析过程中出现错误: {str(e)}"

        # Format storage result for better display
        formatted_storage_result = storage_result
        if isinstance(storage_result, list):
            if len(storage_result) == 1:
                formatted_storage_result = f"存储成功，生成1个子块: {storage_result[0]}"
            else:
                formatted_storage_result = f"存储成功，生成{len(storage_result)}个子块"
        elif isinstance(storage_result, str) and storage_result.startswith("["):
            # Already a string representation of list
            try:
                import ast

                parsed_list = ast.literal_eval(storage_result)
                if isinstance(parsed_list, list):
                    if len(parsed_list) == 1:
                        formatted_storage_result = (
                            f"存储成功，生成1个子块: {parsed_list[0]}"
                        )
                    else:
                        formatted_storage_result = (
                            f"存储成功，生成{len(parsed_list)}个子块"
                        )
            except:
                pass  # Keep original if parsing fails

        return JSONResponse(
            {
                "status": "stored",
                "storage_result": formatted_storage_result,
                "analysis": analysis_result,
            }
        )
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


# ========== Run Server ==========


def run():
    """Run the server."""
    import uvicorn

    uvicorn.run(app, host=WEB_HOST, port=WEB_PORT)


if __name__ == "__main__":
    run()
