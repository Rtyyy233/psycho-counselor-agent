"""
Session Manager - Handles conversation session CRUD and local storage.

Sessions are stored as JSON files in the sessions/ directory.
"""

import asyncio
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field, asdict


@dataclass
class ChatMessage:
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Session:
    id: str
    title: str
    messages: List[ChatMessage] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    summary: Optional[str] = None

    @staticmethod
    def from_dict(data: dict) -> "Session":
        messages = [
            ChatMessage(**m) if isinstance(m, dict) else m
            for m in data.get("messages", [])
        ]
        return Session(
            id=data["id"],
            title=data["title"],
            messages=messages,
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            summary=data.get("summary"),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "messages": [
                asdict(m) if isinstance(m, ChatMessage) else m for m in self.messages
            ],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "summary": self.summary,
        }


class SessionManager:
    """Manages session CRUD operations with local JSON storage."""

    def __init__(self, sessions_dir: str = None):
        if sessions_dir is None:
            # Default to src/web/sessions
            project_root = Path(__file__).parent.parent.parent
            sessions_dir = project_root / "web" / "sessions"
        else:
            sessions_dir = Path(sessions_dir)

        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        # Ensure there's always at least one session
        if not list(self.sessions_dir.glob("*.json")):
            self.create_session("New Conversation")

    # ===== CRUD Operations =====

    async def list_sessions(self) -> List[Session]:
        """List all sessions, sorted by updated_at descending."""
        sessions = []
        for filepath in self.sessions_dir.glob("*.json"):
            try:
                data = json.loads(filepath.read_text(encoding="utf-8"))
                sessions.append(Session.from_dict(data))
            except Exception:
                continue

        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get a specific session by ID."""
        filepath = self.sessions_dir / f"{session_id}.json"
        if not filepath.exists():
            return None

        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
            return Session.from_dict(data)
        except Exception:
            return None

    async def create_session(self, title: str = "New Conversation") -> Session:
        """Create a new session."""
        session = Session(
            id=str(uuid.uuid4()),
            title=title,
            messages=[],
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )

        await self.save_session(session)
        return session

    async def save_session(self, session: Session) -> None:
        """Save a session to disk."""
        session.updated_at = datetime.now().isoformat()
        filepath = self.sessions_dir / f"{session.id}.json"
        filepath.write_text(
            json.dumps(session.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        filepath = self.sessions_dir / f"{session_id}.json"
        if filepath.exists():
            filepath.unlink()
            return True
        return False

    async def add_message(
        self, session_id: str, role: str, content: str
    ) -> Optional[Session]:
        """Add a message to a session."""
        session = await self.get_session(session_id)
        if not session:
            return None

        session.messages.append(ChatMessage(role=role, content=content))
        await self.save_session(session)
        return session

    async def update_title(self, session_id: str, title: str) -> Optional[Session]:
        """Update session title."""
        session = await self.get_session(session_id)
        if not session:
            return None

        session.title = title
        await self.save_session(session)
        return session

    async def generate_title(self, first_message: str) -> str:
        """Generate a title from the first message."""
        # Simple truncation for now
        if len(first_message) <= 30:
            return first_message
        return first_message[:30] + "..."

    async def clear_messages(self, session_id: str) -> Optional[Session]:
        """Clear all messages in a session."""
        session = await self.get_session(session_id)
        if not session:
            return None

        session.messages = []
        await self.save_session(session)
        return session


# Singleton instance
_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    global _manager
    if _manager is None:
        _manager = SessionManager()
    return _manager
