"""
Top Module: Async Observer Pattern for Psychological Counselor

User ↔ Chatter (always responsive)
         ↑ prompt enriched by
Analyst (background, injects when triggered)
Supervisor (background, injects when triggered)
"""
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Literal

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, AIMessage

from chatter import call_chatter
from analysist import call_analysist
from supervisoner import supervisoner_ainvoke  # TODO: fix typo -> supervisor


# ========== Shared Context ==========

@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PromptInjection:
    source: str  # "analyst" or "supervisor"
    content: str
    priority: Literal["gentle", "important", "critical"] = "gentle"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SharedContext:
    """Thread-safe shared context for all agents."""
    messages: list[ChatMessage] = field(default_factory=list)
    current_topic: str = ""
    topic_history: list[str] = field(default_factory=list)

    # Injections for Chatter's next LLM call
    analyst_injection: Optional[PromptInjection] = None
    supervisor_injection: Optional[PromptInjection] = None

    # Control events
    on_new_message = asyncio.Event()
    on_analyst_trigger = asyncio.Event()
    on_supervisor_trigger = asyncio.Event()

    # Locks for thread-safe access
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def safe_set_analyst(self, content: str, priority: str = "important"):
        async with self._lock:
            self.analyst_injection = PromptInjection(
                source="analyst",
                content=content,
                priority=priority
            )

    async def safe_set_supervisor(self, content: str, priority: str = "important"):
        async with self._lock:
            self.supervisor_injection = PromptInjection(
                source="supervisor",
                content=content,
                priority=priority
            )

    async def add_message(self, role: str, content: str):
        async with self._lock:
            self.messages.append(ChatMessage(role=role, content=content))
            self.on_new_message.set()

    async def get_recent_messages(self, n: int = 5) -> list[ChatMessage]:
        async with self._lock:
            return self.messages[-n:]


# ========== Background Observers ==========

async def analyst_observer(ctx: SharedContext, llm):
    """Background analyst - monitors and injects when triggered."""
    while True:
        await ctx.on_analyst_trigger.wait()
        ctx.on_analyst_trigger.clear()

        # Build query from recent messages
        recent = await ctx.get_recent_messages(5)
        if len(recent) < 2:
            continue

        # Call analyst (runs retrieve modules internally)
        try:
            analysis_result = await call_analysist(ctx)
            if analysis_result:
                await ctx.safe_set_analyst(
                    analysis_result,
                    priority="high"
                )
        except Exception as e:
            print(f"Analyst error: {e}")


async def supervisor_observer(ctx: SharedContext, llm):
    """Background supervisor - observes conversation and injects guidance."""
    while True:
        await ctx.on_supervisor_trigger.wait()
        ctx.on_supervisor_trigger.clear()

        # Analyze conversation state
        recent = await ctx.get_recent_messages(3)
        if len(recent) < 2:
            continue

        try:
            # Supervisor decides if guidance is needed
            guidance = await supervisoner_ainvoke({
                "messages": [{"role": m.role, "content": m.content} for m in recent],
                "context": {"current_topic": ctx.current_topic}
            })
            if guidance:
                await ctx.safe_set_supervisor(
                    guidance,
                    priority="important"
                )
        except Exception as e:
            print(f"Supervisor error: {e}")


# ========== Main Conversation Loop ==========

class PsychologicalCounselor:
    """Main orchestrator - Chatter + background observers."""

    def __init__(self):
        self.ctx = SharedContext()
        self.llm = ChatDeepSeek(model="deepseek-chat", temperature=0.5)

        # Background observer tasks
        self._analyst_task: Optional[asyncio.Task] = None
        self._supervisor_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the conversation loop with background observers."""
        # Start background observers
        self._analyst_task = asyncio.create_task(analyst_observer(self.ctx, self.llm))
        self._supervisor_task = asyncio.create_task(supervisor_observer(self.ctx, self.llm))

        print("Psychological Counselor started. Type 'quit' to exit.")

        # Conversation loop
        while True:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, input, "You: "
                )
                if user_input.lower() == "quit":
                    break

                response = await self.handle_message(user_input)
                print(f"Chatter: {response}")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

        # Cleanup
        self._analyst_task.cancel()
        self._supervisor_task.cancel()

    async def handle_message(self, user_input: str) -> str:
        """
        Handle user message - always responsive, background observers run in parallel.
        """
        # 1. Update context immediately (non-blocking for user)
        await self.ctx.add_message("user", user_input)

        # Extract topic
        topic = self.extract_topic(user_input)
        async with self.ctx._lock:
            self.ctx.topic_history.append(self.ctx.current_topic)
            self.ctx.current_topic = topic

        # 2. Trigger background observers (fire and forget)
        self.ctx.on_analyst_trigger.set()
        self.ctx.on_supervisor_trigger.set()

        # 3. Generate immediate response (with any pending injections)
        response = await call_chatter(self.ctx)

        # 4. Add assistant response to context
        await self.ctx.add_message("assistant", response)

        return response

    def extract_topic(self, text: str) -> str:
        """Simple topic extraction - could use LLM for better accuracy."""
        # TODO: implement proper topic extraction
        if not text:
            return ""
        return text[:50]  # Simple placeholder


# ========== Standalone Function for Testing ==========

async def main_graph():
    """Standalone test entry point."""
    counselor = PsychologicalCounselor()
    await counselor.start()


# Alias for backwards compatibility
main = main_graph