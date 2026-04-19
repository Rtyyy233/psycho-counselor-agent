import asyncio
import time
import json
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class PromptInjection:
    """注入内容容器，包含来源和时间戳便于追踪"""
    content: str
    timestamp: float
    source: str  # "analyst" 或 "supervisor"
    
    def is_expired(self, timeout_seconds: float = 300) -> bool:
        """检查注入是否过期（默认5分钟）"""
        return (time.time() - self.timestamp) > timeout_seconds

class SharedContext:
    """
    线程安全的对话上下文容器
    设计原则：
    1. 所有公共方法都是异步的
    2. 所有状态修改都通过锁保护
    3. 提供原子性复合操作
    """
    
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self._lock = asyncio.Lock()  # 核心：异步锁
        
        # 受保护的状态变量
        self._messages: List[Dict] = []
        self._analyst_injection: Optional[PromptInjection] = None
        self._supervisor_injection: Optional[PromptInjection] = None
        
        # 异步事件（用于触发后台Agent）
        self.analyst_trigger = asyncio.Event()
        self.supervisor_trigger = asyncio.Event()

        self.analysist_spare: bool = True
        self.supervisor_spare: bool = True
        
        # 统计信息（用于监控）
        self._stats = {
            "message_count": 0,
            "analyst_injections": 0,
            "supervisor_injections": 0,
            "last_updated": time.time()
        }
    
    # ========== 消息管理 ==========
    
    async def add_message(self, role: str, content: str) -> None:
        """
        添加消息（原子操作）
        自动清理过期消息，防止内存泄漏
        """
        async with self._lock:
            # 清理7天前的旧消息
            cutoff = time.time() - (7 * 24 * 3600)
            self._messages = [
                m for m in self._messages 
                if m.get("timestamp", 0) > cutoff
            ]
            
            # 添加新消息
            msg = {
                "role": role,
                "content": content,
                "timestamp": time.time(),
                "message_id": f"msg_{len(self._messages)}_{int(time.time())}"
            }
            self._messages.append(msg)
            self._stats["message_count"] += 1
            self._stats["last_updated"] = time.time()
            
            # 用户消息自动触发分析
            if role == "user":
                self.analyst_trigger.set()
                logger.debug(f"用户消息触发Analyst: {content[:50]}...")
                self.supervisor_trigger.set()
                logger.debug(f"用户消息触发Supervisor: {content[:50]}...")
                
    
    async def get_recent_messages(self, n: int = 5) -> List[Dict]:
        """获取最近n条消息（复制返回，防止外部修改）"""
        async with self._lock:
            # 返回副本，防止外部代码修改内部状态
            return self._messages[-n:].copy() if self._messages else []
    
    async def get_all_messages(self) -> List[Dict]:
        """获取所有消息（复制返回）"""
        async with self._lock:
            return self._messages.copy()
    
    # ========== Injection管理 ==========
    
    async def set_analyst_injection(self, content: str) -> None:
        """设置Analyst注入（带验证）"""
        if not content or len(content.strip()) < 5:
            logger.warning(f"Analyst注入内容过短: {content}")
            return
            
        async with self._lock:
            injection = PromptInjection(
                content=content.strip(),
                timestamp=time.time(),
                source="analyst"
            )
            self._analyst_injection = injection
            self._stats["analyst_injections"] += 1
            logger.info(f"Analyst注入设置: {len(content)}字符")
    
    async def set_supervisor_injection(self, content: str) -> None:
        """设置Supervisor注入"""
        async with self._lock:
            injection = PromptInjection(
                content=content.strip(),
                timestamp=time.time(),
                source="supervisor"
            )
            self._supervisor_injection = injection
            self._stats["supervisor_injections"] += 1
            logger.info(f"Supervisor注入设置: {len(content)}字符")
    
    async def get_and_clear_injections(self) -> Dict[str, Optional[str]]:
        """
        原子操作：获取并清空所有injections
        防止Chatter读取后，另一个任务又读取同一内容
        """
        async with self._lock:
            result = {
                "analyst": self._analyst_injection.content 
                    if self._analyst_injection else None,
                "supervisor": self._supervisor_injection.content 
                    if self._supervisor_injection else None
            }
            
            # 清理过期injections
            now = time.time()
            if (self._analyst_injection and 
                (now - self._analyst_injection.timestamp) > 300):
                self._analyst_injection = None
                
            if (self._supervisor_injection and 
                (now - self._supervisor_injection.timestamp) > 300):
                self._supervisor_injection = None
            
            # 清空（消费后）
            self._analyst_injection = None
            self._supervisor_injection = None
            
            return result
    
    async def peek_injections(self) -> Dict[str, Optional[str]]:
        """查看injections但不消费（用于监控）"""
        async with self._lock:
            return {
                "analyst": self._analyst_injection.content 
                    if self._analyst_injection else None,
                "supervisor": self._supervisor_injection.content 
                    if self._supervisor_injection else None
            }
    
    # ========== 会话持久化 ==========
    
    async def to_dict(self) -> Dict[str, Any]:
        """将上下文转换为字典，用于持久化"""
        async with self._lock:
            # 获取所有消息
            messages = self._messages.copy()
            
            # 转换时间戳为ISO格式字符串
            formatted_messages = []
            for msg in messages:
                formatted_msg = msg.copy()
                # 将时间戳转换为ISO格式字符串
                timestamp = msg.get("timestamp")
                if isinstance(timestamp, (int, float)):
                    formatted_msg["timestamp"] = datetime.fromtimestamp(timestamp).isoformat()
                formatted_messages.append(formatted_msg)
            
            # 构建会话数据
            session_data = {
                "id": self.session_id,
                "title": self._generate_title(formatted_messages),
                "messages": formatted_messages,
                "created_at": self._get_creation_time(formatted_messages),
                "updated_at": datetime.now().isoformat(),
                "summary": None,  # 可以后续添加摘要生成功能
                "stats": self._stats.copy()
            }
            
            return session_data
    
    def _generate_title(self, messages: List[Dict]) -> str:
        """根据消息生成会话标题"""
        if not messages:
            return "新对话"
        
        # 查找第一条用户消息作为标题
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                # 取前30个字符作为标题
                title = content[:30].strip()
                if len(content) > 30:
                    title += "..."
                return title if title else "新对话"
        
        return "新对话"
    
    def _get_creation_time(self, messages: List[Dict]) -> str:
        """获取创建时间（第一条消息的时间）"""
        if messages:
            first_msg = messages[0]
            timestamp = first_msg.get("timestamp")
            if isinstance(timestamp, str):
                return timestamp
            elif isinstance(timestamp, (int, float)):
                return datetime.fromtimestamp(timestamp).isoformat()
        
        return datetime.now().isoformat()
    
    async def save_to_file(self) -> bool:
        """保存会话到文件"""
        try:
            # 延迟导入以避免循环依赖
            from session_manager import session_manager
            
            session_data = await self.to_dict()
            return session_manager.save_session(session_data)
            
        except Exception as e:
            logger.error(f"保存会话失败: {e}")
            return False
    
    @classmethod
    async def load_from_file(cls, session_id: str) -> Optional["SharedContext"]:
        """从文件加载会话"""
        try:
            from session_manager import session_manager
            
            session_data = session_manager.load_session(session_id)
            if not session_data:
                return None
            
            # 创建新的上下文实例
            ctx = cls(session_id=session_id)
            
            # 恢复消息
            messages = session_data.get("messages", [])
            for msg in messages:
                # 转换时间戳回浮点数
                timestamp_str = msg.get("timestamp")
                if timestamp_str and isinstance(timestamp_str, str):
                    try:
                        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        timestamp = dt.timestamp()
                    except (ValueError, AttributeError):
                        timestamp = time.time()
                else:
                    timestamp = time.time()
                
                # 恢复消息格式
                restored_msg = {
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                    "timestamp": timestamp,
                    "message_id": msg.get("message_id", f"msg_{len(ctx._messages)}_{int(timestamp)}")
                }
                ctx._messages.append(restored_msg)
            
            # 恢复统计信息
            ctx._stats = session_data.get("stats", ctx._stats.copy())
            
            logger.info(f"会话已加载: {session_id}, 消息数: {len(ctx._messages)}")
            return ctx
            
        except Exception as e:
            logger.error(f"加载会话失败 {session_id}: {e}")
            return None
    
    async def auto_save(self) -> bool:
        """自动保存会话（在重要操作后调用）"""
        return await self.save_to_file()
    
    async def add_message_with_auto_save(self, role: str, content: str) -> None:
        """添加消息并自动保存"""
        await self.add_message(role, content)
        await self.auto_save()