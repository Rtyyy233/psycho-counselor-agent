import asyncio
import time
import json
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple, Callable, Awaitable
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
    
    def __init__(self, session_id: str = "default", token_limit: int = 128000, tokenizer = None):
        self.session_id = session_id
        self.token_limit = token_limit  # 令牌上限，默认128k对应DeepSeek V3
        self._tokenizer = tokenizer  # tokenizer实例，应有encode()方法
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
    
    # ========== 令牌管理 ==========
    
    def _calculate_token_count_no_lock(self) -> int:
        """计算令牌数（内部方法，不获取锁）"""
        if not self._tokenizer:
            return self._estimate_token_count()  # 回退到字符估算
        
        total = 0
        for msg in self._messages:
            content = msg.get("content", "")
            if content:
                # 假设tokenizer有encode方法
                total += len(self._tokenizer.encode(content))
        return total
    
    async def calculate_token_count(self) -> int:
        """计算所有消息的总令牌数（不包括注入内容）"""
        async with self._lock:
            return self._calculate_token_count_no_lock()
    
    def _estimate_token_count(self) -> int:
        """字符估算回退方法（3字符≈1token）"""
        total_chars = sum(len(msg.get("content", "")) for msg in self._messages)
        return total_chars // 3
    
    async def is_context_near_limit(self, threshold: float = 0.8) -> bool:
        """检查是否接近令牌上限（默认80%）"""
        token_count = await self.calculate_token_count()
        return token_count >= (self.token_limit * threshold)
    
    async def get_token_usage(self) -> Dict[str, Any]:
        """获取详细的令牌使用情况"""
        token_count = await self.calculate_token_count()
        remaining = self.token_limit - token_count
        usage_percentage = (token_count / self.token_limit) * 100 if self.token_limit > 0 else 0
        
        return {
            "current_tokens": token_count,
            "token_limit": self.token_limit,
            "remaining_tokens": remaining,
            "usage_percentage": usage_percentage,
            "is_near_limit_80": token_count >= (self.token_limit * 0.8),
            "is_near_limit_90": token_count >= (self.token_limit * 0.9),
            "is_over_limit": token_count > self.token_limit
        }
    
    @staticmethod
    def load_deepseek_tokenizer(tokenizer_path: str = "D:/浏览器下载/deepseek_v3_tokenizer"):
        """
        加载DeepSeek官方tokenizer
        
        Args:
            tokenizer_path: tokenizer目录路径
            
        Returns:
            tokenizer实例或None（加载失败时）
            
        Raises:
            ImportError: 未安装transformers时抛出
        """
        import os
        
        # 检查路径是否存在
        if not os.path.exists(tokenizer_path):
            logger.warning(f"Tokenizer路径不存在: {tokenizer_path}")
            return None
        
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, 
                trust_remote_code=True,
                model_max_length=128000  # 与DeepSeek V3一致
            )
            logger.info(f"DeepSeek tokenizer加载成功: {tokenizer_path}")
            return tokenizer
        except ImportError:
            logger.warning("未安装transformers，将使用字符估算模式")
            raise ImportError("需要安装transformers: pip install transformers")
        except Exception as e:
            logger.warning(f"加载tokenizer失败: {e}，将使用字符估算模式")
            return None
    
    @staticmethod
    def create_default_tokenizer():
        """创建默认tokenizer（尝试加载官方，失败则返回None）"""
        try:
            tokenizer = SharedContext.load_deepseek_tokenizer()
            if tokenizer:
                logger.info("默认tokenizer创建成功")
            else:
                logger.info("默认tokenizer创建失败，将使用字符估算模式")
            return tokenizer
        except Exception as e:
            logger.info(f"默认tokenizer创建失败: {e}，将使用字符估算模式")
            return None
    
    # ========== 消息清理 ==========
    
    def _remove_messages_by_indices_no_lock(self, indices: List[int]) -> List[Dict]:
        """
        按索引删除消息（内部方法，不获取锁）
        
        Args:
            indices: 要删除的消息索引列表（从0开始）
            
        Returns:
            被删除的消息列表
        """
        # 确保索引有效且排序（从大到小删除避免索引错位）
        indices = sorted(set(indices), reverse=True)
        removed = []
        
        for idx in indices:
            if 0 <= idx < len(self._messages):
                removed.append(self._messages.pop(idx))
        
        if removed:
            self._stats["message_count"] = len(self._messages)
            self._stats["last_updated"] = time.time()
        
        return removed
    
    async def remove_messages_by_indices(self, indices: List[int]) -> List[Dict]:
        """
        按索引删除消息
        
        Args:
            indices: 要删除的消息索引列表（从0开始）
            
        Returns:
            被删除的消息列表
        """
        async with self._lock:
            return self._remove_messages_by_indices_no_lock(indices)
    
    async def remove_oldest_messages(self, count: int) -> List[Dict]:
        """
        删除最旧的消息
        
        Args:
            count: 要删除的消息数量
            
        Returns:
            被删除的消息列表
        """
        if count <= 0:
            return []
        
        async with self._lock:
            indices = list(range(min(count, len(self._messages))))
            return self._remove_messages_by_indices_no_lock(indices)
    
    def _get_messages_for_summary_no_lock(self, start_idx: int = 0, end_idx: Optional[int] = None) -> str:
        """
        获取指定范围内的消息文本，用于生成摘要（内部方法，不获取锁）
        
        Args:
            start_idx: 起始索引
            end_idx: 结束索引（不包含），None表示到最后
            
        Returns:
            拼接后的对话文本
        """
        if not self._messages:
            return ""
        
        if end_idx is None:
            end_idx = len(self._messages)
        
        messages = self._messages[start_idx:end_idx]
        formatted = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")
        
        return "\n\n".join(formatted)
    
    async def get_messages_for_summary(self, start_idx: int = 0, end_idx: Optional[int] = None) -> str:
        """
        获取指定范围内的消息文本，用于生成摘要
        
        Args:
            start_idx: 起始索引
            end_idx: 结束索引（不包含），None表示到最后
            
        Returns:
            拼接后的对话文本
        """
        async with self._lock:
            return self._get_messages_for_summary_no_lock(start_idx, end_idx)
    
    def _get_oldest_messages_no_lock(self, target_tokens: int) -> Tuple[List[Dict], int]:
        """
        获取最旧的、令牌数约等于target_tokens的消息（内部方法，不获取锁）
        
        Args:
            target_tokens: 目标令牌数
            
        Returns:
            (消息列表, 实际令牌数)
        """
        if not self._messages:
            return [], 0
        
        # 字符估算回退：每3字符≈1token
        if not self._tokenizer:
            selected = []
            total_chars = 0
            target_chars = target_tokens * 3
            
            for msg in self._messages:
                content = msg.get("content", "")
                if not content:
                    continue
                    
                if total_chars + len(content) > target_chars:
                    break
                    
                selected.append(msg)
                total_chars += len(content)
            
            return selected, total_chars // 3
        
        # 使用tokenizer精确计算
        selected = []
        total_tokens = 0
        
        for msg in self._messages:
            content = msg.get("content", "")
            if not content:
                continue
                
            msg_tokens = len(self._tokenizer.encode(content))
            if total_tokens + msg_tokens > target_tokens:
                break
                
            selected.append(msg)
            total_tokens += msg_tokens
        
        return selected, total_tokens
    
    async def get_oldest_messages(self, target_tokens: int) -> Tuple[List[Dict], int]:
        """
        获取最旧的、令牌数约等于target_tokens的消息
        
        Args:
            target_tokens: 目标令牌数
            
        Returns:
            (消息列表, 实际令牌数)
        """
        async with self._lock:
            return self._get_oldest_messages_no_lock(target_tokens)
    
    async def cleanup_context(
        self, 
        target_usage: float = 0.7,
        storage_callback: Optional[Callable[[str, Dict], Awaitable[str]]] = None
    ) -> Dict[str, Any]:
        """
        清理旧消息并存储摘要
        
        Args:
            target_usage: 目标使用率（默认70%）
            storage_callback: 存储回调函数，接收(对话文本, 元数据)返回存储ID
            
        Returns:
            清理统计信息
        """
        async with self._lock:
            # 1. 计算需要清理的令牌数
            current_tokens = self._calculate_token_count_no_lock()
            target_tokens = int(self.token_limit * target_usage)
            
            if current_tokens <= target_tokens:
                return {"status": "no_need", "reason": "当前使用率已低于目标"}
            
            tokens_to_clean = current_tokens - target_tokens
            
            # 2. 获取要清理的消息
            messages_to_clean, actual_tokens = self._get_oldest_messages_no_lock(tokens_to_clean)
            
            if not messages_to_clean:
                return {"status": "no_messages", "reason": "没有可清理的消息"}
            
            # 3. 生成摘要文本
            summary_text = ""
            try:
                summary_text = self._get_messages_for_summary_no_lock(
                    start_idx=0, 
                    end_idx=len(messages_to_clean)
                )
            except Exception as e:
                logger.error(f"生成摘要文本失败: {e}")
                return {"status": "summary_failed", "error": str(e)}
            
            # 4. 存储摘要（如果有回调）
            storage_id = None
            if storage_callback and summary_text:
                try:
                    metadata = {
                        "session_id": self.session_id,
                        "cleaned_at": time.time(),
                        "message_count": len(messages_to_clean),
                        "token_count": actual_tokens,
                        "target_usage": target_usage
                    }
                    storage_id = await storage_callback(summary_text, metadata)
                    logger.info(f"对话摘要存储成功: {storage_id}, 消息数: {len(messages_to_clean)}")
                except Exception as e:
                    logger.error(f"存储摘要失败: {e}")
                    return {"status": "storage_failed", "error": str(e)}
            
            # 5. 删除消息
            indices = list(range(len(messages_to_clean)))
            removed = self._remove_messages_by_indices_no_lock(indices)
            
            # 6. 更新统计信息
            self._stats["message_count"] = len(self._messages)
            self._stats["last_updated"] = time.time()
            
            # 7. 返回统计信息
            remaining_tokens = current_tokens - actual_tokens
            return {
                "status": "success",
                "cleaned_messages": len(removed),
                "cleaned_tokens": actual_tokens,
                "remaining_tokens": remaining_tokens,
                "remaining_messages": len(self._messages),
                "storage_id": storage_id,
                "new_usage_percentage": (remaining_tokens / self.token_limit) * 100 if self.token_limit > 0 else 0,
                "target_usage_percentage": target_usage * 100
            }
    
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