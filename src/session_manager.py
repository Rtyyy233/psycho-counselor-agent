"""
会话管理模块 - 负责会话的持久化、加载和清理
"""
import json
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def find_project_root(start_path=Path(__file__).parent):
    """查找项目根目录（包含.env文件的目录）"""
    for parent in [start_path] + list(start_path.parents):
        if (parent / ".env").exists():
            return parent
    return start_path

PROJECT_ROOT = find_project_root()
SESSIONS_DIR = PROJECT_ROOT / "web" / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

class SessionManager:
    """会话管理器 - 处理会话的持久化存储和生命周期管理"""
    
    def __init__(self, sessions_dir: Path = SESSIONS_DIR):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        
    def get_session_path(self, session_id: str) -> Path:
        """获取会话文件路径"""
        return self.sessions_dir / f"{session_id}.json"
    
    def session_exists(self, session_id: str) -> bool:
        """检查会话是否存在"""
        return self.get_session_path(session_id).exists()
    
    def save_session(self, session_data: Dict[str, Any]) -> bool:
        """
        保存会话数据到文件
        
        Args:
            session_data: 会话数据，必须包含 'id' 字段
            
        Returns:
            bool: 保存是否成功
        """
        try:
            session_id = session_data.get("id")
            if not session_id:
                logger.error("会话数据缺少ID字段")
                return False
                
            filepath = self.get_session_path(session_id)
            
            # 确保数据包含必要的字段
            if "updated_at" not in session_data:
                session_data["updated_at"] = datetime.now().isoformat()
                
            # 如果不存在创建时间，则添加
            if "created_at" not in session_data:
                session_data["created_at"] = session_data["updated_at"]
            
            # 写入文件
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"会话已保存: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"保存会话失败: {e}")
            return False
    
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        从文件加载会话数据
        
        Args:
            session_id: 会话ID
            
        Returns:
            Optional[Dict]: 会话数据，如果不存在或加载失败则返回None
        """
        try:
            filepath = self.get_session_path(session_id)
            if not filepath.exists():
                logger.warning(f"会话文件不存在: {session_id}")
                return None
                
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            logger.debug(f"会话已加载: {session_id}")
            return data
            
        except Exception as e:
            logger.error(f"加载会话失败 {session_id}: {e}")
            return None
    
    def delete_session(self, session_id: str) -> bool:
        """删除会话文件"""
        try:
            filepath = self.get_session_path(session_id)
            if filepath.exists():
                filepath.unlink()
                logger.debug(f"会话已删除: {session_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"删除会话失败 {session_id}: {e}")
            return False
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """列出所有会话的基本信息"""
        sessions = []
        for filepath in self.sessions_dir.glob("*.json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                session_info = {
                    "id": data.get("id", filepath.stem),
                    "title": data.get("title", "未命名对话"),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                    "message_count": len(data.get("messages", [])),
                    "file_size": filepath.stat().st_size
                }
                sessions.append(session_info)
            except Exception as e:
                logger.warning(f"读取会话文件失败 {filepath}: {e}")
        
        # 按最后更新时间倒序排序
        sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return sessions
    
    def cleanup_sessions(self, 
                         max_age_days: int = 30,
                         max_sessions: Optional[int] = 100,
                         max_total_size_mb: int = 500) -> Dict[str, int]:
        """
        清理过期或过多的会话
        
        Args:
            max_age_days: 最大保留天数，超过此天数的会话将被删除
            max_sessions: 最大会话数量，超过此数量将删除最旧的会话（None表示无限制）
            max_total_size_mb: 最大总存储大小（MB），超过此大小将删除最旧的会话
            
        Returns:
            Dict[str, int]: 清理统计信息
        """
        stats = {
            "total_sessions": 0,
            "deleted_age": 0,
            "deleted_limit": 0,
            "deleted_size": 0,
            "remaining_sessions": 0
        }
        
        try:
            # 收集所有会话信息
            sessions_info = []
            for filepath in self.sessions_dir.glob("*.json"):
                try:
                    mtime = filepath.stat().st_mtime
                    size = filepath.stat().st_size
                    sessions_info.append({
                        "path": filepath,
                        "mtime": mtime,
                        "size": size,
                        "age_days": (time.time() - mtime) / (24 * 3600)
                    })
                except Exception as e:
                    logger.warning(f"获取会话文件信息失败 {filepath}: {e}")
            
            stats["total_sessions"] = len(sessions_info)
            
            if not sessions_info:
                return stats
            
            # 按修改时间排序（最旧的在前面）
            sessions_info.sort(key=lambda x: x["mtime"])
            
            # 清理策略1：基于时间
            deleted_age = 0
            for info in sessions_info[:]:  # 使用副本遍历
                if info["age_days"] > max_age_days:
                    try:
                        info["path"].unlink()
                        sessions_info.remove(info)
                        deleted_age += 1
                        logger.debug(f"删除过期会话: {info['path'].name} ({info['age_days']:.1f}天)")
                    except Exception as e:
                        logger.error(f"删除过期会话失败 {info['path']}: {e}")
            
            stats["deleted_age"] = deleted_age
            
            # 清理策略2：基于数量限制
            deleted_limit = 0
            if max_sessions is not None and len(sessions_info) > max_sessions:
                to_delete = len(sessions_info) - max_sessions
                for info in sessions_info[:to_delete]:
                    try:
                        info["path"].unlink()
                        deleted_limit += 1
                        logger.debug(f"删除超限会话: {info['path'].name}")
                    except Exception as e:
                        logger.error(f"删除超限会话失败 {info['path']}: {e}")
                # 更新剩余会话列表
                sessions_info = sessions_info[to_delete:]
            
            stats["deleted_limit"] = deleted_limit
            
            # 清理策略3：基于存储大小
            deleted_size = 0
            if max_total_size_mb > 0:
                total_size_mb = sum(info["size"] for info in sessions_info) / (1024 * 1024)
                max_size_bytes = max_total_size_mb * 1024 * 1024
                
                # 如果仍然超过大小限制，删除最旧的会话
                current_total = sum(info["size"] for info in sessions_info)
                sessions_info.sort(key=lambda x: x["mtime"])  # 确保按时间排序
                
                while current_total > max_size_bytes and sessions_info:
                    info = sessions_info[0]
                    try:
                        info["path"].unlink()
                        current_total -= info["size"]
                        sessions_info.pop(0)
                        deleted_size += 1
                        logger.debug(f"删除超大会话: {info['path'].name} ({info['size'] / 1024:.1f}KB)")
                    except Exception as e:
                        logger.error(f"删除超大会话失败 {info['path']}: {e}")
                        break
            
            stats["deleted_size"] = deleted_size
            stats["remaining_sessions"] = len(sessions_info)
            
            logger.info(f"会话清理完成: 删除{deleted_age + deleted_limit + deleted_size}个会话")
            
        except Exception as e:
            logger.error(f"会话清理失败: {e}")
        
        return stats
    
    def generate_session_id(self) -> str:
        """生成新的会话ID"""
        return str(uuid.uuid4())


# 全局会话管理器实例
session_manager = SessionManager()