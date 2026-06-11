"""
画像编排器

统筹 ProfileCollector 和 ProfileGenerator，提供 init（完整重建）和 update（增量更新）
两个入口，管理版本历史和文件持久化。
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import DATA_DIR
from SharedContext import SharedContext
from .profile_models import DOMAIN_REGISTRY, Profile
from .profile_collector import collect_all_data, collect_targeted_data
from .profile_generator import (
    generate_all_domains,
    generate_domain,
    screen_domains_for_update,
)

logger = logging.getLogger(__name__)

PROFILES_DIR = DATA_DIR / "profiles"


def _profile_path(user_id: str) -> Path:
    """获取画像文件路径"""
    return PROFILES_DIR / user_id / "profile.json"


def _version_dir(user_id: str) -> Path:
    """获取版本快照目录"""
    return PROFILES_DIR / user_id / "versions"


def load_profile(user_id: str) -> Optional[Profile]:
    """从磁盘加载用户画像"""
    path = _profile_path(user_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Profile(**data)
    except Exception as e:
        logger.warning("加载画像失败 (user=%s): %s", user_id, e)
        return None


def save_profile(profile: Profile) -> None:
    """Persist user profile to disk + version snapshot + Chroma fallback."""
    path = _profile_path(profile.user_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    profile_json = profile.model_dump_json(indent=2, ensure_ascii=False)

    # Write main file
    path.write_text(profile_json, encoding="utf-8")

    # Write version snapshot
    vdir = _version_dir(profile.user_id)
    vdir.mkdir(parents=True, exist_ok=True)
    vpath = vdir / f"v{profile.version}.json"
    vpath.write_text(profile_json, encoding="utf-8")

    # Sync to Chroma profile_store (fallback retrieval path)
    try:
        from mem_integration import profile_store
        from langchain_core.documents import Document
        doc = Document(
            page_content=profile_json[:4000],
            metadata={"user_id": profile.user_id, "version": profile.version},
            id=f"profile_{profile.user_id}",
        )
        profile_store.upsert(
            ids=[f"profile_{profile.user_id}"],
            documents=[doc.page_content],
            metadatas=[doc.metadata],
        )
        logger.debug("Profile synced to Chroma: user=%s", profile.user_id)
    except Exception as e:
        logger.debug("Profile Chroma sync skipped (non-critical): %s", e)

    logger.info(
        "Profile saved: user=%s, version=%d, domains=%d",
        profile.user_id,
        profile.version,
        len(profile.domains),
    )


async def init_profile(
    user_id: str,
    shared_context: SharedContext,
    source_session: str = "",
) -> Profile:
    """
    完整重建用户画像（全部 14 个领域）。

    流程：
    1. 从所有数据源全面采集数据
    2. 并行生成所有 14 个领域
    3. 保存画像（version=1）
    4. 返回完整画像

    Args:
        user_id: 用户 ID
        shared_context: 当前会话上下文
        source_session: 来源会话 ID（可选）

    Returns:
        Profile: 新建的用户画像
    """
    now = datetime.now().isoformat()

    logger.info("开始完整重建画像: user=%s", user_id)

    # 1. 采集数据
    collected = await collect_all_data(shared_context)

    if not collected.paip_summaries and not collected.diary_entries \
            and not collected.material_chunks and not collected.conversation_chunks:
        logger.warning("未采集到任何数据，将使用占位文本")
        collected.paip_summaries.append({
            "content": "（暂无来访者数据）",
            "metadata": {"text_type": "paip_summary", "section": "placeholder"},
            "id": "placeholder",
        })

    # 2. 生成所有领域
    domain_contents = await generate_all_domains(
        collected.raw_text,
        source_session=source_session,
        source_map=collected.source_map,
    )

    # 3. 构造 Profile
    profile = Profile(
        user_id=user_id,
        version=1,
        created_at=now,
        updated_at=now,
        version_history=[
            {
                "version": 1,
                "changed_domains": [d.domain_id for d in DOMAIN_REGISTRY],
                "timestamp": now,
                "source_session": source_session,
            }
        ],
        source_sessions=[source_session] if source_session else [],
        domains=domain_contents,
    )

    # 4. 保存
    save_profile(profile)
    logger.info("画像重建完成: user=%s, version=1", user_id)
    return profile


async def update_profile(
    user_id: str,
    new_paip: str,
    shared_context: SharedContext,
    source_session: str = "",
) -> Profile:
    """
    增量更新用户画像。

    流程：
    1. 加载现有画像（如无则完整重建）
    2. LLM 筛选：根据新 PAIP 判断哪些领域需要更新
    3. 针对性采集相关数据
    4. 仅更新被标记的领域
    5. 版本号 +1，更新 version_history
    6. 保存并返回

    Args:
        user_id: 用户 ID
        new_paip: 最新一次咨询的 PAIP 摘要
        shared_context: 当前会话上下文
        source_session: 来源会话 ID（可选）

    Returns:
        Profile: 更新后的用户画像
    """
    now = datetime.now().isoformat()

    # 1. 加载现有画像
    existing = load_profile(user_id)

    if existing is None:
        logger.info("未找到现有画像，执行完整重建: user=%s", user_id)
        return await init_profile(user_id, shared_context, source_session)

    # 2. LLM 筛选需要更新的领域
    logger.info("开始增量更新筛选: user=%s", user_id)
    screening = await screen_domains_for_update(new_paip, existing)

    if not screening.domains_to_update:
        logger.info("筛选结果：无需更新任何领域")
        return existing

    # 3. 针对性采集数据
    collected = await collect_targeted_data(shared_context, screening.domains_to_update)

    # 将新 PAIP 注入采集结果，确保生成时能用到最新信息
    collected.paip_summaries.insert(0, {
        "content": new_paip,
        "metadata": {
            "text_type": "paip_summary",
            "section": "new_update",
            "date": datetime.now().strftime("%y.%m.%d"),
        },
        "id": "paip_newest",
    })

    # 4. 逐个更新领域
    updated_domains: list[int] = []
    for did in screening.domains_to_update:
        existing_domain = existing.domains.get(did)
        profile_domain = await generate_domain(
            did, collected.raw_text, existing_domain, source_session, collected.source_map
        )
        existing.domains[did] = profile_domain
        updated_domains.append(did)

    # 5. 更新元数据
    existing.version += 1
    existing.updated_at = now

    if source_session and source_session not in existing.source_sessions:
        existing.source_sessions.append(source_session)

    existing.version_history.append({
        "version": existing.version,
        "changed_domains": updated_domains,
        "timestamp": now,
        "source_session": source_session,
    })

    # 6. 保存
    save_profile(existing)
    logger.info(
        "画像增量更新完成: user=%s, version=%d, updated_domains=%s",
        user_id,
        existing.version,
        updated_domains,
    )
    return existing


async def get_or_init_profile(
    user_id: str,
    shared_context: SharedContext,
) -> Profile:
    """
    获取用户画像，如不存在则初始化。

    便捷方法，用于在会话开始时调用。

    Args:
        user_id: 用户 ID
        shared_context: 当前会话上下文

    Returns:
        Profile: 用户画像
    """
    existing = load_profile(user_id)
    if existing is not None:
        logger.info("加载现有画像: user=%s, version=%d", user_id, existing.version)
        return existing
    return await init_profile(user_id, shared_context)
