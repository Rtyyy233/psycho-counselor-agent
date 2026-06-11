# src/mem_retrieve_user_profile.py
"""
用户画像检索模块

按 user_id 查找画像，优先从文件系统读取 JSON，回退到 Chroma 语义搜索。
"""

import asyncio
import json
import logging
from typing import Optional, List

from langchain_core.documents import Document
from pydantic import BaseModel, Field
from pathlib import Path

from config import DATA_DIR

logger = logging.getLogger(__name__)

PROFILE_DIR = DATA_DIR / "profiles"


class ProfileDomainInfo(BaseModel):
    domain_number: int
    domain_name: str
    summary: str
    details_text: str
    evidence_text: str = ""


class ProfileRetrievalResult(BaseModel):
    user_id: str
    domains: List[ProfileDomainInfo]
    full_text: str


def _load_profile_json(user_id: str) -> Optional[dict]:
    """从 profile.json 加载画像"""
    path = PROFILE_DIR / user_id / "profile.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("加载画像JSON失败 user=%s: %s", user_id, e)
        return None


def _profile_to_result(data: dict, domain_numbers: Optional[List[int]] = None) -> ProfileRetrievalResult:
    """将 profiler 输出的 JSON dict 转为检索结果"""
    user_id = data.get("user_id", "")
    domains_raw = data.get("domains", {})

    DOMAIN_NAMES = {
        1: "主诉与现状", 2: "成长与发展史", 3: "易感因素", 4: "诱发因素",
        5: "维持因素", 6: "保护因素", 7: "关系模式", 8: "干预反应",
        9: "风险评估", 10: "文化/背景因素", 11: "人格印象", 12: "情感世界",
        13: "沟通风格", 14: "求助与改变模式",
    }

    domain_infos = []
    full_parts = []
    for key, domain_data in domains_raw.items():
        try:
            dnum = int(key)
        except (ValueError, TypeError):
            continue
        if domain_numbers and dnum not in domain_numbers:
            continue

        name = DOMAIN_NAMES.get(dnum, f"领域{dnum}")
        summary = domain_data.get("summary", "")
        narrative = domain_data.get("narrative", "")
        facts = domain_data.get("facts", [])

        details_parts = [f"## {dnum}. {name}", narrative or ""]
        if facts:
            for f in facts:
                evidence_list = f.get('evidence', [])
                ev_str = ""
                if evidence_list:
                    ev_short = ", ".join([e.split(":", 1)[-1][:16] for e in evidence_list[:3]])
                    if len(evidence_list) > 3:
                        ev_short += f"...(+{len(evidence_list)-3})"
                    ev_str = f" 证据:[{ev_short}]"
                details_parts.append(
                    f"- [{f.get('type', '')}] {f.get('statement', '')}"
                    f" (置信度:{f.get('confidence', 0)}){ev_str}"
                )
        details_text = "\n".join(details_parts)

        domain_infos.append(ProfileDomainInfo(
            domain_number=dnum,
            domain_name=name,
            summary=summary,
            details_text=details_text,
        ))
        full_parts.append(details_text)

    domain_infos.sort(key=lambda d: d.domain_number)

    return ProfileRetrievalResult(
        user_id=user_id,
        domains=domain_infos,
        full_text="\n\n".join(full_parts),
    )


async def _resolve_evidence_content(raw_data: dict, domain_numbers: list[int]) -> dict[int, str]:
    """为指定领域解析 evidence 指向的原始内容。

    Args:
        raw_data: profile.json 的原始 JSON 数据
        domain_numbers: 需要解析证据的领域编号列表

    Returns:
        dict[int, str]: domain_number → 格式化的证据内容文本
    """
    from user_profile.profile_collector import lookup_evidence_content

    domains_raw = raw_data.get("domains", {})
    evidence_map: dict[int, str] = {}

    for key, domain_data in domains_raw.items():
        try:
            dnum = int(key)
        except (ValueError, TypeError):
            continue
        if dnum not in domain_numbers:
            continue

        facts = domain_data.get("facts", [])
        all_ev_ids = []
        for f in facts:
            all_ev_ids.extend(f.get("evidence", []))

        if not all_ev_ids:
            continue

        unique_ids = list(set(all_ev_ids))
        try:
            contents = await lookup_evidence_content(unique_ids)
            if contents:
                parts = []
                for eid, content in contents.items():
                    source_type = eid.split(":", 1)[0]
                    type_label = {"diary": "日记", "paip": "PAIP", "mat": "材料", "conv": "对话"}.get(source_type, source_type)
                    parts.append(f"  [{type_label}] {content}")
                evidence_map[dnum] = "\n".join(parts)
        except Exception as e:
            logger.warning("证据解析失败 domain=%d: %s", dnum, e)

    return evidence_map


async def retrieve_user_profile(user_id: str, domain_numbers: Optional[List[int]] = None) -> Optional[ProfileRetrievalResult]:
    """
    按 user_id 检索用户画像。

    Args:
        user_id: 用户标识
        domain_numbers: 需要返回的领域编号列表（1-14），不传则返回全部

    Returns:
        ProfileRetrievalResult 或 None
    """
    data = await asyncio.get_running_loop().run_in_executor(None, _load_profile_json, user_id)
    if data is None:
        result = await _retrieve_from_chroma(user_id, domain_numbers)
        return result

    result = _profile_to_result(data, domain_numbers)

    if result and domain_numbers:
        evidence_map = await _resolve_evidence_content(data, domain_numbers)
        for d in result.domains:
            ev_text = evidence_map.get(d.domain_number, "")
            if ev_text:
                d.evidence_text = ev_text
                d.details_text += f"\n\n【证据来源】\n{ev_text}"
        result.full_text = "\n\n".join([d.details_text for d in result.domains])

    return result


async def _retrieve_from_chroma(user_id: str, domain_numbers: Optional[List[int]] = None) -> Optional[ProfileRetrievalResult]:
    """从 Chroma 的 user_profile 集合回退检索"""
    try:
        from mem_integration import profile_store
    except (ImportError, AttributeError):
        logger.info("profile_store Chroma 集合不可用")
        return None

    def _sync():
        return profile_store.get(where={"user_id": user_id})

    raw = await asyncio.get_running_loop().run_in_executor(None, _sync)

    documents = raw.get("documents") or []
    if not documents:
        return None

    full_text = "\n\n".join(documents)
    return ProfileRetrievalResult(
        user_id=user_id,
        domains=[],
        full_text=full_text,
    )


async def retrieve_user_profile_summary(user_id: str) -> str:
    """
    获取画像文本摘要，适合注入 chatter 上下文。

    Args:
        user_id: 用户标识

    Returns:
        画像文本，不存在则返回空字符串
    """
    result = await retrieve_user_profile(user_id)
    if not result:
        return ""

    text = result.full_text
    if len(text) > 3000:
        text = text[:3000] + "\n...(画像已截断)"
    return text
