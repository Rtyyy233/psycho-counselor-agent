"""
ProfileCollector — 用户画像数据采集模块

直接访问 Chroma 集合，为画像初始化/更新采集原始数据。
使用四种采集策略：
  1. 全量扫描 conv_store 中的 PAIP 摘要
  2. 全量扫描 original_diary（非 annotation）
  3. 对 material_store 做语义搜索（child→parent）
  4. 对 conv_store 中的原始对话块做语义搜索

与 mem_retrieve_* 模块的区别：后者使用 LangGraph 状态机 + LLM 规划器，
适用于精确检索；本模块直接访问底层 Chroma，适用于批量数据采集。
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import langchain_core.documents

from mem_integration import (
    original_diary,
    conv_store,
    material_store,
    parent_store,
)
from SharedContext import SharedContext

logger = logging.getLogger(__name__)


# =====================================================================
# Data containers
# =====================================================================


@dataclass
class CollectedData:
    """采集到的原始数据包，供画像生成 LLM 使用"""

    paip_summaries: list[dict] = field(default_factory=list)
    """text_type == 'paip_summary' 的文档列表"""

    diary_entries: list[dict] = field(default_factory=list)
    """original_diary 中的原始日记条目"""

    material_chunks: list[dict] = field(default_factory=list)
    """匹配到的材料子块（含父级上下文）"""

    conversation_chunks: list[dict] = field(default_factory=list)
    """text_type == 'conversation' 的原始对话块"""

    recent_messages: list[dict] = field(default_factory=list)
    """当前会话最近消息"""

    source_map: dict[str, tuple[str, str]] = field(default_factory=dict)
    """ref_id → (source_type, chroma_id) 映射，用于 fact.evidence 溯源和内容反查"""

    _raw_text_override: str = field(default="", init=False, repr=False)
    _ref_counter: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        self._raw_text_override = ""
        self._ref_counter = 0

    def _next_ref(self, source_type: str) -> str:
        """生成下一个来源引用 ID，如 paip_005, diary_012"""
        self._ref_counter += 1
        return f"{source_type}_{self._ref_counter:04d}"

    def _tag_entry(self, entry: dict, source_type: str) -> str:
        """为条目生成 ref 标记并注册到 source_map。返回 '[ref:xxx]' 字符串"""
        chroma_id = entry.get("id", "")
        ref = self._next_ref(source_type)
        if chroma_id:
            self.source_map[ref] = (source_type, chroma_id)
        return f"[ref:{ref}]"

    @property
    def raw_text(self) -> str:
        """将所有采集内容合并为纯文本（供 LLM 使用）"""
        if self._raw_text_override:
            return self._raw_text_override
        return self._build_raw_text()

    @raw_text.setter
    def raw_text(self, value: str):
        """允许外部直接覆盖 raw_text（用于空数据时的占位文案）"""
        self._raw_text_override = value

    def _build_raw_text(self, max_items: int = 20) -> str:
        """合并文本，限制每个来源最多 max_items 条。每条数据带 [ref:xxx] 来源标记。"""
        sections = []

        if self.recent_messages:
            msg_text = "\n".join(
                f"{m.get('role', '?')}: {m.get('content', '')}"
                for m in self.recent_messages[-10:]
            )
            sections.append(f"【最近对话】\n{msg_text}")

        if self.paip_summaries:
            paip_lines = []
            for s in self.paip_summaries[:max_items]:
                ref_tag = self._tag_entry(s, "paip")
                meta = s.get("metadata") or {}
                date = meta.get("date", "")
                section = meta.get("section", "")
                tag = f"[{date}][{section}]" if date else f"[{section}]"
                paip_lines.append(f"{ref_tag} {tag} {s['content'][:300]}")
            sections.append("【对话PAIP摘要】\n" + "\n".join(paip_lines))

        if self.diary_entries:
            diary_lines = []
            for e in self.diary_entries[:max_items]:
                ref_tag = self._tag_entry(e, "diary")
                meta = e.get("metadata") or {}
                date = meta.get("date", "")
                tag = f"[{date}]" if date else "[未知日期]"
                diary_lines.append(f"{ref_tag} {tag} {e['content'][:500]}")
            sections.append("【日记】\n" + "\n\n".join(diary_lines))

        if self.material_chunks:
            mat_lines = []
            for m in self.material_chunks[:max_items]:
                ref_tag = self._tag_entry(m, "mat")
                child = m.get("child_content", "")[:200]
                parent = m.get("parent_content", "")
                if parent:
                    mat_lines.append(f"{ref_tag} 子块: {child}\n原文: {parent[:400]}")
                else:
                    mat_lines.append(f"{ref_tag} {child}")
            sections.append("【材料】\n" + "\n---\n".join(mat_lines))

        if self.conversation_chunks:
            conv_lines = []
            for c in self.conversation_chunks[:max_items]:
                ref_tag = self._tag_entry(c, "conv")
                meta = c.get("metadata") or {}
                date = meta.get("date", "")
                tag = f"[{date}]" if date else ""
                conv_lines.append(f"{ref_tag} {tag} {c['content'][:400]}")
            sections.append("【对话片段】\n" + "\n".join(conv_lines))

        return "\n\n".join(sections)


# =====================================================================
# Source type → Chroma collection mapping (for evidence lookup)
# =====================================================================

_SOURCE_COLLECTION_MAP = {
    "paip": conv_store,
    "diary": original_diary,
    "mat": material_store,
    "conv": conv_store,
}


async def lookup_evidence_content(evidence_ids: list[str]) -> dict[str, str]:
    """根据 evidence ID 列表查询原始数据内容。

    evidence ID 格式为 source_type:chroma_id（如 diary:abc123、paip:xyz789），
    该函数根据 source_type 定位到正确的 Chroma 集合，用 chroma_id 查出原文。

    Args:
        evidence_ids: evidence ID 列表，每项格式为 source_type:chroma_id

    Returns:
        dict[str, str]: evidence_id → 原始文本内容（截取前 500 字符）
    """
    results: dict[str, str] = {}

    for eid in evidence_ids:
        if ":" not in eid:
            continue
        source_type, chroma_id = eid.split(":", 1)
        collection = _SOURCE_COLLECTION_MAP.get(source_type)
        if collection is None:
            continue

        try:
            raw = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda cid=chroma_id: collection.get(ids=[cid]),
            )
            docs = raw.get("documents") or []
            if docs:
                results[eid] = docs[0][:500]
        except Exception as e:
            logger.warning("证据查询失败 (type=%s, id=%s): %s", source_type, chroma_id, e)

    return results


# =====================================================================
# Collector helpers
# =====================================================================


def _doc_to_dict(doc: langchain_core.documents.Document) -> dict:
    """将 Document 转为纯字典，方便序列化和传递"""
    return {
        "content": doc.page_content,
        "metadata": doc.metadata,
        "id": doc.id or "",
    }


def _chunk_diary_by_date(
    entries: list[dict],
    months: int = 3,
) -> list[list[dict]]:
    """将日记条目按日期分块，用于大集合的分批处理。

    Args:
        entries: 日记条目列表（已排序）
        months: 每块的月份跨度，默认 3（季度）

    Returns:
        分块后的条目列表，每块内部保持原有顺序
    """
    if not entries:
        return []

    buckets: dict[str, list[dict]] = defaultdict(list)

    for e in entries:
        date_str = (e.get("metadata") or {}).get("date", "")
        if date_str:
            try:
                dt = datetime.strptime(date_str, "%y.%m.%d")
                if months >= 3:
                    quarter = (dt.month - 1) // 3 + 1
                    key = f"{dt.year}-Q{quarter}"
                else:
                    key = f"{dt.year}-{dt.month:02d}"
            except (ValueError, OSError):
                key = "_no_date"
        else:
            key = "_no_date"

        buckets[key].append(e)

    result = []
    for key in sorted(buckets.keys()):
        if key == "_no_date":
            continue
        result.append(buckets[key])
    if "_no_date" in buckets:
        result.append(buckets["_no_date"])

    return result


# =====================================================================
# Data sources — four collection strategies
# =====================================================================


async def _collect_paip_summaries() -> list[dict]:
    """策略 1: 全量扫描 conv_store 中的 PAIP 摘要"""
    def _sync():
        return conv_store.get(where={"text_type": "paip_summary"})

    raw = await asyncio.get_running_loop().run_in_executor(None, _sync)

    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []
    ids = raw.get("ids") or []

    summaries = [
        {"content": pc, "metadata": m, "id": i}
        for pc, m, i in zip(documents, metadatas, ids)
    ]

    logger.info("ProfileCollector: PAIP summaries = %d", len(summaries))
    return summaries


async def _collect_diary_entries(batch_size: int = 200) -> list[dict]:
    """策略 2: 全量扫描 original_diary（区分于 diary_annotation）"""
    def _sync():
        return original_diary.get()

    raw = await asyncio.get_running_loop().run_in_executor(None, _sync)
    if raw is None:
        logger.info("ProfileCollector: diary entries = 0 (no data)")
        return []

    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []
    ids = raw.get("ids") or []

    entries = [
        {"content": pc, "metadata": m, "id": i}
        for pc, m, i in zip(documents, metadatas, ids)
    ]

    # 按日期排序
    def _date_key(e: dict) -> str:
        return (e.get("metadata") or {}).get("date", "")

    entries.sort(key=_date_key)

    logger.info("ProfileCollector: diary entries = %d", len(entries))
    return entries


async def _collect_materials(
    base_queries: Optional[list[str]] = None,
    max_results: int = 50,
) -> list[dict]:
    """策略 3: 对材料做语义搜索（child → parent）

    用一组覆盖性查询捕捉所有相关材料，对结果去重。
    """
    if base_queries is None:
        base_queries = [
            "情绪调节与情感表达",
            "认知模式与思维偏差",
            "人际关系与依恋模式",
            "心理防御与应对机制",
            "自我认知与价值观",
            "创伤经历与复原力",
            "行为模式与习惯",
            "心理评估与诊断知识",
        ]

    seen_ids: set[str] = set()
    results: list[dict] = []

    for query in base_queries:
        raw = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda q=query: material_store.similarity_search_with_score(q, k=10),
        )

        for doc, score in raw:
            doc_id = doc.id or str(id(doc))
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)

            # 获取父级上下文
            parent_content = ""
            parent_id = doc.metadata.get("parent_id", "")
            if parent_id:
                parent_raw = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda pid=parent_id: parent_store.get(ids=[pid]),
                )
                parents = parent_raw.get("documents") or []
                if parents:
                    parent_content = parents[0]

            results.append({
                "child_content": doc.page_content,
                "child_metadata": doc.metadata,
                "parent_content": parent_content,
                "score": float(score),
                "id": doc_id,
            })

        if len(results) >= max_results:
            break

    results = results[:max_results]
    logger.info("ProfileCollector: material chunks = %d", len(results))
    return results


_CONV_QUERIES = [
    "来访者自述与主诉",
    "情绪表达与情感反应",
    "咨询对话过程",
    "用户反馈与治疗反应",
    "心理评估与诊断对话",
    "治疗目标与进展讨论",
    "来访者人际关系描述",
    "自我认知与反思",
]


async def _collect_conversation_chunks(
    base_queries: Optional[list[str]] = None,
    max_results: int = 80,
) -> list[dict]:
    """策略 4: 对原始对话块做语义搜索（text_type == 'conversation'）"""
    if base_queries is None:
        base_queries = _CONV_QUERIES

    seen_ids: set[str] = set()
    results: list[dict] = []

    for query in base_queries:
        raw = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda q=query: conv_store.similarity_search_with_score(
                q, k=20, filter={"text_type": "conversation"}
            ),
        )

        for doc, score in raw:
            doc_id = doc.id or str(id(doc))
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
                "id": doc_id,
            })

        if len(results) >= max_results:
            break

    results = results[:max_results]
    logger.info("ProfileCollector: conversation chunks = %d", len(results))
    return results


# =====================================================================
# Public API
# =====================================================================


async def collect_all_data(
    shared_context: Optional[SharedContext] = None,
    *,
    diary_chunk_months: Optional[int] = None,
    diary_batch_size: int = 200,
    material_max: int = 50,
    conv_max: int = 80,
) -> CollectedData:
    """从所有数据源全面采集数据，用于完整画像重建。

    四种采集策略并行执行：
    1. 全量扫描 PAIP 摘要
    2. 全量扫描原始日记
    3. 语义搜索材料
    4. 语义搜索对话片段

    Args:
        shared_context: 当前会话上下文（可选，用于获取最近消息）
        diary_chunk_months: 按月分块（如 3=季度分块），None 不分块
        diary_batch_size: 日记每批大小
        material_max: 材料返回上限
        conv_max: 对话片段返回上限

    Returns:
        CollectedData 数据包
    """
    tasks = [
        asyncio.create_task(_collect_paip_summaries()),
        asyncio.create_task(_collect_diary_entries(batch_size=diary_batch_size)),
        asyncio.create_task(_collect_materials(max_results=material_max)),
        asyncio.create_task(_collect_conversation_chunks(max_results=conv_max)),
    ]

    paip, diary, materials, conversations = await asyncio.gather(*tasks)

    data = CollectedData(
        paip_summaries=paip,
        diary_entries=diary,
        material_chunks=materials,
        conversation_chunks=conversations,
    )

    # 可选：获取最近对话消息
    if shared_context is not None:
        try:
            msgs = await shared_context.get_all_messages()
            if msgs:
                data.recent_messages = msgs[-30:]
        except Exception as e:
            logger.warning("采集最近消息失败: %s", e)

    # 对大量日记做日期分块
    if diary_chunk_months is not None and len(diary) > diary_batch_size:
        chunks = _chunk_diary_by_date(diary, months=diary_chunk_months)
        logger.info(
            "ProfileCollector: diary chunked into %d parts (months=%d)",
            len(chunks),
            diary_chunk_months,
        )
        data._diary_chunks = chunks  # type: ignore[attr-defined]

    logger.info(
        "ProfileCollector: done — paip=%d diary=%d material=%d conv=%d msgs=%d",
        len(paip), len(diary), len(materials), len(conversations),
        len(data.recent_messages),
    )
    return data


async def collect_targeted_data(
    shared_context: SharedContext,
    domain_ids: list[int],
    *,
    material_max: int = 30,
    conv_max: int = 40,
) -> CollectedData:
    """针对特定领域采集数据，用于增量更新。

    根据 domain_ids 所属类别决定查询哪些数据源，减少 LLM 处理量。

    Args:
        shared_context: 当前会话上下文
        domain_ids: 需要更新的领域编号列表
        material_max: 材料返回上限
        conv_max: 对话片段返回上限

    Returns:
        CollectedData 数据包（可能部分数据源为空）
    """
    from .profile_models import get_domain_info, DomainCategory

    # 按类别分组
    categories: set[DomainCategory] = set()
    for did in domain_ids:
        info = get_domain_info(did)
        if info:
            categories.add(info.category)

    data = CollectedData()

    # 总是采集最近消息
    try:
        msgs = await shared_context.get_all_messages()
        if msgs:
            data.recent_messages = msgs[-20:]
    except Exception as e:
        logger.warning("采集最近消息失败: %s", e)

    DYNAMIC = DomainCategory.DYNAMIC
    BASELINE = DomainCategory.BASELINE
    EMOTIONAL = DomainCategory.EMOTIONAL_WORLD
    COMM = DomainCategory.COMMUNICATION

    # material: baseline + emotional
    if {BASELINE, EMOTIONAL} & categories:
        data.material_chunks = await _collect_materials(max_results=material_max)

    # paip + conversation: dynamic + emotional + communication
    if {DYNAMIC, EMOTIONAL, COMM} & categories:
        data.paip_summaries = await _collect_paip_summaries()
        data.conversation_chunks = await _collect_conversation_chunks(max_results=conv_max)

    # diary: dynamic
    if DYNAMIC in categories:
        data.diary_entries = await _collect_diary_entries()

    logger.info(
        "ProfileCollector: targeted done — domains=%s paip=%d diary=%d material=%d conv=%d",
        domain_ids,
        len(data.paip_summaries), len(data.diary_entries),
        len(data.material_chunks), len(data.conversation_chunks),
    )
    return data
