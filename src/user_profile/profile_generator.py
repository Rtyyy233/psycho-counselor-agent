"""
画像生成模块

使用 LLM 按 3 轮策略生成用户画像的 14 个领域，并提供增量更新的领域筛选与单领域更新功能。

3 轮初始化策略：
  Round 1: PAIP 摘要 + 日记 → 领域 1-5（主诉与现状/成长与发展史/易感因素/诱发因素/维持因素）
  Round 2: R1 输出 + 材料 → 领域 6-10（保护因素/关系模式/干预反应/风险评估/文化/背景因素）
  Round 3: R1+R2 输出 + 对话片段 → 领域 11-14（人格印象/情感世界/沟通风格/求助与改变模式）
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any

from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field, model_validator

from config import LLM_MODEL
from .profile_models import (
    DOMAIN_REGISTRY,
    DomainInfo,
    DomainCategory,
    Fact,
    FactType,
    Profile,
    ProfileDomain,
    ScreeningOutput,
)
from .profile_collector import CollectedData

logger = logging.getLogger(__name__)


# =====================================================================
# LLM Output Schemas
# =====================================================================


class GeneratedFact(BaseModel):
    """单条事实（LLM 输出用，不含 ID）"""
    type: FactType = Field(
        description="事实类型: observation/self_report/inference/pattern/risk/treatment_response"
    )
    statement: str = Field(description="事实陈述")
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="置信度 0.0-1.0。多次出现的模式或明确陈述为高(0.8+)，单次提及为中等(0.5-0.7)，推测为低(<0.5)",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="支持该事实的数据来源引用。填入文本中的 [ref:xxx] 标记，如 ['diary_0001', 'paip_0003']。只填确实支持该事实的引用",
    )
    relates_to: list[str] = Field(
        default_factory=list,
        description="关联的其他事实编号（如 fact_001_001）。基于内容判断哪些fact可能相关，便于后续构建事实网络",
    )

    @model_validator(mode='before')
    @classmethod
    def fix_invalid_type(cls, data: Any) -> Any:
        if isinstance(data, dict):
            valid = {"observation", "self_report", "inference", "pattern", "risk", "treatment_response"}
            typ = data.get("type")
            if isinstance(typ, str) and typ not in valid:
                fallback = {
                    "assessment": "observation",
                    "analysis": "inference",
                    "evaluation": "observation",
                    "note": "observation",
                    "history": "self_report",
                    "prediction": "inference",
                }.get(typ, "observation")
                logger.warning("Fact type '%s' not in enum, falling back to '%s'", typ, fallback)
                data = dict(data)
                data["type"] = fallback
        return data


class GeneratedDomain(BaseModel):
    """单个领域的 LLM 输出内容"""
    summary: str = Field(description="一句话概括该领域")
    narrative: str = Field(description="详细叙事分析，覆盖所有子字段")
    facts: list[GeneratedFact] = Field(description="事实列表")

    @model_validator(mode='before')
    @classmethod
    def parse_string_facts(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get('facts'), str):
            try:
                data['facts'] = json.loads(data['facts'])
            except (json.JSONDecodeError, TypeError):
                pass
        return data


class DomainOutput(BaseModel):
    """单领域增量更新输出（batch update API 用）"""
    domain_number: int = Field(description="领域编号 1-14")
    narrative: str = Field(description="更新后的叙事")
    facts: list[dict] = Field(description="更新后的事实字典列表")


# ---- 每轮输出模型：4-5 个领域一组，避免输出截断 ----

class Round1Output(BaseModel):
    """第1轮输出：领域 1-5"""
    domain_1: GeneratedDomain = Field(
        description="领域1: 主诉与现状 — TA目前最大的困扰是什么？症状表现、频率/强度、功能影响、治疗史"
    )
    domain_2: GeneratedDomain = Field(
        description="领域2: 成长与发展史 — 什么经历塑造了今天的TA？生命故事脉络，客观事实为主"
    )
    domain_3: GeneratedDomain = Field(
        description="领域3: 易感因素 — 什么让TA对当前问题更脆弱？因果推断，区别于#2的'发生了什么'"
    )
    domain_4: GeneratedDomain = Field(
        description="领域4: 诱发因素 — 为什么是现在来求助？触发事件或情境"
    )
    domain_5: GeneratedDomain = Field(
        description="领域5: 维持因素 — 什么在让问题持续？维持问题的循环因素"
    )

    @model_validator(mode='before')
    @classmethod
    def parse_string_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        for field_name in ('domain_1', 'domain_2', 'domain_3', 'domain_4', 'domain_5'):
            val = data.get(field_name)
            if isinstance(val, str):
                try:
                    data[field_name] = json.loads(val)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning("Round1: failed to parse %s as JSON (%s), raw[:200]=%s",
                                   field_name, e, val[:200])
        return data


class Round2Output(BaseModel):
    """第2轮输出：领域 6-10"""
    domain_6: GeneratedDomain = Field(
        description="领域6: 保护因素 — 什么在支撑着TA？资源、优势、支持系统"
    )
    domain_7: GeneratedDomain = Field(
        description="领域7: 关系模式 — TA如何与重要他人互动？依恋模式、人际循环"
    )
    domain_8: GeneratedDomain = Field(
        description="领域8: 干预反应 — 什么对TA有效/无效？effective_interventions, ineffective_or_harmful, alliance_quality"
    )
    domain_9: GeneratedDomain = Field(
        description="领域9: 风险评估 — 需警惕的风险：self_harm, suicide, violence, dropout, crisis_history"
    )
    domain_10: GeneratedDomain = Field(
        description="领域10: 文化/背景因素 — 文化、家庭、社会如何影响TA？"
    )

    @model_validator(mode='before')
    @classmethod
    def parse_string_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        for field_name in ('domain_6', 'domain_7', 'domain_8', 'domain_9', 'domain_10'):
            val = data.get(field_name)
            if isinstance(val, str):
                try:
                    data[field_name] = json.loads(val)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning("Round2: failed to parse %s as JSON (%s), raw[:200]=%s",
                                   field_name, e, val[:200])
        return data


class Round3Output(BaseModel):
    """第3轮输出：领域 11-14"""
    domain_11: GeneratedDomain = Field(
        description="领域11: 人格印象 — TA是个什么样的人？性格底色、价值观、自我认同、骄傲与羞耻（侧重稳定特质）"
    )
    domain_12: GeneratedDomain = Field(
        description="领域12: 情感世界 — TA的深层渴望与恐惧是什么？情感表达方式、意义感来源（侧重动态情感）"
    )
    domain_13: GeneratedDomain = Field(
        description="领域13: 沟通风格 — TA习惯如何表达和接收？language_style, metaphor_preference, pacing_preference, response_receptivity"
    )
    domain_14: GeneratedDomain = Field(
        description="领域14: 求助与改变模式 — TA如何面对改变？change_stage, help_seeking_style, resistance_pattern, counseling_expectation"
    )

    @model_validator(mode='before')
    @classmethod
    def parse_string_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        for field_name in ('domain_11', 'domain_12', 'domain_13', 'domain_14'):
            val = data.get(field_name)
            if isinstance(val, str):
                try:
                    data[field_name] = json.loads(val)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning("Round3: failed to parse %s as JSON (%s), raw[:200]=%s",
                                   field_name, e, val[:200])
        return data


# ---- 内部输出模型 ----

class _SingleDomainOutput(BaseModel):
    """单个领域的 LLM 输出（增量更新用）"""
    domain: GeneratedDomain = Field(description="该领域的内容")


class _UpdateOutput(BaseModel):
    """简化输出模型（batch update 用）"""
    narrative: str = Field(description="更新后的叙事")
    facts: list[GeneratedFact] = Field(description="事实列表")


# =====================================================================
# System Prompts
# =====================================================================

SYSTEM_PROMPT_BASE = """你是一位专业的心理咨询用户画像分析师。你的任务是根据采集到的来访者数据，系统性地生成用户画像。

用户画像包含 14 个领域，分为四大类别：
- baseline（基线信息）：相对稳定的个人特质和背景
- dynamic（动态信息）：随咨询进程变化的状态
- emotional_world（情感世界）：深层情感动力
- communication（沟通风格）：表达与接收方式

事实类型说明：
- observation: 咨询师直接观察到的事实
- self_report: TA 自我报告的内容
- inference: 基于证据的推断
- pattern: 识别出的重复模式
- risk: 风险评估发现
- treatment_response: 干预效果观察

置信度指南：多次出现的模式或明确陈述为高(0.8+)，单次提及为中等(0.5-0.7)，推测为低(<0.5)。

数据来源引用：采集数据中每条记录前有 [ref:xxx] 标记（如 [ref:paip_0001]、[ref:diary_0003]），表示该条数据的来源ID。
生成事实时，请在 evidence 字段中列出支持该事实的数据引用（填写 ref: 后面的值，如 paip_0001）。只填写确实支持该事实的引用，不要全部复制。"""

ROUND_1_SYSTEM_PROMPT = SYSTEM_PROMPT_BASE + """

【第1轮：生成领域 1-5】

请根据采集到的来访者数据，生成领域 1-5 的画像内容。

你需要注意：
- 仔细阅读所有数据，识别出与每个领域相关的信息
- 对每个领域提炼一句话摘要和详细的叙事分析
- 标记具体事实条目，每条需指定类型和置信度
- 如果数据不足以支撑某个领域的完整分析，请诚实说明，不要编造

需要输出的 5 个领域：
1. 主诉与现状 — 核心困扰、症状表现、功能影响、治疗历史
2. 成长与发展史 — 生命故事脉络（客观事实），时间线和重要事件
3. 易感因素 — 让TA更脆弱的因素（因果推断层面）
4. 诱发因素 — 为什么是现在来求助
5. 维持因素 — 什么让问题持续存在"""

ROUND_2_SYSTEM_PROMPT = SYSTEM_PROMPT_BASE + """

【第2轮：生成领域 6-10】

请根据第1轮已生成的领域 1-5 和新采集的数据，生成领域 6-10 的画像内容。

注意：
- 利用第1轮的结果作为背景参考，避免矛盾
- 结合材料数据和PAIP信息，进行综合分析
- 风险评估(领域9)要特别谨慎——有明确证据才标记风险
- 每个领域需要一句话摘要、详细叙事和事实条目

需要输出的 5 个领域：
6. 保护因素 — 什么在TA困难时提供支持
7. 关系模式 — TA与重要他人的互动方式
8. 干预反应 — 有效和无效的干预方式、咨访关系质量
9. 风险评估 — 自伤/自杀/暴力/脱落风险及危机历史
10. 文化/背景因素 — 文化、家庭、社会环境如何塑造和影响TA"""

ROUND_3_SYSTEM_PROMPT = SYSTEM_PROMPT_BASE + """

【第3轮：生成领域 11-14】

请根据已生成的领域 1-10 和新采集的数据，生成领域 11-14 的画像内容。

注意：
- 利用前两轮结果作为背景参考，保持一致性
- 关注非问题陈述性内容：表达方式、情感反应、互动模式、语言习惯
- 领域11(人格印象)侧重稳定的性格特质，领域12(情感世界)侧重动态情感动力
- 每个领域需要一句话摘要、详细叙事和事实条目

需要输出的 4 个领域：
11. 人格印象 — 性格底色、价值观、自我认同、骄傲与羞耻
12. 情感世界 — 深层渴望与恐惧、情感表达、意义感来源
13. 沟通风格 — 语言风格、用词习惯、隐喻偏好、对话节奏、接受度
14. 求助与改变模式 — 改变阶段、求助风格、阻抗模式、对咨询的期待"""

FULL_GENERATION_SYSTEM_PROMPT = SYSTEM_PROMPT_BASE  # 旧名保留，指向基础提示

UPDATE_SYSTEM_PROMPT = """你是一位专业的心理咨询用户画像分析师。你的任务是根据新增的数据，更新用户画像中的特定领域。

请结合现有领域内容和新增数据，对领域内容进行增量更新：
1. 保留仍然有效的原有信息
2. 根据新数据补充或修正内容
3. 更新事实列表（新增事实、调整置信度）

注意不要丢失已有的重要信息，只做必要的补充和修正。"""

SCREENING_SYSTEM_PROMPT = """你是一位专业的心理咨询用户画像更新分析师。

你的任务是根据最新一次咨询的 PAIP（Problem/Assessment/Intervention/Plan）摘要，判断用户画像中的哪些领域需要更新。

规则：
- 只有当新信息确实对某个领域的理解有实质性改变时，才标记为需要更新
- 不要过度标记——只选择那些确实被新 PAIP 内容触及的领域
- 如果 PAIP 主要重复已知信息，则可以标记为"无需更新"
- 风险评估相关的内容（自伤、自杀、暴力等）即使轻微提及也应标记"""


# =====================================================================
# Internal helpers
# =====================================================================

_ROUND_MODELS: Dict[int, type[BaseModel]] = {
    1: Round1Output,
    2: Round2Output,
    3: Round3Output,
}

_ROUND_DOMAIN_RANGES: Dict[int, range] = {
    1: range(1, 6),
    2: range(6, 11),
    3: range(11, 15),
}


def _get_domain_name(did: int) -> str:
    """按编号获取领域名称"""
    info = next((d for d in DOMAIN_REGISTRY if d.domain_id == did), None)
    return info.name if info else f"领域{did}"


async def _call_round(
    llm: ChatDeepSeek,
    round_num: int,
    system_prompt: str,
    user_prompt: str,
    source_session: str = "",
    source_map: dict[str, tuple[str, str]] | None = None,
) -> Dict[int, ProfileDomain]:
    """调用 LLM 执行一轮生成，返回 domain_id -> ProfileDomain 的映射。

    Args:
        llm: LLM 实例
        round_num: 轮次 (1/2/3)
        system_prompt: 本轮的系统提示
        user_prompt: 本轮的输入数据 + 指令
        source_map: ref_id → (source_type, chroma_id) 映射

    Returns:
        Dict[int, ProfileDomain]: 该轮生成的领域内容（可能部分失败）
    """
    output_model = _ROUND_MODELS[round_num]
    structured = llm.with_structured_output(output_model)

    try:
        result = await structured.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
    except Exception as e:
        logger.warning("Round %d LLM call failed: %s", round_num, e)
        return {}

    now = datetime.now().isoformat()
    domains: Dict[int, ProfileDomain] = {}

    for did in _ROUND_DOMAIN_RANGES[round_num]:
        attr = f"domain_{did}"
        gd = getattr(result, attr, None)
        if gd is None:
            logger.warning("Round %d: domain_%d missing in LLM output", round_num, did)
            continue

        facts = []
        for i, gf in enumerate(gd.facts):
            fact_id = f"fact_{did:03d}_{i+1:03d}"
            # 解析 ref → source_type:chroma_id
            evidence_ids = []
            if source_map:
                for ref in gf.evidence:
                    entry = source_map.get(ref)
                    if entry:
                        source_type, chroma_id = entry
                        evidence_ids.append(f"{source_type}:{chroma_id}")
                    else:
                        evidence_ids.append(ref)
            facts.append(Fact(
                id=fact_id,
                type=gf.type,
                statement=gf.statement,
                evidence=evidence_ids,
                confidence=gf.confidence,
                relates_to=gf.relates_to,
            ))

        domains[did] = ProfileDomain(
            summary=gd.summary,
            narrative=gd.narrative,
            facts=facts,
            last_updated=now,
        )

    logger.info(
        "Round %d: generated %d/%d domains",
        round_num,
        len(domains),
        len(_ROUND_DOMAIN_RANGES[round_num]),
    )
    return domains


def _build_round_summary(domains: Dict[int, ProfileDomain], start: int, end: int) -> str:
    """为后续轮次构建已生成领域的摘要参考"""
    lines = []
    for did in range(start, end + 1):
        d = domains.get(did)
        if d is None:
            continue
        name = _get_domain_name(did)
        fact_lines = "\n".join(
            f"  - [{f.type.value}] {f.statement[:120]}" for f in d.facts[:5]
        )
        if len(d.facts) > 5:
            fact_lines += f"\n  ... 还有 {len(d.facts) - 5} 条事实"
        lines.append(
            f"## {did}. {name}\n"
            f"摘要: {d.summary}\n"
            f"叙事: {d.narrative[:300]}...\n"
            f"事实:\n{fact_lines}"
        )
    return "\n\n".join(lines)


# =====================================================================
# Input builders — 从 CollectedData 中提取各轮所需的数据
# =====================================================================


def _build_r1_input(materials: CollectedData) -> str:
    """第1轮输入：PAIP 摘要 + 日记 + 最近对话"""
    sections = []

    if materials.paip_summaries:
        lines = []
        for s in materials.paip_summaries[:20]:
            meta = s.get("metadata", {})
            date = meta.get("date", "")
            section = meta.get("section", "")
            tag = f"[{date}][{section}]" if date else f"[{section}]"
            content = s.get("content", "")[:500]
            lines.append(f"{tag}\n{content}")
        sections.append("【PAIP 摘要】\n" + "\n\n".join(lines))

    if materials.diary_entries:
        lines = []
        for e in materials.diary_entries[:15]:
            meta = e.get("metadata", {})
            date = meta.get("date", "")
            tag = f"[{date}]" if date else "[未知日期]"
            content = e.get("content", "")[:600]
            lines.append(f"{tag}\n{content}")
        sections.append("【日记】\n" + "\n\n".join(lines))

    if materials.recent_messages:
        lines = []
        for m in materials.recent_messages[-10:]:
            role = m.get("role", "?")
            content = m.get("content", "")
            if isinstance(content, str):
                lines.append(f"{role}: {content[:300]}")
        sections.append("【最近对话】\n" + "\n".join(lines))

    return "\n\n=====\n\n".join(sections) if sections else "（暂无数据）"


def _build_r2_input(materials: CollectedData) -> str:
    """第2轮输入：材料结果 + PAIP 摘要"""
    sections = []

    if materials.material_chunks:
        lines = []
        for m in materials.material_chunks[:15]:
            child = m.get("child_content", "")[:300]
            parent = m.get("parent_content", "")
            if parent:
                lines.append(f"子块: {child}\n原文: {parent[:500]}")
            else:
                lines.append(f"子块: {child}")
        sections.append("【材料】\n" + "\n---\n".join(lines))

    if materials.paip_summaries:
        lines = []
        for s in materials.paip_summaries[:15]:
            meta = s.get("metadata", {})
            date = meta.get("date", "")
            tag = f"[{date}]" if date else ""
            lines.append(f"{tag} {s.get('content', '')[:500]}")
        sections.append("【PAIP 摘要】\n" + "\n\n".join(lines))

    if materials.recent_messages:
        lines = []
        for m in materials.recent_messages[-5:]:
            role = m.get("role", "?")
            content = m.get("content", "")
            if isinstance(content, str):
                lines.append(f"{role}: {content[:300]}")
        sections.append("【最近对话】\n" + "\n".join(lines))

    return "\n\n=====\n\n".join(sections) if sections else "（暂无数据）"


def _build_r3_input(materials: CollectedData) -> str:
    """第3轮输入：原始对话片段 + PAIP"""
    sections = []

    if materials.conversation_chunks:
        lines = []
        for c in materials.conversation_chunks[:20]:
            meta = c.get("metadata", {})
            date = meta.get("date", "")
            tag = f"[{date}]" if date else ""
            content = c.get("content", "")[:500]
            lines.append(f"{tag} {content}")
        sections.append("【对话片段】\n" + "\n".join(lines))

    if materials.paip_summaries:
        lines = []
        for s in materials.paip_summaries[:5]:
            meta = s.get("metadata", {})
            date = meta.get("date", "")
            tag = f"[{date}]" if date else ""
            lines.append(f"{tag} {s.get('content', '')[:500]}")
        sections.append("【PAIP 摘要】\n" + "\n".join(lines))

    return "\n\n=====\n\n".join(sections) if sections else "（暂无数据）"


# =====================================================================
# Public API: 3-Round Initialization (preferred for init)
# =====================================================================


async def generate_init_profile(
    user_id: str,
    materials: CollectedData,
    llm: Optional[ChatDeepSeek] = None,
) -> Dict[int, ProfileDomain]:
    """3 轮 LLM 生成初始用户画像（14 个领域）。

    第1轮: PAIP 摘要 + 日记 → 领域 1-5
    第2轮: 第1轮输出 + 材料 + PAIP → 领域 6-10
    第3轮: 第1/2轮输出 + 对话片段 → 领域 11-14

    Args:
        user_id: 用户 ID（用于日志）
        materials: 采集到的原始数据包（含 PAIP、日记、材料、对话等）
        llm: LLM 实例（None 时使用默认配置: ChatDeepSeek(model=LLM_MODEL, temperature=0.1)）

    Returns:
        Dict[int, ProfileDomain]: domain_id (1-14) -> ProfileDomain 的映射
    """
    if llm is None:
        llm = ChatDeepSeek(model=LLM_MODEL, temperature=0.1)

    results: Dict[int, ProfileDomain] = {}

    # ---- Round 1: 领域 1-5 ----
    logger.info("ProfileGenerator: Round 1 (domains 1-5) for user=%s", user_id)
    r1_input = _build_r1_input(materials)
    r1_prompt = (
        f"【采集数据】\n{r1_input}\n\n"
        f"请根据以上数据，生成领域 1-5 的画像内容。"
    )
    r1_result = await _call_round(llm, 1, ROUND_1_SYSTEM_PROMPT, r1_prompt, "", materials.source_map)
    results.update(r1_result)

    # ---- Round 2: 领域 6-10 ----
    logger.info("ProfileGenerator: Round 2 (domains 6-10) for user=%s", user_id)
    r2_input = _build_r2_input(materials)
    r1_summary = _build_round_summary(results, 1, 5)
    r2_prompt = (
        f"【第1轮已生成的领域 1-5】\n{r1_summary}\n\n"
        f"【新采集数据】\n{r2_input}\n\n"
        f"请基于以上背景和新数据，生成领域 6-10 的画像内容。"
    )
    r2_result = await _call_round(llm, 2, ROUND_2_SYSTEM_PROMPT, r2_prompt, "", materials.source_map)
    results.update(r2_result)

    # ---- Round 3: 领域 11-14 ----
    logger.info("ProfileGenerator: Round 3 (domains 11-14) for user=%s", user_id)
    r3_input = _build_r3_input(materials)
    r12_summary = _build_round_summary(results, 1, 10)
    r3_prompt = (
        f"【已生成的领域 1-10】\n{r12_summary}\n\n"
        f"【新采集数据】\n{r3_input}\n\n"
        f"请基于以上背景和新数据，生成领域 11-14 的画像内容。"
    )
    r3_result = await _call_round(llm, 3, ROUND_3_SYSTEM_PROMPT, r3_prompt, "", materials.source_map)
    results.update(r3_result)

    logger.info(
        "ProfileGenerator: init complete for user=%s — %d/14 domains",
        user_id, len(results),
    )
    return results


# =====================================================================
# Public API: Text-based all-domains generation (backward compatible)
# =====================================================================


async def generate_all_domains(
    collected_text: str,
    existing_profile: Optional[Profile] = None,
    source_session: str = "",
    source_map: dict[str, tuple[str, str]] | None = None,
) -> Dict[int, ProfileDomain]:
    """基于纯文本数据生成所有 14 个领域（向后兼容接口）。

    内部使用 3 轮策略（每轮将同一文本传给 LLM，轮次间传递已生成的领域摘要）。
    建议新代码优先使用 generate_init_profile(CollectedData) 以获得更好的结果。

    Args:
        collected_text: 采集到的原始数据（纯文本，含 [ref:xxx] 标记）
        existing_profile: 已有 Profile（暂用于未来扩展，目前 unused）
        source_session: 来源会话 ID（保留，当前未用于 evidence）
        source_map: ref_id → (source_type, chroma_id) 映射

    Returns:
        Dict[int, ProfileDomain]: domain_id -> ProfileDomain 的映射
    """
    llm = ChatDeepSeek(model=LLM_MODEL, temperature=0.1)
    results: Dict[int, ProfileDomain] = {}

    # Round 1: 领域 1-5
    r1_prompt = (
        f"【采集数据】\n{collected_text[:8000]}\n\n"
        f"请根据以上数据，生成领域 1-5 的画像内容。每条数据前的 [ref:xxx] 标记是该数据的来源引用。"
    )
    r1_result = await _call_round(llm, 1, ROUND_1_SYSTEM_PROMPT, r1_prompt, source_session, source_map)
    results.update(r1_result)

    # Round 2: 领域 6-10
    r1_summary = _build_round_summary(results, 1, 5)
    r2_prompt = (
        f"【第1轮已生成的领域 1-5】\n{r1_summary}\n\n"
        f"【采集数据】\n{collected_text[:8000]}\n\n"
        f"请基于以上背景和数据，生成领域 6-10 的画像内容。每条数据前的 [ref:xxx] 标记是该数据的来源引用。"
    )
    r2_result = await _call_round(llm, 2, ROUND_2_SYSTEM_PROMPT, r2_prompt, source_session, source_map)
    results.update(r2_result)

    # Round 3: 领域 11-14
    r12_summary = _build_round_summary(results, 1, 10)
    r3_prompt = (
        f"【已生成的领域 1-10】\n{r12_summary}\n\n"
        f"【采集数据】\n{collected_text[:8000]}\n\n"
        f"请基于以上背景和数据，生成领域 11-14 的画像内容。每条数据前的 [ref:xxx] 标记是该数据的来源引用。"
    )
    r3_result = await _call_round(llm, 3, ROUND_3_SYSTEM_PROMPT, r3_prompt, source_session, source_map)
    results.update(r3_result)

    logger.info("generate_all_domains: %d/14 domains generated", len(results))
    return results


# =====================================================================
# Public API: Single domain generation/update
# =====================================================================


async def generate_domain(
    domain_id: int,
    collected_text: str,
    existing_domain: Optional[ProfileDomain] = None,
    source_session: str = "",
    source_map: dict[str, tuple[str, str]] | None = None,
) -> ProfileDomain:
    """生成或更新单个领域。

    用于增量更新中逐个领域更新。

    Args:
        domain_id: 领域编号 (1-14)
        collected_text: 相关数据文本（含 [ref:xxx] 标记）
        existing_domain: 该领域现有内容（增量更新时传入）
        source_session: 来源会话 ID（保留，当前未用于 evidence）
        source_map: ref_id → (source_type, chroma_id) 映射

    Returns:
        ProfileDomain: 更新后的领域内容
    """
    domain_info = next(
        (d for d in DOMAIN_REGISTRY if d.domain_id == domain_id), None
    )
    if not domain_info:
        logger.warning("无效的领域编号: %d", domain_id)
        return ProfileDomain()

    now = datetime.now().isoformat()

    existing_text = ""
    if existing_domain and existing_domain.summary:
        existing_text = (
            f"现有摘要: {existing_domain.summary}\n"
            f"现有叙事: {existing_domain.narrative}\n"
            f"现有事实数: {len(existing_domain.facts)}"
        )

    system_prompt = UPDATE_SYSTEM_PROMPT if existing_text else FULL_GENERATION_SYSTEM_PROMPT

    if existing_text:
        user_prompt = (
            f"请为领域「{domain_info.name}」更新画像内容。\n\n"
            f"领域说明: {domain_info.description}\n"
            f"类别: {domain_info.category.value}\n\n"
            f"新数据:\n{collected_text[:6000]}\n\n"
            f"现有内容（请保留有效信息，补充新内容）:\n{existing_text}"
        )
    else:
        user_prompt = (
            f"请为领域「{domain_info.name}」生成画像内容。\n\n"
            f"领域说明: {domain_info.description}\n"
            f"类别: {domain_info.category.value}\n\n"
            f"数据:\n{collected_text[:6000]}"
        )

    llm = ChatDeepSeek(model=LLM_MODEL, temperature=0.1)
    structured = llm.with_structured_output(_SingleDomainOutput)

    try:
        result = await structured.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
    except Exception as e:
        logger.warning("领域 %d (%s) 生成失败: %s", domain_id, domain_info.name, e)
        return existing_domain or ProfileDomain(last_updated=now)

    gd = result.domain
    facts = []
    for i, gf in enumerate(gd.facts):
        fact_id = f"fact_{domain_id:03d}_{i+1:03d}"
        evidence_ids = []
        if source_map:
            for ref in gf.evidence:
                entry = source_map.get(ref)
                if entry:
                    source_type, chroma_id = entry
                    evidence_ids.append(f"{source_type}:{chroma_id}")
                else:
                    evidence_ids.append(ref)
        facts.append(Fact(
            id=fact_id,
            type=gf.type,
            statement=gf.statement,
            evidence=evidence_ids,
            confidence=gf.confidence,
            relates_to=gf.relates_to,
        ))

    return ProfileDomain(
        summary=gd.summary,
        narrative=gd.narrative,
        facts=facts,
        last_updated=now,
    )


# =====================================================================
# Public API: Batch update of specific domains
# =====================================================================


async def update_domains(
    user_id: str,
    domains_to_update: List[int],
    existing_domain_contents: Dict[int, str],
    new_material: str,
    llm: Optional[ChatDeepSeek] = None,
) -> Dict[int, DomainOutput]:
    """批量更新指定的领域，每个领域的 LLM 调用独立并行。

    Args:
        user_id: 用户 ID（用于日志）
        domains_to_update: 需要更新的领域编号列表
        existing_domain_contents: {domain_number: 该领域现有叙事文本}
        new_material: 用于更新的新数据
        llm: LLM 实例（None 时使用默认配置）

    Returns:
        Dict[int, DomainOutput]: domain_number -> 更新后的内容
    """
    if llm is None:
        llm = ChatDeepSeek(model=LLM_MODEL, temperature=0.1)

    async def _update_one(did: int) -> tuple[int, DomainOutput]:
        existing = existing_domain_contents.get(did, "")
        domain_name = _get_domain_name(did)

        if existing:
            prompt = (
                f"请为领域「{domain_name}」更新画像内容。\n\n"
                f"现有内容:\n{existing[:2000]}\n\n"
                f"新数据:\n{new_material[:4000]}\n\n"
                f"请输出更新后的叙事和事实列表。"
            )
        else:
            prompt = (
                f"请为领域「{domain_name}」生成画像内容。\n\n"
                f"数据:\n{new_material[:4000]}"
            )

        structured = llm.with_structured_output(_UpdateOutput)

        try:
            result = await structured.ainvoke([
                {"role": "system", "content": UPDATE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ])
            return did, DomainOutput(
                domain_number=did,
                narrative=result.narrative,
                facts=[f.model_dump() for f in result.facts],
            )
        except Exception as e:
            logger.warning("update_domains: 领域 %d (%s) 失败: %s", did, domain_name, e)
            return did, DomainOutput(
                domain_number=did,
                narrative=existing,
                facts=[],
            )

    tasks = [_update_one(did) for did in domains_to_update]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    output: Dict[int, DomainOutput] = {}
    for item in results:
        if isinstance(item, Exception):
            logger.warning("update_domains: 任务异常: %s", item)
            continue
        did, domain_output = item
        output[did] = domain_output

    logger.info(
        "update_domains: user=%s, requested=%d, success=%d",
        user_id, len(domains_to_update), len(output),
    )
    return output


# =====================================================================
# Public API: Screening
# =====================================================================


async def screen_domains_for_update(
    new_paip: str,
    existing_profile: Profile,
) -> ScreeningOutput:
    """根据新 PAIP 判断哪些领域需要更新。

    Args:
        new_paip: 最新一次咨询的 PAIP 摘要文本
        existing_profile: 当前用户画像

    Returns:
        ScreeningOutput: 包含需要更新的领域编号列表和理由
    """
    if existing_profile.is_empty():
        return ScreeningOutput(
            domains_to_update=[d.domain_id for d in DOMAIN_REGISTRY],
            reason="画像为空，需要完整初始化",
        )

    domain_list_str = "\n".join([
        f"{d.domain_id}. {d.name}（{d.description}）[{d.category.value}]"
        for d in DOMAIN_REGISTRY
    ])

    profile_summary_str = "\n".join([
        f"领域 {did}: {pd.summary[:80] if pd.summary else '(空)'}"
        for did, pd in existing_profile.domains.items()
    ]) if existing_profile.domains else "(无现有内容)"

    user_prompt = (
        f"最新 PAIP 摘要：\n{new_paip}\n\n"
        f"可选领域：\n{domain_list_str}\n\n"
        f"现有画像摘要：\n{profile_summary_str}\n\n"
        f"请判断哪些领域需要根据新 PAIP 内容进行更新。"
    )

    llm = ChatDeepSeek(model=LLM_MODEL, temperature=0.1)
    structured = llm.with_structured_output(ScreeningOutput)

    try:
        result = await structured.ainvoke([
            {"role": "system", "content": SCREENING_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])
    except Exception as e:
        logger.warning("更新筛选失败，回退到默认策略: %s", e)
        return ScreeningOutput(
            domains_to_update=[
                d.domain_id for d in DOMAIN_REGISTRY
                if d.category == DomainCategory.DYNAMIC
            ],
            reason=f"LLM 筛选失败，回退到动态领域默认策略: {e}",
        )

    # 验证输出中的领域编号
    valid_ids = {d.domain_id for d in DOMAIN_REGISTRY}
    result.domains_to_update = [
        did for did in result.domains_to_update if did in valid_ids
    ]

    if not result.domains_to_update:
        result.domains_to_update = [1, 4, 5, 6, 8, 9, 14]

    logger.info(
        "更新筛选结果: domains=%s, reason=%s",
        result.domains_to_update,
        result.reason[:100],
    )
    return result
