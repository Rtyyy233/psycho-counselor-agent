"""
Skill Loader — Progressive disclosure for therapy and supervision skills.

Uses skillkit to manage SKILL.md files. Agents receive only metadata (name + description)
in their system prompt; full instructions are loaded on-demand via lookup_skill().

Architecture:
  L1 (metadata, ~100 tokens per skill) — always in system prompt via skill catalog
  L2 (full instructions, loaded on activation) — returned by lookup_skill(name)
  L3 (resources, on-demand) — referenced within full instructions
"""

from skillkit import SkillManager
from langchain_core.tools import tool
import os
import logging

logger = logging.getLogger(__name__)

_SKILLS_ROOT = os.path.join(os.path.dirname(__file__), "skills")

_chatter_manager = SkillManager(project_skill_dir=os.path.join(_SKILLS_ROOT, "chatter"))
_supervisor_manager = SkillManager(project_skill_dir=os.path.join(_SKILLS_ROOT, "supervisor"))

_chatter_manager.discover()
_supervisor_manager.discover()

logger.info(
    "SkillLoader initialized: %d chatter skills, %d supervisor skills",
    len(_chatter_manager.list_skills()),
    len(_supervisor_manager.list_skills()),
)

_skill_cache: dict[str, str] = {}
_lookup_counts: dict[str, int] = {}
_total_lookups: int = 0
_MAX_TOTAL_LOOKUPS = 3
_MAX_AVAIL_CALLS = 1

_avail_call_count = 0


def _build_catalog(manager: SkillManager, exclude_indices: bool = True) -> str:
    """Build a compact skill catalog string for system prompt injection."""
    lines = []
    for skill in manager.list_skills():
        if exclude_indices and skill.name.endswith("-index"):
            continue
        lines.append(f"- `{skill.name}`: {skill.description}")
    return "\n".join(lines)


def get_chatter_skill_catalog() -> str:
    """Return chatter's skill catalog (L1 metadata) for system prompt."""
    return _build_catalog(_chatter_manager)


def get_supervisor_skill_catalog() -> str:
    """Return supervisor's skill catalog (L1 metadata) for system prompt."""
    return _build_catalog(_supervisor_manager)


def _build_plan_aware_catalog(
    manager: SkillManager,
    primary_approaches: list[str],
    secondary_approaches: list[str],
    cautionary_approaches: list[str],
    stage: str = "engagement",
) -> str:
    """Build a treatment-plan-aware skill catalog with priority markers.

    Sorts skills into three groups:
      1. ★ Primary — plan's primary + secondary approaches (try first)
      2. Normal — all other skills
      3. ⚠ Cautionary — plan's cautionary approaches (use only with clear rationale)

    Crisis-intervention is always preserved at its Tier 3 position, not buried.
    Person-centered is always marked ★ as the relational foundation.

    Adds “over-activation risk” markers for skills prone to text-intensity bias.
    """
    from treatment_plan import STAGE_LABELS

    # Normalize approach names for comparison
    primary_set = set(primary_approaches)
    secondary_set = set(secondary_approaches)
    cautionary_set = set(cautionary_approaches)

    # Always treat person-centered as primary (relational foundation)
    primary_set.add("person-centered")

    # Skills prone to text-intensity over-activation (LLM reads strong words,
    # lacks access to tone/face/body-language signals that would moderate judgment)
    OVER_ACTIVATION_RISK = {
        "crisis-intervention": (
            "【⚠ 激活风险：高】纯文字易将情绪词汇误判为危机信号。"
            "必须同时满足 (a)具体伤害意图/计划 (b)既往行为史或明确手段描述 才激活。仅凭'崩溃''活着好累'等用语不足以触发。"
        ),
        "gestalt": (
            "【⚠ 激活风险：中】来访者频繁使用情绪词 ≠ 需要体验性深化。"
            "必须同时满足 (a)联盟稳固 (b)来访者能区分体验情绪vs被情绪淹没 (c)当前阶段允许 才激活。"
        ),
        "psychodynamic": (
            "【⚠ 激活风险：中】单次移情表现 ≠ 稳定的关系模式。"
            "必须是跨越多次对话的反复模式才激活，避免对单次表现过度解读。"
        ),
        "existential": (
            "【⚠ 激活风险：低-中】来访者表达无意义感可能是短暂心境而非深层存在危机。"
            "先区分：是情绪状态还是稳定的存在议题。建立期不主动引入存在性探索。"
        ),
    }

    stage_label = STAGE_LABELS.get(stage, stage)

    primary_lines = []
    normal_lines = []
    cautionary_lines = []

    for skill in manager.list_skills():
        if skill.name.endswith("-index"):
            continue

        skill_name = skill.name.lower().replace(" ", "")
        line = f"- `{skill.name}`: {skill.description}"

        if skill_name in cautionary_set and skill_name not in primary_set:
            cautionary_lines.append(f"⚠ {line} 【当前阶段（{stage_label}）谨慎使用 — 仅在明确适应症时调用】")
        elif skill_name in primary_set:
            primary_lines.append(f"★ {line} 【当前主要方法，优先使用】")
        elif skill_name in secondary_set:
            primary_lines.append(f"★ {line} 【辅助方法】")
        else:
            normal_lines.append(f"  {line}")

        # Append over-activation risk warning for high-risk skills
        if skill_name in OVER_ACTIVATION_RISK:
            risk_note = OVER_ACTIVATION_RISK[skill_name]
            if skill_name in cautionary_set and skill_name not in primary_set:
                cautionary_lines[-1] += f"\n  {risk_note}"
            elif skill_name in primary_set or skill_name in secondary_set:
                primary_lines[-1] += f"\n  {risk_note}"
            else:
                normal_lines[-1] += f"\n  {risk_note}"

    # Assemble: primary → normal → cautionary
    sections = []
    if primary_lines:
        sections.append("【主要/推荐方法 — 优先选择】\n" + "\n".join(primary_lines))
    if normal_lines:
        sections.append("【可用方法】\n" + "\n".join(normal_lines))
    if cautionary_lines:
        sections.append("【谨慎使用 — 仅在明确适应症时调用】\n" + "\n".join(cautionary_lines))

    return "\n\n".join(sections)


def get_chatter_skill_catalog_for_plan(
    primary_approaches: list[str] | None = None,
    secondary_approaches: list[str] | None = None,
    cautionary_approaches: list[str] | None = None,
    stage: str = "engagement",
) -> str:
    """Return chatter's skill catalog, reordered by treatment plan priorities.

    Args:
        primary_approaches: plan's primary approach (e.g. ["cbt"])
        secondary_approaches: plan's secondary (e.g. ["person-centered", "behavioral-third-wave"])
        cautionary_approaches: approaches to use cautiously (e.g. ["psychodynamic", "gestalt"])
        stage: current treatment stage

    Returns:
        Reordered skill catalog string for system prompt injection
    """
    return _build_plan_aware_catalog(
        _chatter_manager,
        primary_approaches or [],
        secondary_approaches or [],
        cautionary_approaches or [],
        stage,
    )


def get_supervisor_skill_catalog_for_plan(
    primary_approaches: list[str] | None = None,
    secondary_approaches: list[str] | None = None,
    cautionary_approaches: list[str] | None = None,
    stage: str = "engagement",
) -> str:
    """Return supervisor's skill catalog, reordered by treatment plan priorities.

    Args:
        primary_approaches: plan's primary approach
        secondary_approaches: plan's secondary
        cautionary_approaches: approaches to use cautiously
        stage: current treatment stage

    Returns:
        Reordered skill catalog string for system prompt injection
    """
    return _build_plan_aware_catalog(
        _supervisor_manager,
        primary_approaches or [],
        secondary_approaches or [],
        cautionary_approaches or [],
        stage,
    )


def _load_skill_content(name: str) -> str | None:
    """Load full skill content from either manager, with caching."""
    if name in _skill_cache:
        return _skill_cache[name]
    for manager in (_chatter_manager, _supervisor_manager):
        try:
            skill = manager.load_skill(name)
            if skill and skill.content:
                _skill_cache[name] = skill.content
                return skill.content
        except Exception:
            continue
    return None


def reset_skill_lookup_counts():
    """重置技能查询计数 — 每次 agent 调用前应调用此函数。"""
    _lookup_counts.clear()
    global _total_lookups, _avail_call_count
    _total_lookups = 0
    _avail_call_count = 0


def _get_available_skill_names() -> list[str]:
    """Get all available skill names (excluding indices)."""
    names = []
    for manager in (_chatter_manager, _supervisor_manager):
        for skill in manager.list_skills():
            if not skill.name.endswith("-index"):
                names.append(skill.name)
    return names


@tool
async def lookup_skill(name: str) -> str:
    """加载指定治疗技能或督导技能的完整操作指南。

    当来访者状态触发某个技能的适用条件时（如来访者使用绝对化语言→cbt，
    表达无意义感→existential，表达自伤意念→crisis-intervention等），
    调用此工具获取该技能的详细操作指南。

    参数 name: 技能名称，如 'person-centered', 'cbt', 'crisis-intervention',
              'alliance-monitoring', 'countertransference' 等。
    使用 get_available_skills 查看所有可用技能及其触发条件。

    重要限制：每轮最多调用此工具3次（不论技能名称）。优先选择最匹配当前状态的1-2个技能。
    返回：该技能的完整操作指南（包含核心概念、适用场景、具体做法、注意事项）。
    """
    global _total_lookups
    _total_lookups += 1

    if _total_lookups > _MAX_TOTAL_LOOKUPS:
        return (
            f"已达到技能查询上限（{_MAX_TOTAL_LOOKUPS}次）。"
            f"已加载的技能: {', '.join(_lookup_counts.keys())}。"
            f"请基于以上已加载的技能立即完成分析并输出结构化结果，不要再调用任何工具。"
        )

    _lookup_counts[name] = _lookup_counts.get(name, 0) + 1
    count = _lookup_counts[name]

    if count > 1:
        return (
            f"【重复查询警告 — 第{count}次加载 '{name}'】"
            f"该技能已加载过，请使用已有信息。剩余查询次数: {_MAX_TOTAL_LOOKUPS - _total_lookups}。\n\n"
            f"{_load_skill_content(name)}"
        )

    content = _load_skill_content(name)
    if content is None:
        available = ", ".join(_get_available_skill_names())
        return f"未找到技能 '{name}'。可用技能: {available}"

    return (
        f"已加载技能 '{name}'（剩余查询次数: {_MAX_TOTAL_LOOKUPS - _total_lookups}）。\n\n"
        f"{content}"
    )




@tool
async def get_available_skills(dummy: str = "") -> str:
    """获取所有可用治疗技能和督导技能的列表，包含触发条件说明。

    当你需要：
    1. 了解有哪些技能可用（通常在首次判断时调用一次）
    2. 根据来访者状态查找适合的技能
    3. 不确定哪个技能最适合当前情况时

    调用此工具获取完整的技能目录（含触发条件）。

    重要限制：本工具每轮只能调用1次。重复调用将返回错误。

    参数 dummy: 占位参数，可以留空。
    """
    global _avail_call_count
    _avail_call_count += 1

    if _avail_call_count > _MAX_AVAIL_CALLS:
        return (
            f"已达到技能目录查询上限（{_MAX_AVAIL_CALLS}次）。技能目录没有变化。"
            f"请直接根据当前对话信号调用 lookup_skill 加载所需的1-2个技能，然后立即完成分析。"
        )

    lines = []
    lines.append("=== 治疗技能（Chatter）===\n")
    for skill in _chatter_manager.list_skills():
        if skill.name.endswith("-index"):
            continue
        lines.append(f"- `{skill.name}`: {skill.description}")

    lines.append("\n=== 督导技能（Supervisor）===\n")
    for skill in _supervisor_manager.list_skills():
        if skill.name.endswith("-index"):
            continue
        lines.append(f"- `{skill.name}`: {skill.description}")

    return "\n".join(lines)
