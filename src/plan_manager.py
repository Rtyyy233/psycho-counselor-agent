"""
PlanManager — 治疗计划持久化、会话后更新和阶段切换管理

负责：
- 计划的 JSON 文件持久化（与画像系统一致的文件优先模式）
- 会话结束后基于进展评估更新计划
- 阶段切换判断（规则 + LLM 混合）

注意：治疗的初始制定由 PlanGenerator Agent 负责（src/plan_generator.py）。
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain_deepseek import ChatDeepSeek

from config import LLM_MODEL, DATA_DIR
from treatment_plan import (
    TreatmentPlan,
    TreatmentGoal,
    SkillUsageRecord,
    InterventionResponse,
    PlanUpdateOutput,
    ProgressStatus,
    STAGE_ORDER,
    STAGE_DEFAULT_SKILLS,
)

logger = logging.getLogger(__name__)

# ===== 持久化路径 =====

PLANS_DIR = DATA_DIR / "treatment_plans"


def _plan_path(user_id: str) -> Path:
    return PLANS_DIR / user_id / "plan.json"


def _version_dir(user_id: str) -> Path:
    return PLANS_DIR / user_id / "versions"


# ===== 加载/保存 =====


def load_plan(user_id: str) -> Optional[TreatmentPlan]:
    """从磁盘加载治疗计划

    Args:
        user_id: 用户 ID

    Returns:
        TreatmentPlan 或 None（不存在时）
    """
    path = _plan_path(user_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return TreatmentPlan(**data)
    except Exception as e:
        logger.warning("加载治疗计划失败 (user=%s): %s", user_id, e)
        return None


def save_plan(plan: TreatmentPlan) -> None:
    """保存治疗计划到磁盘，并创建版本快照

    Args:
        plan: 治疗计划实例
    """
    path = _plan_path(plan.user_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 写入主文件
    path.write_text(
        plan.model_dump_json(indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 写入版本快照
    vdir = _version_dir(plan.user_id)
    vdir.mkdir(parents=True, exist_ok=True)
    vpath = vdir / f"v{plan.version}.json"
    vpath.write_text(
        plan.model_dump_json(indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info(
        "治疗计划已保存: user=%s, version=%d, stage=%s, goals=%d",
        plan.user_id, plan.version, plan.stage, len(plan.goals),
    )


# ===== 会话后更新 =====

UPDATE_SYSTEM_PROMPT = """你是一位资深临床督导，负责在每次心理咨询会话后评估治疗计划的进展。

你的任务是审查本次会话，更新治疗计划的目标进展、技能使用效果和阶段状态。

## 进展评估标准（Lambert & Hawkins 彩色编码）

- **green（绿色）**：来访者在本目标上有明确进展 — 症状减轻、功能改善或自我理解加深
- **yellow（黄色）**：进展不足，可能需要调整方法 — 来访者原地踏步或进步微弱
- **red（红色）**：严重不足或倒退 — 需要重新评估治疗计划
- **white（白色）**：目标已基本达成，可考虑结束该目标的工作

## 技能效果评分 (1-5)

- 5：来访者反应积极，有明确的"啊哈"时刻或行为改变
- 4：来访者参与良好，对话有进展
- 3：中性，无明显效果
- 2：来访者防御、回避或转移话题
- 1：来访者明显不适、情绪恶化或联盟破裂

## 阶段切换判断

只有当以下条件同时满足时才建议切换阶段：
1. 当前阶段的主要目标进展 >= 60%（绿色或白色）
2. 治疗联盟稳定（无未修复的破裂）
3. 无活跃的危机信号
4. 来访者显示出对下一阶段工作的准备度

## 注意事项

- 如果本次会话未涉及某些目标，保持其进展不变，不要在 goal_updates 中包含它们
- 干预效果判断需引用对话中的具体证据
- 不确定时倾向于保守 — 不急于切换阶段"""


async def update_plan(
    user_id: str,
    session_id: str,
    session_date: str,
    session_paip: str,
    recent_dialogue: str,
    plan: TreatmentPlan,
) -> TreatmentPlan:
    """会话结束后更新治疗计划。

    使用 LLM 评估本次会话的进展，更新目标状态、技能使用记录、
    干预反应记录，并判断是否需要阶段切换。

    Args:
        user_id: 用户 ID
        session_id: 本次会话 ID
        session_date: 会话日期
        session_paip: 本次会话的 PAIP 摘要
        recent_dialogue: 最近对话文本（用于上下文）
        plan: 当前治疗计划

    Returns:
        更新后的 TreatmentPlan
    """
    now = datetime.now().isoformat()

    # 构建现有目标摘要
    goals_summary = "\n".join([
        f"  {g.id}: {g.description} | 指标: {g.indicator} | "
        f"进度: {g.progress:.0%} | 状态: {g.progress_status}"
        for g in plan.goals
    ]) if plan.goals else "（暂无目标）"

    # 构建技能使用历史摘要
    skills_summary = "\n".join([
        f"  {r.skill_name}: 会话{r.session_id} 评分{r.effectiveness_rating}"
        for r in plan.skill_usage_history[-5:]
    ]) if plan.skill_usage_history else "（暂无记录）"

    user_prompt = (
        f"## 当前治疗计划\n"
        f"用户: {user_id}\n"
        f"阶段: {plan.stage} ({plan.get_stage_label()})\n"
        f"主要方法: {plan.primary_approach}\n"
        f"辅助方法: {', '.join(plan.secondary_approaches) if plan.secondary_approaches else '无'}\n"
        f"谨慎使用: {', '.join(plan.cautionary_approaches) if plan.cautionary_approaches else '无'}\n\n"
        f"## 当前目标\n{goals_summary}\n\n"
        f"## 近期技能使用\n{skills_summary}\n\n"
        f"## 上次会话的 Plan\n{plan.last_session_plan or '无'}\n\n"
        f"## 未完成事项\n{chr(10).join('- ' + item for item in plan.pending_items) if plan.pending_items else '无'}\n\n"
        f"## 本次会话 PAIP\n{session_paip}\n\n"
        f"## 最近对话\n{recent_dialogue[:3000]}\n\n"
        f"请评估本次会话进展，输出 PlanUpdateOutput 结构化结果。"
    )

    llm = ChatDeepSeek(model=LLM_MODEL, temperature=0.1)
    structured = llm.with_structured_output(PlanUpdateOutput)

    try:
        result = await structured.ainvoke([
            {"role": "system", "content": UPDATE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])
    except Exception as e:
        logger.error("会话后计划更新 LLM 调用失败: %s", e)
        # 返回原计划，不做更新
        return plan

    # 应用更新
    plan.version += 1
    plan.updated_at = now

    # 更新目标进展
    changed_fields = []
    for goal in plan.goals:
        if goal.id in result.goal_updates:
            new_progress = result.goal_updates[goal.id]
            if abs(new_progress - goal.progress) > 0.01:
                goal.progress = new_progress
                changed_fields.append(f"goal:{goal.id}")
        if goal.id in result.goal_status_changes:
            new_status = result.goal_status_changes[goal.id]
            if new_status != goal.progress_status:
                goal.progress_status = new_status
        goal.updated_at = now

    # 记录技能使用
    for skill_name in result.skills_used:
        effectiveness = result.skill_effectiveness.get(skill_name)
        plan.skill_usage_history.append(SkillUsageRecord(
            session_id=session_id,
            session_date=session_date,
            skill_name=skill_name,
            effectiveness_rating=effectiveness,
        ))
    if result.skills_used:
        changed_fields.append("skill_usage_history")

    # 记录干预反应
    for intervention in result.effective_interventions:
        plan.intervention_responses.append(InterventionResponse(
            skill_name=intervention,
            session_id=session_id,
            effect="effective",
            evidence=f"会话 {session_id} 中观察到积极反应",
            recorded_at=now,
        ))
    for intervention in result.ineffective_interventions:
        plan.intervention_responses.append(InterventionResponse(
            skill_name=intervention,
            session_id=session_id,
            effect="ineffective",
            evidence=f"会话 {session_id} 中观察到防御或无效反应",
            recorded_at=now,
        ))
    if result.effective_interventions or result.ineffective_interventions:
        changed_fields.append("intervention_responses")

    # 更新跨会话连续性
    plan.last_session_plan = result.session_summary

    # 更新 pending_items
    for item in result.completed_items:
        if item in plan.pending_items:
            plan.pending_items.remove(item)
    for item in result.new_pending_items:
        if item not in plan.pending_items:
            plan.pending_items.append(item)
    if result.completed_items or result.new_pending_items:
        changed_fields.append("pending_items")

    # 阶段切换
    if result.should_transition_stage and result.suggested_stage:
        old_stage = plan.stage
        plan.stage = result.suggested_stage
        changed_fields.append("stage")
        logger.info(
            "阶段切换: %s → %s (原因: %s)",
            old_stage, plan.stage, result.stage_transition_reason,
        )

    # 记录版本历史
    if session_id and session_id not in plan.source_sessions:
        plan.source_sessions.append(session_id)

    plan.version_history.append({
        "version": plan.version,
        "changed_fields": changed_fields,
        "timestamp": now,
        "source_session": session_id,
    })

    # 持久化
    save_plan(plan)

    logger.info(
        "治疗计划已更新: user=%s, version=%d, changed=%s",
        user_id, plan.version, changed_fields,
    )

    return plan


# ===== 阶段切换判断 =====


def should_transition_stage(plan: TreatmentPlan) -> tuple[bool, str, str]:
    """基于规则的阶段切换判断。

    规则优先，边界情况由 LLM 综合判断。

    Args:
        plan: 当前治疗计划

    Returns:
        (should_transition, suggested_stage, reason)
    """
    current_idx = plan.get_stage_index()

    # 规则 1：检查是否已到最后阶段
    if current_idx >= len(STAGE_ORDER) - 1:
        return False, "", "已处于最后阶段（整合/收尾期）"

    # 规则 2：检查当前阶段目标进展
    stage_goals = [g for g in plan.goals if g.priority <= 2]  # 只看高优先级目标
    if not stage_goals:
        return False, "", "当前阶段无高优先级目标"

    green_or_white = sum(1 for g in stage_goals
                         if g.progress_status in (ProgressStatus.GREEN, ProgressStatus.WHITE))
    ratio = green_or_white / len(stage_goals) if stage_goals else 0

    if ratio < 0.7:
        return False, "", f"当前阶段目标达成率 {ratio:.0%}，不足 70%"

    # 规则 3：检查是否有活跃的危机标记
    crisis_indicators = ["自伤", "自杀", "暴力", "危机", "急性创伤"]
    for goal in plan.goals:
        if any(indicator in goal.description for indicator in crisis_indicators):
            if goal.progress_status == ProgressStatus.RED:
                return False, "", "存在活跃的危机信号，不宜切换阶段"

    # 满足条件，建议切换到下一阶段
    next_stage = STAGE_ORDER[current_idx + 1]
    stage_label = plan.get_stage_label()

    return (
        True,
        next_stage,
        f"当前阶段（{stage_label}）高优先级目标达成率 {ratio:.0%}，"
        f"满足切换条件，建议进入下一阶段",
    )


# ===== 治疗连续性文本构建 =====


def build_continuity_context(plan: TreatmentPlan) -> str:
    """构建治疗连续性上下文文本，用于注入 chatter 对话。

    Args:
        plan: 治疗计划

    Returns:
        格式化的连续性上下文字符串
    """
    if plan.is_empty():
        return ""

    # 方法优先级：主 > 辅 > 谨慎，加视觉标记
    approaches_line = f"★ 主要方法：{plan.primary_approach}"
    if plan.secondary_approaches:
        approaches_line += f"  |  辅助：{', '.join(plan.secondary_approaches)}"
    if plan.cautionary_approaches:
        approaches_line += f"  |  ⚠ 谨慎：{', '.join(plan.cautionary_approaches)}"

    lines = [
        "【治疗连续性 — 跨会话上下文】",
        f"当前阶段：{plan.get_stage_label()}",
        approaches_line,
    ]

    if plan.last_session_plan:
        lines.append(f"上次会话计划：{plan.last_session_plan}")

    if plan.pending_items:
        lines.append("未完成事项：")
        for item in plan.pending_items[:5]:
            lines.append(f"  - {item}")

    # 目标进展摘要
    if plan.goals:
        goals_text = []
        for g in plan.goals:
            status_icon = {"green": "🟢", "yellow": "🟡", "red": "🔴", "white": "⚪"}.get(
                g.progress_status, "⚪"
            )
            goals_text.append(f"  {status_icon} {g.description} ({g.progress:.0%})")
        lines.append("目标进展：\n" + "\n".join(goals_text))

    if plan.supervisor_notes:
        lines.append(f"督导备注：{plan.supervisor_notes}")

    # 技能选择优先级引导
    skill_hints = _build_skill_priority_hints(plan)
    if skill_hints:
        lines.append(skill_hints)

    return "\n".join(lines)


def _build_skill_priority_hints(plan: "TreatmentPlan") -> str:
    """生成技能选择优先级提示，追加在 continuity context 末尾。

    基于治疗计划的主要/辅助/谨慎方法标记，提示 chatter 优先选择计划一致的方法。
    """
    from skill_loader import get_chatter_skill_catalog_for_plan
    catalog = get_chatter_skill_catalog_for_plan(
        primary_approaches=plan.get_active_approaches(),
        secondary_approaches=plan.secondary_approaches,
        cautionary_approaches=plan.cautionary_approaches,
        stage=plan.stage,
    )
    return f"\n---\n【技能选择优先级 — 基于治疗计划】\n{catalog}"
