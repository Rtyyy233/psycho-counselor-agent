"""
TreatmentPlan 数据模型

定义治疗计划、治疗目标、技能使用记录和干预反应记录的 Pydantic 模型。
基于 Talen & Schindler 目标导向督导计划模型和 Lambert & Hawkins 进展评估框架。
"""

from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field


# ===== 技能名称常量（与 skills/ 目录一致） =====

CHATTER_SKILL_NAMES = [
    "person-centered", "existential",
    "psychodynamic", "adlerian", "gestalt", "cbt", "behavioral-third-wave",
    "choice-reality", "sfbt", "narrative", "feminist", "family-systems",
    "alliance-repair", "clinical-interviewing", "crisis-intervention",
]

SUPERVISOR_SKILL_NAMES = [
    "alliance-monitoring", "process-quality", "countertransference",
    "pattern-recognition", "crisis-detection",
]

ALL_SKILL_NAMES = CHATTER_SKILL_NAMES + SUPERVISOR_SKILL_NAMES

# 治疗阶段
TreatmentStage = Literal["engagement", "cognitive_behavioral", "emotional_deepening", "consolidation"]

STAGE_ORDER: list[TreatmentStage] = [
    "engagement", "cognitive_behavioral", "emotional_deepening", "consolidation"
]

STAGE_LABELS: dict[str, str] = {
    "engagement": "建立期",
    "cognitive_behavioral": "工作期-认知行为",
    "emotional_deepening": "工作期-情感深化",
    "consolidation": "整合/收尾期",
}

# 每个阶段默认推荐的方法
STAGE_DEFAULT_SKILLS: dict[str, dict[str, list[str]]] = {
    "engagement": {
        "primary": ["clinical-interviewing", "person-centered"],
        "secondary": ["existential"],
        "cautionary": ["gestalt", "psychodynamic", "crisis-intervention"],
    },
    "cognitive_behavioral": {
        "primary": ["cbt", "person-centered"],
        "secondary": ["behavioral-third-wave", "sfbt", "choice-reality"],
        "cautionary": ["gestalt", "psychodynamic", "existential"],
    },
    "emotional_deepening": {
        "primary": ["gestalt", "psychodynamic"],
        "secondary": ["person-centered", "existential", "narrative", "feminist"],
        "cautionary": [],
    },
    "consolidation": {
        "primary": ["narrative", "person-centered"],
        "secondary": ["sfbt", "cbt", "existential", "adlerian"],
        "cautionary": ["psychodynamic", "gestalt"],
    },
}


# ===== 进展状态（Lambert & Hawkins 彩色编码） =====

class ProgressStatus:
    WHITE = "white"       # 功能正常，可考虑结束
    GREEN = "green"       # 进展充分，无需调整计划
    YELLOW = "yellow"     # 进展不足，需调整计划
    RED = "red"           # 严重不足，需重新评估

    @classmethod
    def from_progress(cls, progress: float) -> str:
        """基于进展百分比 (0.0-1.0) 判断状态"""
        if progress >= 0.85:
            return cls.WHITE
        elif progress >= 0.60:
            return cls.GREEN
        elif progress >= 0.30:
            return cls.YELLOW
        else:
            return cls.RED


# ===== Pydantic 模型 =====


class TreatmentGoal(BaseModel):
    """单个治疗目标（Talen & Schindler 模型）"""
    id: str = Field(description="目标 ID，如 goal_001")
    description: str = Field(description="具体目标描述")
    indicator: str = Field(description="可测量指标 — 如何判断目标是否达成")
    timeline: str = Field(description="预估时间线，如 '2-4次会谈'")
    priority: int = Field(default=1, ge=1, le=3, description="优先级 1-3，1 最高")
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="进展百分比 0.0-1.0")
    progress_status: str = Field(default="green", description="绿/黄/红/白 状态")
    notes: str = Field(default="", description="进展备注")
    created_at: str = Field(default="", description="创建时间")
    updated_at: str = Field(default="", description="最后更新时间")


class SkillUsageRecord(BaseModel):
    """单次技能使用记录"""
    session_id: str = Field(description="会话 ID")
    session_date: str = Field(description="会话日期")
    skill_name: str = Field(description="使用的技能名称")
    turn_count: int = Field(default=0, description="本次会话中该技能使用的轮数")
    effectiveness_rating: Optional[int] = Field(
        default=None, ge=1, le=5,
        description="效果评分 1-5（基于来访者反应和对话进展）"
    )
    notes: str = Field(default="", description="使用效果备注")


class InterventionResponse(BaseModel):
    """干预反应记录（同步到画像 Domain 8）"""
    skill_name: str = Field(description="技能名称")
    session_id: str = Field(description="来源会话 ID")
    effect: Literal["effective", "neutral", "ineffective", "harmful"] = Field(
        default="neutral", description="效果分类"
    )
    evidence: str = Field(description="支持该判断的对话证据")
    recorded_at: str = Field(default="", description="记录时间")


class TreatmentPlanOutput(BaseModel):
    """PlanGenerator Agent 的结构化输出"""

    # 个案概念化
    case_conceptualization: str = Field(
        description="基于原始资料的整体理解 — 核心困扰、维持因素、资源与风险"
    )

    # 治疗方向
    primary_approach: str = Field(description="主要治疗方法，如 cbt / psychodynamic / sfbt")
    primary_approach_rationale: str = Field(description="选择该方法的理由，引用找到的原始资料")
    secondary_approaches: list[str] = Field(
        default_factory=list,
        description="辅助方法列表（1-3项）"
    )
    cautionary_approaches: list[str] = Field(
        default_factory=list,
        description="需要谨慎使用的方法及原因简述"
    )

    # 阶段
    stage: str = Field(description="当前治疗阶段: engagement / cognitive_behavioral / emotional_deepening / consolidation")
    stage_rationale: str = Field(description="阶段判断依据")

    # 目标
    goals: list[dict] = Field(
        default_factory=list,
        description="2-4个治疗目标 [{description, indicator, timeline, priority}]"
    )

    # 不确定性
    information_gaps: list[str] = Field(
        default_factory=list,
        description="当前缺少的信息"
    )
    verification_needed: list[str] = Field(
        default_factory=list,
        description="需要在治疗中验证的假设"
    )


class PlanUpdateOutput(BaseModel):
    """PlanManager.update_plan() 的 LLM 输出"""

    # 会话总结
    session_summary: str = Field(description="本次会话一句话总结")

    # 目标进展更新
    goal_updates: dict[str, float] = Field(
        default_factory=dict,
        description="goal_id → 新进展百分比"
    )
    goal_status_changes: dict[str, str] = Field(
        default_factory=dict,
        description="goal_id → 新进展状态 (green/yellow/red/white)"
    )

    # 技能使用
    skills_used: list[str] = Field(default_factory=list, description="本次会话使用的技能")
    skill_effectiveness: dict[str, int] = Field(
        default_factory=dict,
        description="skill_name → 效果评分 1-5"
    )

    # 干预效果
    effective_interventions: list[str] = Field(
        default_factory=list,
        description="本次有效的干预方式"
    )
    ineffective_interventions: list[str] = Field(
        default_factory=list,
        description="本次无效或引发防御的干预方式"
    )

    # 连续性
    new_pending_items: list[str] = Field(
        default_factory=list,
        description="新的未完成事项/下次任务"
    )
    completed_items: list[str] = Field(
        default_factory=list,
        description="本次完成的 pending_items"
    )

    # 阶段
    should_transition_stage: bool = Field(default=False, description="是否建议切换阶段")
    suggested_stage: str = Field(default="", description="建议切换到的阶段")
    stage_transition_reason: str = Field(default="", description="阶段切换理由")


class TreatmentPlan(BaseModel):
    """完整的治疗计划"""
    user_id: str = Field(default="default", description="用户 ID")
    version: int = Field(default=0, description="版本号")
    created_at: str = Field(default="", description="创建时间 ISO 格式")
    updated_at: str = Field(default="", description="最后更新时间 ISO 格式")

    # 个案概念化
    case_conceptualization: str = Field(default="", description="个案概念化文本")

    # 核心治疗方向
    primary_approach: str = Field(default="person-centered", description="主要治疗方法")
    primary_approach_rationale: str = Field(default="", description="选择理由")
    secondary_approaches: list[str] = Field(default_factory=list, description="辅助方法")
    cautionary_approaches: list[str] = Field(default_factory=list, description="谨慎使用的方法")

    # 治疗阶段
    stage: str = Field(default="engagement", description="当前治疗阶段")

    # 目标体系
    goals: list[TreatmentGoal] = Field(default_factory=list, description="治疗目标列表")

    # 历史追踪
    skill_usage_history: list[SkillUsageRecord] = Field(
        default_factory=list, description="技能使用历史"
    )
    intervention_responses: list[InterventionResponse] = Field(
        default_factory=list, description="干预反应记录"
    )

    # 版本历史
    version_history: list[dict] = Field(
        default_factory=list,
        description="版本历史 [{version, changed_fields, timestamp, source_session}]"
    )

    # 跨会话连续性
    last_session_plan: str = Field(default="", description="上一会话 PAIP 的 Plan 部分")
    pending_items: list[str] = Field(default_factory=list, description="未完成的任务/议程")
    supervisor_notes: str = Field(default="", description="督导最新评估和建议")
    source_sessions: list[str] = Field(default_factory=list, description="已参与的会话 ID 列表")

    # 辅助方法
    def get_active_approaches(self) -> list[str]:
        """获取当前激活的所有方法（主要 + 辅助）"""
        result = [self.primary_approach] if self.primary_approach else []
        result.extend(self.secondary_approaches)
        return result

    def get_stage_label(self) -> str:
        """获取当前阶段的中文标签"""
        return STAGE_LABELS.get(self.stage, self.stage)

    def get_stage_index(self) -> int:
        """获取当前阶段在阶段序列中的索引"""
        try:
            return STAGE_ORDER.index(self.stage)
        except ValueError:
            return 0

    def is_empty(self) -> bool:
        """检查计划是否为空（尚未通过 PlanGenerator 初始化）。"""
        return self.version == 0
