# test/test_treatment_plan.py
"""
Unit tests for TreatmentPlan data model.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from treatment_plan import (
    TreatmentPlan,
    TreatmentGoal,
    SkillUsageRecord,
    InterventionResponse,
    TreatmentPlanOutput,
    PlanUpdateOutput,
    ProgressStatus,
    STAGE_ORDER,
    STAGE_LABELS,
    STAGE_DEFAULT_SKILLS,
)


class TestTreatmentGoal:
    def test_create_minimal_goal(self):
        goal = TreatmentGoal(
            id="goal_001",
            description="减少焦虑症状",
            indicator="GAD-7 评分降至 5 以下",
            timeline="4-6次会谈",
        )
        assert goal.priority == 1
        assert goal.progress == 0.0
        assert goal.progress_status == "green"

    def test_create_full_goal(self):
        goal = TreatmentGoal(
            id="goal_002",
            description="改善睡眠质量",
            indicator="入睡时间 < 30分钟，每周>=5天",
            timeline="3-5次会谈",
            priority=2,
            progress=0.4,
            progress_status="yellow",
            notes="本周有改善但仍有波动",
        )
        assert goal.id == "goal_002"
        assert goal.priority == 2
        assert goal.progress == 0.4
        assert goal.progress_status == "yellow"

    def test_goal_serialization_roundtrip(self):
        goal = TreatmentGoal(
            id="goal_003",
            description="建立健康的人际边界",
            indicator="每周至少1次成功表达拒绝",
            timeline="6-8次会谈",
            priority=1,
            progress=0.6,
            progress_status="green",
        )
        data = goal.model_dump()
        restored = TreatmentGoal(**data)
        assert restored.id == goal.id
        assert restored.description == goal.description
        assert restored.indicator == goal.indicator
        assert restored.progress == goal.progress
        assert restored.progress_status == goal.progress_status


class TestProgressStatus:
    def test_white_at_high_progress(self):
        assert ProgressStatus.from_progress(0.9) == ProgressStatus.WHITE
        assert ProgressStatus.from_progress(0.85) == ProgressStatus.WHITE

    def test_green_at_moderate_progress(self):
        assert ProgressStatus.from_progress(0.7) == ProgressStatus.GREEN
        assert ProgressStatus.from_progress(0.6) == ProgressStatus.GREEN

    def test_yellow_at_low_progress(self):
        assert ProgressStatus.from_progress(0.5) == ProgressStatus.YELLOW
        assert ProgressStatus.from_progress(0.3) == ProgressStatus.YELLOW

    def test_red_at_very_low_progress(self):
        assert ProgressStatus.from_progress(0.2) == ProgressStatus.RED
        assert ProgressStatus.from_progress(0.0) == ProgressStatus.RED


class TestTreatmentPlan:
    def test_create_empty_plan(self):
        plan = TreatmentPlan(user_id="test_user")
        assert plan.user_id == "test_user"
        assert plan.version == 0
        assert plan.stage == "engagement"
        assert plan.primary_approach == "person-centered"
        # A plan with only defaults (no goals, no conceptualization) is considered empty
        plan2 = TreatmentPlan(user_id="test_user2")
        assert plan2.is_empty() is True  # no goals, no conceptualization

    def test_plan_with_goals(self):
        plan = TreatmentPlan(
            user_id="test_user",
            version=1,
            primary_approach="cbt",
            secondary_approaches=["person-centered", "behavioral-third-wave"],
            stage="cognitive_behavioral",
            goals=[
                TreatmentGoal(
                    id="goal_001",
                    description="减少自动化负面思维",
                    indicator="思维记录表连续2周>=5天完成",
                    timeline="4-6次会谈",
                    progress=0.5,
                    progress_status="yellow",
                )
            ],
        )
        assert plan.is_empty() is False
        assert plan.get_stage_label() == "工作期-认知行为"
        assert plan.get_stage_index() == 1
        assert plan.get_active_approaches() == ["cbt", "person-centered", "behavioral-third-wave"]

    def test_serialization_roundtrip(self):
        plan = TreatmentPlan(
            user_id="test_user",
            version=2,
            primary_approach="sfbt",
            secondary_approaches=["person-centered"],
            cautionary_approaches=["psychodynamic"],
            stage="engagement",
            goals=[
                TreatmentGoal(
                    id="goal_001",
                    description="明确治疗目标",
                    indicator="来访者能用一句话说出咨询期望",
                    timeline="1-2次会谈",
                    priority=1,
                    progress=0.8,
                    progress_status="green",
                )
            ],
            skill_usage_history=[
                SkillUsageRecord(
                    session_id="session_01",
                    session_date="26.06.10",
                    skill_name="clinical-interviewing",
                    effectiveness_rating=4,
                )
            ],
            intervention_responses=[
                InterventionResponse(
                    skill_name="person-centered",
                    session_id="session_01",
                    effect="effective",
                    evidence="来访者表示'第一次有人真正理解我'",
                )
            ],
            last_session_plan="下周完成思维记录表",
            pending_items=["下周完成思维记录表"],
        )
        data = plan.model_dump()
        restored = TreatmentPlan(**data)
        assert restored.user_id == plan.user_id
        assert restored.version == 2
        assert restored.primary_approach == "sfbt"
        assert restored.cautionary_approaches == ["psychodynamic"]
        assert len(restored.goals) == 1
        assert restored.goals[0].description == "明确治疗目标"
        assert restored.goals[0].progress == 0.8
        assert len(restored.skill_usage_history) == 1
        assert restored.skill_usage_history[0].skill_name == "clinical-interviewing"
        assert len(restored.intervention_responses) == 1
        assert restored.intervention_responses[0].effect == "effective"
        assert restored.last_session_plan == "下周完成思维记录表"


class TestStageConstants:
    def test_stage_order(self):
        assert len(STAGE_ORDER) == 4
        assert STAGE_ORDER[0] == "engagement"
        assert STAGE_ORDER[-1] == "consolidation"

    def test_stage_labels(self):
        assert STAGE_LABELS["engagement"] == "建立期"
        assert STAGE_LABELS["cognitive_behavioral"] == "工作期-认知行为"
        assert STAGE_LABELS["emotional_deepening"] == "工作期-情感深化"
        assert STAGE_LABELS["consolidation"] == "整合/收尾期"

    def test_each_stage_has_default_skills(self):
        for stage in STAGE_ORDER:
            assert stage in STAGE_DEFAULT_SKILLS
            defaults = STAGE_DEFAULT_SKILLS[stage]
            assert "primary" in defaults
            assert "secondary" in defaults
            assert "cautionary" in defaults
            # person-centered should always be in primary or secondary
            all_approaches = defaults["primary"] + defaults["secondary"]
            assert "person-centered" in all_approaches, f"{stage} should include person-centered"


class TestTreatmentPlanOutput:
    def test_minimal_output(self):
        output = TreatmentPlanOutput(
            case_conceptualization="来访者表现出中度焦虑和完美主义倾向",
            primary_approach="cbt",
            primary_approach_rationale="认知重构已被证明对完美主义相关的焦虑有效",
            secondary_approaches=["person-centered"],
            stage="cognitive_behavioral",
            stage_rationale="来访者已建立基本联盟，准备好进行结构化认知工作",
            goals=[
                {
                    "description": "识别并挑战自动化负面思维",
                    "indicator": "思维记录表连续两周完成",
                    "timeline": "4-6次会谈",
                    "priority": 1,
                }
            ],
        )
        assert output.primary_approach == "cbt"
        assert len(output.goals) == 1
        assert output.goals[0]["description"] == "识别并挑战自动化负面思维"


class TestPlanUpdateOutput:
    def test_minimal_update(self):
        output = PlanUpdateOutput(
            session_summary="本次讨论了自动化思维的识别练习",
        )
        assert output.session_summary == "本次讨论了自动化思维的识别练习"
        assert output.should_transition_stage is False
        assert output.suggested_stage == ""

    def test_stage_transition(self):
        output = PlanUpdateOutput(
            session_summary="认知行为阶段目标基本达成",
            should_transition_stage=True,
            suggested_stage="emotional_deepening",
            stage_transition_reason="当前阶段3个目标中2个已绿色，来访者表达了对更深层情感探索的兴趣",
        )
        assert output.should_transition_stage is True
        assert output.suggested_stage == "emotional_deepening"
