# test/test_treatment_continuity.py
"""
Integration tests for treatment continuity system.

Tests the full flow:
  PlanGenerator output → TreatmentPlan model → build continuity context →
  Supervisor plan monitoring fields → PlanManager update
"""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from treatment_plan import (
    TreatmentPlan,
    TreatmentGoal,
    TreatmentPlanOutput,
    PlanUpdateOutput,
    SkillUsageRecord,
    InterventionResponse,
    ProgressStatus,
    STAGE_ORDER,
    STAGE_DEFAULT_SKILLS,
)
from plan_manager import (
    load_plan,
    save_plan,
    build_continuity_context,
    should_transition_stage,
)


class TestFullContinuityFlow:
    """Test the end-to-end treatment continuity flow."""

    def test_generate_to_save_to_load_roundtrip(self, tmp_path, monkeypatch):
        """Simulate: PlanGenerator output → save → load → build context → inject."""
        import plan_manager
        monkeypatch.setattr(plan_manager, "PLANS_DIR", tmp_path / "plans")

        # Step 1: Simulate PlanGenerator output
        plan_output = TreatmentPlanOutput(
            case_conceptualization="来访者表现中度焦虑，有明显的完美主义和灾难化思维模式，来自高期望家庭背景。联盟已建立。",
            primary_approach="cbt",
            primary_approach_rationale="检索到日记中反复出现'我必须完美'等绝对化语言，PAIP摘要显示认知层面歪曲显著",
            secondary_approaches=["person-centered", "behavioral-third-wave"],
            cautionary_approaches=["psychodynamic", "gestalt"],
            stage="cognitive_behavioral",
            stage_rationale="来访者已完成建立阶段，联盟稳固，准备进行结构化认知工作",
            goals=[
                {
                    "description": "识别自动化负面思维",
                    "indicator": "思维记录表连续2周>=5天完成",
                    "timeline": "4-6次会谈",
                    "priority": 1,
                },
                {
                    "description": "减少焦虑评分至临床阈值以下",
                    "indicator": "GAD-7 < 5",
                    "timeline": "6-8次会谈",
                    "priority": 1,
                },
                {
                    "description": "建立替代性认知模式",
                    "indicator": "至少3种常见情境中能使用认知重构",
                    "timeline": "4-6次会谈",
                    "priority": 2,
                },
            ],
            information_gaps=["来访者是否尝试过既往治疗未知"],
            verification_needed=["完美主义是否泛化到所有生活领域，还是仅限于工作"],
        )

        # Step 2: Convert to TreatmentPlan
        from plan_generator import plan_output_to_treatment_plan
        plan = plan_output_to_treatment_plan(
            plan_output.model_dump(),
            user_id="test_user",
            source_session="session_initial",
        )

        assert plan.primary_approach == "cbt"
        assert plan.stage == "cognitive_behavioral"
        assert "psychodynamic" in plan.cautionary_approaches
        assert "gestalt" in plan.cautionary_approaches
        assert len(plan.goals) == 3
        assert plan.version == 1

        # Step 3: Save
        save_plan(plan)

        # Step 4: Load
        loaded = load_plan("test_user")
        assert loaded is not None
        assert loaded.primary_approach == plan.primary_approach
        assert loaded.stage == plan.stage
        assert loaded.cautionary_approaches == plan.cautionary_approaches

        # Step 5: Build continuity context for chatter
        ctx = build_continuity_context(loaded)
        assert "cbt" in ctx
        assert "工作期-认知行为" in ctx
        assert "识别自动化负面思维" in ctx  # goal description, not indicator

    def test_stage_progression_flow(self):
        """Test the stage progression from engagement → cognitive_behavioral → emotional_deepening → consolidation."""
        # Start at engagement
        plan = TreatmentPlan(
            user_id="test_user",
            stage="engagement",
            primary_approach="clinical-interviewing",
            goals=[
                TreatmentGoal(
                    id="goal_001",
                    description="建立治疗联盟",
                    indicator="WAI评分>=60",
                    timeline="2次会谈",
                    priority=1,
                    progress=0.9,
                    progress_status="green",
                ),
                TreatmentGoal(
                    id="goal_002",
                    description="完成初步评估",
                    indicator="14领域画像初始化完成",
                    timeline="1-2次会谈",
                    priority=1,
                    progress=0.8,
                    progress_status="green",
                ),
            ],
        )

        should, suggested, reason = should_transition_stage(plan)
        assert should is True
        assert suggested == "cognitive_behavioral"

        # Move to cognitive_behavioral stage
        plan.stage = "cognitive_behavioral"
        plan.primary_approach = "cbt"
        plan.goals = [
            TreatmentGoal(
                id="goal_003",
                description="识别自动化思维",
                indicator="完成率>80%",
                timeline="4-6次",
                priority=1,
                progress=0.7,
                progress_status="green",
            ),
            TreatmentGoal(
                id="goal_004",
                description="减少焦虑",
                indicator="GAD-7 < 5",
                timeline="6-8次",
                priority=1,
                progress=0.65,
                progress_status="green",
            ),
        ]

        should2, suggested2, reason2 = should_transition_stage(plan)
        assert should2 is True
        assert suggested2 == "emotional_deepening"

    def test_supervisor_plan_feedback_flow(self, tmp_path, monkeypatch):
        """Test that supervisor plan feedback is properly captured and stored."""
        import plan_manager
        monkeypatch.setattr(plan_manager, "PLANS_DIR", tmp_path / "plans")

        # Create a plan
        plan = TreatmentPlan(
            user_id="test_user",
            version=1,
            primary_approach="cbt",
            stage="cognitive_behavioral",
            goals=[
                TreatmentGoal(
                    id="goal_001",
                    description="减少焦虑",
                    indicator="GAD-7<5",
                    timeline="6次",
                    progress=0.3,
                    progress_status="yellow",
                ),
            ],
        )
        save_plan(plan)

        # Simulate supervisor feedback accumulation across session
        feedbacks = [
            "第1轮：Chatter使用person-centered修复联盟，属正常灵活性调整",
            "第3轮：Chatter偏离计划，连续使用psychodynamic探索，缺乏CBT结构化工作。建议提醒回归计划。",
            "会话结束：方法一致性评估 — 60%时间在计划内，建议下次重点回归CBT认知重构",
        ]

        plan.supervisor_notes = "\n".join(feedbacks)
        save_plan(plan)

        # Reload and verify
        loaded = load_plan("test_user")
        assert loaded is not None
        assert "回归计划" in loaded.supervisor_notes
        assert "psychodynamic" in loaded.supervisor_notes

    def test_dynamic_skill_catalog_ordering(self):
        """Test that the skill catalog reordering logic correctly prioritizes plan approaches."""
        from skill_loader import get_chatter_skill_catalog_for_plan, get_supervisor_skill_catalog_for_plan

        # Plan: CBT primary, gestalt/psychodynamic cautious
        catalog = get_chatter_skill_catalog_for_plan(
            primary_approaches=["cbt"],
            secondary_approaches=["person-centered"],
            cautionary_approaches=["psychodynamic", "gestalt"],
            stage="cognitive_behavioral",
        )

        assert "cbt" in catalog
        assert "★" in catalog  # Should have star markers
        assert "psychodynamic" in catalog
        assert "gestalt" in catalog

        # Verify ordering: primary section comes before cautious
        cbt_pos = catalog.find("cbt")
        cautionary_pos = catalog.find("谨慎使用")
        assert cbt_pos < cautionary_pos, "Primary approaches should appear before cautionary section"

        # Crisis-intervention should NOT be in the cautious section
        # (it's always preserved as Tier 3 crisis priority)
        assert "crisis-intervention" in catalog

        # Test supervisor catalog too
        sup_catalog = get_supervisor_skill_catalog_for_plan(
            primary_approaches=["cbt"],
            stage="cognitive_behavioral",
        )
        assert "alliance-monitoring" in sup_catalog or "process-quality" in sup_catalog
