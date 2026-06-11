# test/test_plan_manager.py
"""
Unit tests for PlanManager — load/save, update plan, stage transitions.
"""
import json
import sys
import tempfile
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from treatment_plan import (
    TreatmentPlan,
    TreatmentGoal,
    SkillUsageRecord,
    InterventionResponse,
    ProgressStatus,
    STAGE_ORDER,
)
from plan_manager import (
    load_plan,
    save_plan,
    build_continuity_context,
    should_transition_stage,
)


class TestLoadSave:
    def test_load_nonexistent_plan(self, tmp_path, monkeypatch):
        """Loading a plan that doesn't exist returns None."""
        import plan_manager
        monkeypatch.setattr(plan_manager, "PLANS_DIR", tmp_path / "plans")
        plan = load_plan("nonexistent_user")
        assert plan is None

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        """Save a plan then load it — should match."""
        import plan_manager
        monkeypatch.setattr(plan_manager, "PLANS_DIR", tmp_path / "plans")

        plan = TreatmentPlan(
            user_id="test_user",
            version=1,
            primary_approach="cbt",
            stage="cognitive_behavioral",
            goals=[
                TreatmentGoal(
                    id="goal_001",
                    description="减少焦虑",
                    indicator="GAD-7 < 5",
                    timeline="6 sessions",
                    progress=0.5,
                    progress_status="yellow",
                )
            ],
        )
        save_plan(plan)

        loaded = load_plan("test_user")
        assert loaded is not None
        assert loaded.user_id == "test_user"
        assert loaded.version == 1
        assert loaded.primary_approach == "cbt"
        assert loaded.stage == "cognitive_behavioral"
        assert len(loaded.goals) == 1
        assert loaded.goals[0].progress == 0.5

    def test_save_creates_version_snapshot(self, tmp_path, monkeypatch):
        """Saving should also write a version snapshot."""
        import plan_manager
        monkeypatch.setattr(plan_manager, "PLANS_DIR", tmp_path / "plans")

        plan = TreatmentPlan(user_id="test_user", version=3, primary_approach="sfbt")
        save_plan(plan)

        version_file = tmp_path / "plans" / "test_user" / "versions" / "v3.json"
        assert version_file.exists()
        data = json.loads(version_file.read_text(encoding="utf-8"))
        assert data["primary_approach"] == "sfbt"


class TestBuildContinuityContext:
    def test_empty_plan(self):
        plan = TreatmentPlan(user_id="test_user")
        assert plan.is_empty()
        ctx = build_continuity_context(plan)
        assert ctx == ""

    def test_minimal_plan_context(self):
        plan = TreatmentPlan(
            user_id="test_user",
            version=1,
            primary_approach="cbt",
            stage="cognitive_behavioral",
            last_session_plan="继续思维记录练习",
        )
        ctx = build_continuity_context(plan)
        assert "工作期-认知行为" in ctx
        assert "cbt" in ctx
        assert "继续思维记录练习" in ctx

    def test_plan_with_cautionary_approaches(self):
        plan = TreatmentPlan(
            user_id="test_user",
            version=1,
            primary_approach="cbt",
            secondary_approaches=["person-centered"],
            cautionary_approaches=["psychodynamic", "gestalt"],
            stage="cognitive_behavioral",
        )
        ctx = build_continuity_context(plan)
        assert "⚠" in ctx or "谨慎" in ctx
        assert "psychodynamic" in ctx or "gestalt" in ctx

    def test_plan_with_goals(self):
        plan = TreatmentPlan(
            user_id="test_user",
            version=1,
            primary_approach="cbt",
            stage="cognitive_behavioral",
            goals=[
                TreatmentGoal(
                    id="goal_001",
                    description="减少自动化负面思维",
                    indicator="思维记录完成率>80%",
                    timeline="4-6次",
                    progress=0.0,
                    progress_status="green",
                ),
                TreatmentGoal(
                    id="goal_002",
                    description="改善睡眠",
                    indicator="入睡<30分钟",
                    timeline="3-5次",
                    progress=0.2,
                    progress_status="red",
                ),
            ],
        )
        ctx = build_continuity_context(plan)
        assert "减少自动化负面思维" in ctx
        assert "改善睡眠" in ctx

    def test_plan_with_pending_items(self):
        plan = TreatmentPlan(
            user_id="test_user",
            version=1,
            primary_approach="cbt",
            pending_items=["完成思维记录表", "阅读CBT手册第3章"],
        )
        ctx = build_continuity_context(plan)
        assert "思维记录表" in ctx
        assert "CBT手册" in ctx


class TestShouldTransitionStage:
    def test_no_transition_at_last_stage(self):
        plan = TreatmentPlan(
            user_id="test_user",
            stage="consolidation",
            goals=[
                TreatmentGoal(
                    id="goal_001",
                    description="维持改善",
                    indicator="连续4周稳定",
                    timeline="4周",
                    progress=0.9,
                    progress_status="white",
                )
            ],
        )
        should, suggested, reason = should_transition_stage(plan)
        assert should is False

    def test_no_transition_when_goals_not_green(self):
        plan = TreatmentPlan(
            user_id="test_user",
            stage="cognitive_behavioral",
            goals=[
                TreatmentGoal(
                    id="goal_001",
                    description="减少焦虑",
                    indicator="GAD-7 < 5",
                    timeline="6次",
                    priority=1,
                    progress=0.2,
                    progress_status="red",
                )
            ],
        )
        should, suggested, reason = should_transition_stage(plan)
        assert should is False

    def test_transition_when_goals_green(self):
        plan = TreatmentPlan(
            user_id="test_user",
            stage="cognitive_behavioral",
            goals=[
                TreatmentGoal(
                    id="goal_001",
                    description="识别自动化思维",
                    indicator="思维记录完成率>80%",
                    timeline="4次",
                    priority=1,
                    progress=0.8,
                    progress_status="green",
                ),
                TreatmentGoal(
                    id="goal_002",
                    description="减少焦虑",
                    indicator="GAD-7 < 5",
                    timeline="6次",
                    priority=1,
                    progress=0.7,
                    progress_status="green",
                ),
            ],
        )
        should, suggested, reason = should_transition_stage(plan)
        assert should is True
        assert suggested == "emotional_deepening"

    def test_no_transition_with_crisis_signal(self):
        plan = TreatmentPlan(
            user_id="test_user",
            stage="cognitive_behavioral",
            goals=[
                TreatmentGoal(
                    id="goal_001",
                    description="管理自杀风险",
                    indicator="无自杀意念连续2周",
                    timeline="持续",
                    priority=1,
                    progress=0.5,
                    progress_status="red",
                ),
                TreatmentGoal(
                    id="goal_002",
                    description="减少焦虑",
                    indicator="GAD-7 < 5",
                    timeline="6次",
                    priority=2,
                    progress=0.8,
                    progress_status="green",
                ),
            ],
        )
        should, suggested, reason = should_transition_stage(plan)
        assert should is False
