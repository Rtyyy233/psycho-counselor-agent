"""
PlanGenerator — adaptive multi-round treatment plan formulation.

Round 1:  14 parallel broad queries (fixed coverage)
Round 2-4: LLM reviews data → decides: output_plan OR need_more_data (specifies queries dynamically)
Round 5:   forced output (hard cutoff)

Code layer guarantees:
- Max 5 rounds, max 30 total queries
- All queries within a round run in parallel (asyncio.gather)
- Duplicate queries detected and skipped
- LLM binary choice per round: done or need more
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from langchain_deepseek import ChatDeepSeek

from config import LLM_MODEL, DATA_DIR

logger = logging.getLogger(__name__)

# ===== LLM instances =====

plan_llm = ChatDeepSeek(model=LLM_MODEL, temperature=0.1)

# ===== Paths =====

PLANS_DIR = DATA_DIR / "treatment_plans"


def _plan_path(user_id: str) -> Path:
    return PLANS_DIR / user_id / "plan.json"


def _version_dir(user_id: str) -> Path:
    return PLANS_DIR / user_id / "versions"


# ===== Round decision model =====

class RoundDecision(BaseModel):
    """LLM decides at end of each round: ready to output, or need more data?"""
    action: str = Field(
        description="'output_plan' if you have enough data to formulate a complete plan; "
                    "'need_more_data' if critical information is still missing"
    )
    reasoning: str = Field(description="Why this decision — what do you have, what is missing?")

    # When action == "output_plan" — full plan output
    case_conceptualization: str = Field(default="", description="Full case conceptualization")
    primary_approach: str = Field(default="", description="Primary treatment method, e.g. 'cbt'")
    primary_approach_rationale: str = Field(default="", description="Why this approach, citing data")
    secondary_approaches: list[str] = Field(default_factory=list)
    cautionary_approaches: list[str] = Field(default_factory=list)
    stage: str = Field(default="engagement", description="engagement|cognitive_behavioral|emotional_deepening|consolidation")
    stage_rationale: str = Field(default="")
    goals: list[dict] = Field(default_factory=list, description="[{description, indicator, timeline, priority}]")
    information_gaps: list[str] = Field(default_factory=list)
    verification_needed: list[str] = Field(default_factory=list)

    # When action == "need_more_data" — specify what to search next
    follow_up_queries: list[dict] = Field(
        default_factory=list,
        description="[{'source': 'diary'|'conv'|'material'|'profile', 'query': '...'}]"
    )


# ===== Retrieval helpers =====

async def _retrieve_diary(query: str) -> str:
    try:
        from mem_retrieve_diary import retrieve_diary
        results = await retrieve_diary(query)
        if not results:
            return ""
        parts = []
        for r in results[:5]:
            docs = getattr(r, "documents", []) or []
            for doc in docs[:3]:
                content = getattr(doc, "page_content", str(doc))[:500]
                if content.strip():
                    parts.append(content)
        return "\n---\n".join(parts) if parts else ""
    except Exception as e:
        logger.warning("Diary retrieval failed for '%s': %s", query[:40], e)
        return ""


async def _retrieve_materials(query: str) -> str:
    try:
        from mem_retrieve_material import retrieve_materials
        results = await retrieve_materials(query)
        if not results:
            return ""
        parts = []
        for r in results[:5]:
            docs = getattr(r, "matched_children", []) or []
            for doc in docs[:3]:
                content = getattr(doc, "page_content", str(doc))[:500]
                if content.strip():
                    parts.append(content)
        return "\n---\n".join(parts) if parts else ""
    except Exception as e:
        logger.warning("Materials retrieval failed for '%s': %s", query[:40], e)
        return ""


async def _retrieve_conv_outline(query: str) -> str:
    try:
        from mem_retrieve_conv_outline import retrieve_conv_outline
        results = await retrieve_conv_outline(query)
        if not results:
            return ""
        parts = []
        for r in results[:5]:
            docs = getattr(r, "matched_docs", []) or []
            for doc in docs[:3]:
                content = getattr(doc, "page_content", str(doc))[:500]
                if content.strip():
                    parts.append(content)
        return "\n---\n".join(parts) if parts else ""
    except Exception as e:
        logger.warning("Conv outline retrieval failed for '%s': %s", query[:40], e)
        return ""


async def _retrieve_user_profile(user_id: str) -> str:
    try:
        from mem_retrieve_user_profile import retrieve_user_profile
        result = await retrieve_user_profile(user_id)
        if result:
            return result.full_text[:5000]
        return ""
    except Exception as e:
        logger.warning("User profile retrieval failed: %s", e)
        return ""


# Map source name → retrieval function
_SOURCE_RETRIEVERS = {
    "diary": _retrieve_diary,
    "conv": _retrieve_conv_outline,
    "material": _retrieve_materials,
    "profile": lambda q: _retrieve_user_profile(q),  # q is user_id
}


def _normalize_query(query: str) -> str:
    """Strip whitespace, lowercase for dedup."""
    return " ".join(query.strip().lower().split())


# ===== Stored-data accumulation =====

class DataAccumulator:
    """Accumulates retrieval results across rounds, with dedup."""

    def __init__(self):
        self.seen_queries: set[str] = set()
        self.sections: list[tuple[str, str]] = []  # (label, content)
        self.total_queries = 0

    def has_query(self, query: str) -> bool:
        return _normalize_query(query) in self.seen_queries

    def mark_query(self, query: str) -> None:
        self.seen_queries.add(_normalize_query(query))
        self.total_queries += 1

    async def execute_queries(self, queries: list[dict]) -> list[dict]:
        """Run a batch of source+query pairs in parallel. Skips duplicates."""
        results = []
        tasks = []
        executed_queries = []  # track which queries actually get executed

        for q in queries:
            source = q.get("source", "diary")
            query_text = q.get("query", "")
            if not query_text or self.has_query(query_text):
                continue
            self.mark_query(query_text)

            retriever = _SOURCE_RETRIEVERS.get(source, _retrieve_diary)
            tasks.append(retriever(query_text))
            executed_queries.append(q)

        if not tasks:
            return results

        raw = await asyncio.gather(*tasks, return_exceptions=True)

        for q, content in zip(executed_queries, raw):
            source = q.get("source", "diary")
            query_text = q.get("query", "")
            if isinstance(content, Exception):
                logger.warning("Query '%s' failed: %s", query_text[:40], content)
                continue
            if isinstance(content, str) and content.strip():
                source_label = {"diary": "日记", "conv": "PAIP摘要", "material": "材料", "profile": "画像"}.get(source, source)
                self.sections.append((f"[{source_label}] {query_text}", content))
                results.append({"source": source, "query": query_text, "content": content})

        return results

    def assemble(self, user_id: str, round_num: int) -> str:
        """Build the combined data text for LLM round input."""
        header = (
            f"Client: {user_id}\nRound: {round_num}\n"
            f"Queries so far: {self.total_queries}\n\n"
        )
        body_parts = [f"=== {label} ===\n\n{content}" for label, content in self.sections]
        return header + "\n\n".join(body_parts)


# ===== Prompts =====

ROUND_SYSTEM_PROMPT = """You are a senior treatment planning expert reviewing retrieved client data.

At each round you are given ALL the data accumulated so far. Your job:

## If you have enough data → action: "output_plan"
Produce a complete treatment plan with ALL these fields:
- case_conceptualization: Core problem, maintaining factors, Prochaska change stage, resources, risks. Cite specific data.
- primary_approach (1-2 from list below): With rationale citing evidence found.
- secondary_approaches (1-3)
- cautionary_approaches: Only mark if you have SPECIFIC evidence the method is unsuitable.
- stage: engagement | cognitive_behavioral | emotional_deepening | consolidation
- stage_rationale: Why this stage.
- goals (2-4): Each with {description, indicator, timeline, priority 1-3}.
- information_gaps: What is still unknown.
- verification_needed: Assumptions to test in therapy.

## If you need MORE data → action: "need_more_data"
Specify follow_up_queries as [{"source": "diary"|"conv"|"material"|"profile", "query": "..."}]
- Only ask for CRITICAL missing information — not nice-to-have.
- Each query should target ONE specific angle.
- source "profile" queries will retrieve the full user profile (the query text is the user_id).

## Available approaches
person-centered, existential, psychodynamic, adlerian, gestalt, cbt, behavioral-third-wave,
choice-reality, sfbt, narrative, feminist, family-systems, alliance-repair,
clinical-interviewing, crisis-intervention

## Stage guidance
- engagement: session 1-2, alliance + assessment. Favors clinical-interviewing + person-centered.
- cognitive_behavioral: structured cognitive/behavioral work. cbt/sfbt/behavioral-third-wave.
- emotional_deepening: experiential work. Only after solid alliance. gestalt/psychodynamic/existential.
- consolidation: integrate gains, prevent relapse. narrative + person-centered.

## Principles
- Decide based on DATA FOUND, not assumptions
- Safety first — risk factors must be addressed
- Personalize — same symptoms ≠ same plan
- If data truly insufficient → output_plan with honest information_gaps"""

FORCE_OUTPUT_PROMPT_EXTRA = """
THIS IS YOUR LAST ROUND. You MUST output a plan now with action="output_plan".
If data is insufficient, produce your best plan and mark gaps honestly in information_gaps.
Do NOT ask for more data."""


# ===== Round 1 broad queries =====

def _build_round1_queries() -> list[dict]:
    """14 broad queries covering all data sources."""
    return [
        # Diary — 5 angles
        {"source": "diary", "query": "情绪状态 心情 焦虑 抑郁 恐惧"},
        {"source": "diary", "query": "人际关系 家庭 朋友 同事 亲密关系"},
        {"source": "diary", "query": "自我认知 自我价值 自信 自尊"},
        {"source": "diary", "query": "成长经历 童年 创伤 重要事件"},
        {"source": "diary", "query": "近期状态 压力 睡眠 饮食 日常"},

        # Conv outlines (PAIP) — 5 angles
        {"source": "conv", "query": "来访者主诉 核心困扰 求助原因"},
        {"source": "conv", "query": "评估 诊断 精神状态 风险评估"},
        {"source": "conv", "query": "干预历史 治疗方法 技术使用"},
        {"source": "conv", "query": "治疗效果 来访者反应 改善或恶化"},
        {"source": "conv", "query": "治疗计划 后续步骤 目标 未完成事项"},

        # Materials — 3 angles
        {"source": "material", "query": "心理咨询 治疗框架 评估方法"},
        {"source": "material", "query": "情绪调节 认知模式 行为改变"},
        {"source": "material", "query": "人际关系 依恋模式 家庭系统"},

        # User profile — 1 pull
        {"source": "profile", "query": ""},  # filled in by generate_plan
    ]


# ===== Public API =====

async def generate_plan(user_id: str, source_session: str = "") -> dict:
    """Run 1-5 round adaptive retrieval → produce a TreatmentPlan.

    Round 1: 14 broad parallel queries.
    Round 2-4: LLM reviews all accumulated data → decides output or more queries.
    Round 5: forced output.

    Args:
        user_id: User identifier.
        source_session: Optional session ID.

    Returns:
        dict with TreatmentPlanOutput fields.

    Raises:
        RuntimeError: All rounds exhausted without producing a plan.
    """
    MAX_ROUNDS = 5
    MAX_QUERIES = 30

    acc = DataAccumulator()
    decision_llm = plan_llm.with_structured_output(RoundDecision)

    # ---- Round 1: broad scan ----
    logger.info("PlanGenerator R1: broad scan for user=%s", user_id)
    r1_queries = _build_round1_queries()
    # Fill user_id into profile query
    for q in r1_queries:
        if q["source"] == "profile":
            q["query"] = user_id
    await acc.execute_queries(r1_queries)

    # ---- Rounds 2-5: adaptive ----
    for round_num in range(2, MAX_ROUNDS + 1):
        is_last_round = (round_num == MAX_ROUNDS)

        combined = acc.assemble(user_id, round_num)
        logger.info(
            "PlanGenerator R%d: %d queries so far, %d chars of data",
            round_num, acc.total_queries, len(combined),
        )

        system_prompt = ROUND_SYSTEM_PROMPT
        if is_last_round:
            system_prompt += FORCE_OUTPUT_PROMPT_EXTRA

        user_prompt = (
            f"Review ALL the data below and decide: output_plan or need_more_data?\n\n"
            f"DATA ({acc.total_queries} queries, {len(combined)} chars):\n\n{combined[:15000]}"
        )

        try:
            decision: RoundDecision = await decision_llm.ainvoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])
        except Exception as e:
            logger.error("PlanGenerator R%d LLM call failed: %s", round_num, e)
            if is_last_round:
                raise RuntimeError(f"Final round LLM call failed: {e}") from e
            continue  # retry next round

        if decision.action == "output_plan":
            logger.info(
                "PlanGenerator: plan ready at round %d — primary=%s stage=%s goals=%d",
                round_num, decision.primary_approach, decision.stage, len(decision.goals),
            )
            return _decision_to_output_dict(decision)

        # action == "need_more_data"
        fups = decision.follow_up_queries
        if not fups:
            logger.info("PlanGenerator R%d: LLM wants more data but specified no queries — forcing output", round_num)
            continue  # next round will be closer to forced output

        # Check query budget
        if acc.total_queries + len(fups) > MAX_QUERIES:
            logger.warning(
                "PlanGenerator R%d: would exceed query budget (%d + %d > %d), skipping",
                round_num, acc.total_queries, len(fups), MAX_QUERIES,
            )
            continue

        logger.info(
            "PlanGenerator R%d: LLM requests %d follow-up queries: %s",
            round_num, len(fups),
            ", ".join(q["query"][:50] for q in fups[:5]),
        )
        await acc.execute_queries(fups)

    # Should never reach here — round 5 is forced output
    raise RuntimeError("PlanGenerator exhausted all rounds without producing a plan")


def _decision_to_output_dict(d: RoundDecision) -> dict:
    """Extract plan fields from a RoundDecision into a plain dict."""
    return {
        "case_conceptualization": d.case_conceptualization,
        "primary_approach": d.primary_approach,
        "primary_approach_rationale": d.primary_approach_rationale,
        "secondary_approaches": d.secondary_approaches,
        "cautionary_approaches": d.cautionary_approaches,
        "stage": d.stage,
        "stage_rationale": d.stage_rationale,
        "goals": d.goals,
        "information_gaps": d.information_gaps,
        "verification_needed": d.verification_needed,
    }


def _minimal_plan(user_id: str) -> dict:
    """Placeholder when absolutely no data is available."""
    return {
        "case_conceptualization": "(No client data available; recommend initial assessment interview first.)",
        "primary_approach": "clinical-interviewing",
        "primary_approach_rationale": "No data — start with clinical interviewing to gather baseline information",
        "secondary_approaches": ["person-centered"],
        "cautionary_approaches": [],
        "stage": "engagement",
        "stage_rationale": "New client — begin with engagement phase",
        "goals": [
            {"description": "Complete initial assessment interview", "indicator": "Client can describe core concerns and help-seeking expectations", "timeline": "1-2 sessions", "priority": 1},
            {"description": "Build therapeutic alliance", "indicator": "Client reports feeling heard and accepted", "timeline": "1-2 sessions", "priority": 1},
        ],
        "information_gaps": ["No data available; all information must be gathered through assessment"],
        "verification_needed": ["Core presenting problems", "Change motivation and stage", "Risk factor assessment"],
    }


def plan_output_to_treatment_plan(
    output: dict,
    user_id: str,
    source_session: str = "",
) -> "TreatmentPlan":
    """Convert PlanGenerator output dict to TreatmentPlan model."""
    from treatment_plan import TreatmentPlan, TreatmentGoal

    now = datetime.now().isoformat()

    goals = []
    for i, g in enumerate(output.get("goals", [])):
        goals.append(TreatmentGoal(
            id=f"goal_{i+1:03d}",
            description=g.get("description", ""),
            indicator=g.get("indicator", ""),
            timeline=g.get("timeline", ""),
            priority=g.get("priority", 1),
            created_at=now,
            updated_at=now,
        ))

    plan = TreatmentPlan(
        user_id=user_id,
        version=1,
        created_at=now,
        updated_at=now,
        case_conceptualization=output.get("case_conceptualization", ""),
        primary_approach=output.get("primary_approach", "person-centered"),
        primary_approach_rationale=output.get("primary_approach_rationale", ""),
        secondary_approaches=output.get("secondary_approaches", []),
        cautionary_approaches=output.get("cautionary_approaches", []),
        stage=output.get("stage", "engagement"),
        goals=goals,
        version_history=[{
            "version": 1,
            "changed_fields": [
                "case_conceptualization", "primary_approach",
                "secondary_approaches", "cautionary_approaches",
                "stage", "goals",
            ],
            "timestamp": now,
            "source_session": source_session,
        }],
        source_sessions=[source_session] if source_session else [],
    )
    return plan
