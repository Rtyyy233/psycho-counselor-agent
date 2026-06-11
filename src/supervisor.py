from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field
from langchain.agents.structured_output import ToolStrategy
from typing import Optional
from SharedContext import SharedContext
from config import LLM_MODEL
from skill_loader import lookup_skill, get_available_skills, get_supervisor_skill_catalog, reset_skill_lookup_counts
import logging
import asyncio
import json

logger = logging.getLogger(__name__)
_supervisor_lock = asyncio.Lock()  # prevent concurrent supervisor runs

base_model = ChatDeepSeek(
    model=LLM_MODEL,
    temperature=0.2
)


class SupervisionOutput(BaseModel):
    should_inject: bool = Field(description="是否有分析结果需要注入到对话中")
    injection_content: str = Field(description="注入内容，包含分析和方向建议")
    should_retrieve: bool = Field(description="是否需要主动触发检索")
    retrieve_query: str = Field(description="检索查询，仅在should_retrieve为true时填写")
    should_revise_plan: bool = Field(default=False, description="是否建议在会话结束时修订治疗计划")
    plan_feedback: str = Field(default="", description="对治疗计划一致性的评估反馈")


from mem_integration import (
    retrieve_conv_outline_tool, retrieve_diary_tool, retrieve_materials_tool,
    read_file_tool, store_diary_tool, store_material_tool
)

SKILL_CATALOG = get_supervisor_skill_catalog()

supervisor = create_agent(
    model=base_model,
    system_prompt=(
        "你是一名资深临床督导，基于三本核心教材的理论框架（Corey第10版 + Falender & Shafranske临床督导 "
        "+ Sommers-Flanagan临床面谈第4版），静默监听心理咨询对话，提供专业的分析、评估与督导建议。\n\n"
        "=== 督导技能库（渐进式披露）===\n\n"
        "以下是你可用的督导监测技能目录。每项的触发条件列在描述中。"
        "当对话状态匹配某个技能的触发条件时，调用 lookup_skill(name) 获取该技能的完整操作指南。\n"
        "如需浏览所有可用技能，调用 get_available_skills()。\n\n"
        f"{SKILL_CATALOG}\n\n"
        "=== 分层监测框架 ===\n\n"
        "第一层：基础监测 — 每轮自动运行（alliance-monitoring / process-quality）\n"
        "第二层：按需触发 — 特定信号出现时调用（countertransference / pattern-recognition / 信息检索）\n"
        "第三层：危机响应 — 高风险情境立即介入（crisis-detection）\n\n"
        "=== 工具使用规则（重要）===\n\n"
        "1. 每轮最多调用 lookup_skill 2次 — 只查最匹配当前对话信号的1-2个技能，不要遍历所有技能\n"
        "2. get_available_skills 仅在首次不确定有哪些技能时调用一次，不要反复调用\n"
        "3. 调用检索工具之前先读完已匹配技能的完整内容，避免边查边搜造成循环\n"
        "4. 完成分析后立即输出结构化结果，不要再回头补充查询\n\n"
        "=== 督导决策指南 ===\n\n"
        "何时注入督导意见？根据三层级判断：\n"
        "1. 第三层信号 → 立即注入（高风险）— 自杀意念、严重联盟破裂、强烈反移情\n"
        "2. 第二层信号 + 持续2轮以上 → 注入（中优先级）— 反复出现的模式、技术偏离、未跟进的历史建议\n"
        "3. 第一层轻度偏离 → 静默观察或仅做简短提醒（低优先级）— 偶尔的节奏问题、轻微的情绪未充分探索\n\n"
        "注入原则：\n"
        "- 一次注入聚焦于一个核心问题，避免信息过载\n"
        "- 使用具体、可操作的语言，避免含糊的理论讨论\n"
        "- 引用相关理论来源时自然融入，不刻意提及教材名称\n"
        "- 当你认为有重要的分析或督导意见需要注入时，将should_inject设为true\n"
        "- 不要反复调用工具处理已经处理过的内容\n"
        "- 检索结果中出现的文件路径或文件名仅供参考——这些文件可能已不可访问。不要自行构造或猜测文件路径去调用read_file_tool。需要内容时使用检索工具，不要试图直接读取原始文件。\n"
        "- 当你认为重要的督导意见需要在本次会话结束后修订治疗计划时，将should_revise_plan设为true并填写plan_feedback\n\n"
        "=== 治疗计划监测（新增）===\n\n"
        "每轮对话前会自动注入来访者的治疗计划摘要（当前阶段、主要/辅助/谨慎方法、目标进展、未完成事项）。"
        "你作为督导，需要额外评估：\n\n"
        "1. **方法一致性**：Chatter的方法选择是否与治疗计划一致？\n"
        "2. **偏离评估**：如果偏离，是否有临床理由（如联盟修复、危机响应）？\n"
        "3. **进展评估**：来访者是否在向治疗目标推进？有无“虚假改善”的迹象？\n\n"
        "偏离等级与应对：\n"
        "- 正常灵活性：临时切换person-centered修复联盟、危机时切换crisis-intervention → 允许，不做特别标注\n"
        "- 需要提醒：连续3+轮未使用计划指定的主要方法，且无合理临床理由 → 注入提醒，should_inject=true\n"
        "- 需要重新评估：来访者对计划中的方法持续反应不佳（防御、回避、情绪恶化）→ 标记 should_revise_plan=true\n\n"
        "plan_feedback 应包含：\n"
        "- 本次会话中观察到的方法使用情况\n"
        "- 哪些方法有效/无效（引用对话证据）\n"
        "- 阶段切换的建议（如果有）\n"
        "- 下次会话的方法调整建议\n"
        "- 如果检测到言语强度错判，标注「over-activation: [技能名]」，格式如 \"over-activation: crisis-intervention — 来访者表达了强烈情绪但无行为层面风险信号\"\n\n"
        "=== 言语强度错判监测（新增）===\n\n"
        "你是通过纯文字监听对话的——与Chatter一样，你也缺少语调、表情等现实信号。\n"
        "Chatter可能因来访者使用了强烈情绪词汇而过度激活crisis-intervention或深度情绪方法（gestalt/psychodynamic）。\n"
        "你的任务是作为第二层校验：\n\n"
        "当Chatter激活了以下方法时，进行额外的行为证据检验：\n"
        "- crisis-intervention → 必须有：(a)明确的伤害意图或具体计划 + (b)既往行为史或明确手段描述\n"
        "- gestalt → 必须有：(a)联盟已建立且稳固 + (b)来访者显示出区分「体验情绪」和「被情绪淹没」的能力\n"
        "- psychodynamic → 必须有：(a)联盟稳固 + (b)有明显的移情/防御机制模式（非单次事件）\n\n"
        "如果你判断是言语强度误判（词汇驱动而非行为证据驱动）：\n"
        "1. 将 should_revise_plan=true，plan_feedback 中标注 over-activation\n"
        "2. injection_content 中提醒Chatter降级：「来访者的情绪语言可能比实际状态强烈——先观察行为层面再决定是否升级干预」\n"
        "3. 如果同一方法在一轮对话中被标记2次以上 → 建议将该方法纳入治疗计划的cautionary_approaches\n"
        "4. 连续错判3次以上 → 标记为紧急，立即注入纠正并建议中止该方法\n"
    ),
    tools=[
        retrieve_conv_outline_tool,
        retrieve_diary_tool,
        retrieve_materials_tool,
        read_file_tool,
        store_diary_tool,
        store_material_tool,
        lookup_skill,
        get_available_skills,
    ],
    response_format=ToolStrategy(SupervisionOutput)
)


async def call_supervisor(SharedContext: SharedContext, plan_context: str = "", session_turn_count: int = 0):
    if _supervisor_lock.locked():
        logger.debug("Supervisor任务已在运行，跳过")
        return

    try:
        await _supervisor_lock.acquire()
    except Exception as e:
        logger.debug(f"获取Supervisor锁失败: {e}")
        return

    try:
        await SharedContext.supervisor_trigger.wait()

        messages = None
        async with SharedContext._lock:
            SharedContext.supervisor_spare = False
            messages = SharedContext._messages[-10:] if SharedContext._messages else []

        if not messages:
            logger.debug("没有消息可处理")
            return

        formatted_history = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in messages
        ])

        # Prepend treatment plan context for monitoring
        if plan_context:
            formatted_history = f"{plan_context}\n\n{formatted_history}"

        # Auto-trigger pattern-recognition for cross-session continuity in first 3 turns
        if session_turn_count <= 3:
            formatted_history = (
                "【跨会话连续性检查】这是本次会话的前3轮之一。"
                "请运行 pattern-recognition 技能，检查：\n"
                "1. 历史督导建议是否被跟进？\n"
                "2. 来访者的叙事是否与治疗计划中的个案例概念化一致？\n"
                "3. 治疗方向是否保持连续性？\n\n"
            ) + formatted_history

        reset_skill_lookup_counts()
        result = await supervisor.ainvoke(
            {"messages": [{"role": "user", "content": formatted_history}]},
            config={"recursion_limit": 12}
        )

        if not result or "messages" not in result or not result["messages"]:
            logger.warning("Supervisor返回无效响应格式")
            return

        # Extract structured output from ToolStrategy result
        last_msg = result["messages"][-1]
        should_inject = False
        injection_content = ""
        should_retrieve = False
        retrieve_query = ""
        should_revise_plan = False
        plan_feedback = ""

        # ToolStrategy may output via tool_calls or JSON content
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            args = last_msg.tool_calls[0].get("args", {})
            should_inject = args.get("should_inject", False)
            injection_content = args.get("injection_content", "")
            should_retrieve = args.get("should_retrieve", False)
            retrieve_query = args.get("retrieve_query", "")
            should_revise_plan = args.get("should_revise_plan", False)
            plan_feedback = args.get("plan_feedback", "")
        elif hasattr(last_msg, "content") and last_msg.content:
            try:
                parsed = json.loads(last_msg.content) if isinstance(last_msg.content, str) else {}
                should_inject = parsed.get("should_inject", False)
                injection_content = parsed.get("injection_content", "")
                should_retrieve = parsed.get("should_retrieve", False)
                retrieve_query = parsed.get("retrieve_query", "")
                should_revise_plan = parsed.get("should_revise_plan", False)
                plan_feedback = parsed.get("plan_feedback", "")
            except json.JSONDecodeError:
                injection_content = last_msg.content
                if len(injection_content.strip()) >= 5:
                    should_inject = True

        # Apply injection if needed
        if should_inject and injection_content and len(injection_content.strip()) >= 5:
            async with SharedContext._lock:
                if SharedContext._supervisor_injection is None:
                    from SharedContext import PromptInjection
                    import time
                    SharedContext._supervisor_injection = PromptInjection(
                        content=injection_content,
                        timestamp=time.time(),
                        source="supervisor"
                    )
                else:
                    SharedContext._supervisor_injection.content = injection_content

        # Store plan feedback for end-of-session processing
        if should_revise_plan and plan_feedback:
            async with SharedContext._lock:
                SharedContext._plan_feedback = plan_feedback
            await SharedContext.increment_revise_plan_count()
            logger.info("Supervisor标记需要修订治疗计划 (第%d次): %s",
                        SharedContext._revise_plan_count, plan_feedback[:100])

        # Log retrieval request (delegated — agent can call tools directly during invoke)
        if should_retrieve and retrieve_query:
            logger.info(f"Supervisor请求主动检索: {retrieve_query}")

    except asyncio.CancelledError:
        logger.warning("Supervisor任务被取消（可能是超时或服务关闭）")
    except Exception as e:
        logger.error(f"Supervisor调用失败: {e}")
    finally:
        async with SharedContext._lock:
            SharedContext.supervisor_spare = True
        SharedContext.supervisor_trigger.clear()
        if _supervisor_lock.locked():
            _supervisor_lock.release()

    return
