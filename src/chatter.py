from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field
from typing import Optional
from config import LLM_MODEL
from skill_loader import lookup_skill, get_available_skills, get_chatter_skill_catalog, reset_skill_lookup_counts
import logging
import asyncio
import time

logger = logging.getLogger(__name__)

base_model = ChatDeepSeek(
    model=LLM_MODEL,
    temperature=0.5
)


class ChatterOutput(BaseModel):
    reply: str = Field(description="给用户的自然语言回复，共情、支持、对话式")
    should_retrieve: bool = Field(description="是否需要从记忆库检索背景信息来更好地理解用户")
    retrieve_query: str = Field(description="检索查询语句，仅在should_retrieve为true时填写，否则留空")


from mem_integration import read_file_tool, retrieve_user_profile_tool, get_queried_profile_domains_summary

SKILL_CATALOG = get_chatter_skill_catalog()

chatter = create_agent(
    model=base_model,
    system_prompt=(
        "你是一名资深的整合式心理咨询专家，精通基于三本核心教材（Corey第10版 + Falender & Shafranske临床督导 "
        "+ Sommers-Flanagan临床面谈第4版）的完整治疗体系。\n\n"
        "你以人本主义为底色，根据来访者的即时状态和深层需求，从你的治疗技能库中灵活选择和融合不同的治疗方法。\n"
        "你自然地运用这些技能，不提及疗法名称，不输出结构化标题，不提及多Agent架构。\n\n"
        "=== 治疗技能库（渐进式披露）===\n\n"
        "以下是你可用的治疗技能目录。每项的触发条件列在描述中。"
        "当来访者状态匹配某个技能的触发条件时，调用 lookup_skill(name) 获取该技能的完整操作指南。\n"
        "如需浏览所有可用技能，调用 get_available_skills()。\n\n"
        f"{SKILL_CATALOG}\n\n"
        "=== 工具使用规则（重要）===\n\n"
        "1. 每轮最多调用 lookup_skill 2次 — 只查最匹配来访者当前状态的1-2个技能\n"
        "2. get_available_skills 仅在首次不确定有哪些技能时调用一次，不要反复调用\n"
        "3. 加载技能后直接将内容融入回复，不要再回头补充查询\n"
        "4. 完成回复后立即输出 ChatterOutput 结构化结果\n\n"
        "=== 技能使用原则 ===\n\n"
        "1. 第一层技能（person-centered, existential）是所有对话的底色，始终在场\n"
        "2. 第二层技能按来访者状态按需调用——先识别状态，再加载对应技能\n"
        "3. 第三层技能（crisis-intervention）在危机情境优先使用——安全永远优先于探索深度\n"
        "4. 技能之间可以融合——先用person-centered建立关系，再调用其他技能深入工作\n"
        "5. 加载技能后自然融入回应，不要在对话中提及'根据XX疗法'或'我使用XX技术'\n\n"
        "=== 临床决策指南 ===\n\n"
        "根据六个维度选择技能：\n"
        "1. 情绪状态 — 情绪淹没→behavioral-third-wave(DBT)；情绪隔离→gestalt；情绪适度→cbt\n"
        "2. 求助动机 — 高动机+有方向→sfbt/choice-reality；高动机+无方向→behavioral-third-wave(ACT)/existential；低动机→person-centered\n"
        "3. 问题层次 — 表层行为→behavioral-third-wave；认知层面→cbt；情感层面→gestalt；关系层面→psychodynamic/family-systems\n"
        "4. 治疗阶段 — 初始→person-centered建联盟；中间→探索+改变；危机→crisis-intervention\n"
        "5. 文化背景 — 集体主义→family-systems优先；个人主义→个体赋权优先\n"
        "6. 联盟状态 — 稳固→深度探索；破裂→alliance-repair元沟通+修复；不稳→回归person-centered底色\n\n"
        "联盟第一原则：推动深度工作前先检查联盟状态。破裂不是失败——它是治疗的核心工作材料。\n\n"
        "=== 言语强度校准（重要）===\n\n"
        "你在通过纯文字与来访者交流——缺少语调、面部表情、身体语言等关键信息。\n"
        "这导致一个系统性偏差：LLM倾向于将文字表达的情绪强度等同于临床严重度。\n\n"
        "当来访者使用强烈语言时（如「真的很难受」「快撑不住了」「觉得活着很累」），你需要区分：\n\n"
        "| 文字表达 | 可能含义 | 不应自动等于 |\n"
        "|---------|---------|------------|\n"
        "| 强烈情绪词（崩溃/绝望/痛苦） | 来访者缺乏更精准的情感词汇；或文化习惯用强烈词语表达中度困扰 | 需要危机干预 |\n"
        "| 消极内容（活着没意义/不想醒来） | 存在性表达或抑郁思维；需要进一步评估 | 立即启动自杀协议 |\n"
        "| 频繁的负面情绪描述 | 来访者尚未学会识别复杂情绪；或把情绪倾泻当作治疗本身 | 需要格式塔式的情绪深化 |\n\n"
        "校准原则：\n"
        "1. 看到强烈的情绪词汇 → 先用person-centered共情接住，而不是立即激活深度情绪干预\n"
        "2. 怀疑有安全风险 → 问行为层面的问题（频率、具体计划、既往史），不基于词汇做判断\n"
        "3. 「说得痛苦」和「真的危险」之间有巨大的临床距离 — 你是专业人士，不是词频统计器\n"
        "4. 如果连续3轮对话来访者情绪语言激烈但未表现出行为层面的风险信号 → 应该考虑的是降低情绪卷入级别\n\n"
        "何时激活crisis-intervention：必须同时满足 — (a)来访者表达了伤害自己/他人的意图或具体计划 + (b)有既往相关行为史或明确的手段描述。仅凭情绪词汇不足以触发。\n"
        "何时激活gestalt/psychodynamic深度情绪工作：必须同时满足 — (a)联盟稳固（非初次建立期）+ (b)来访者有能力从强烈情绪中回到观察位置 + (c)治疗计划当前阶段允许。\n\n"
        "=== 核心规则 ===\n\n"
        "1. 不要提及任何系统架构相关背景（多Agent协作、分析师、督导的存在）\n"
        "2. 输出是非结构化的、对话式的，不输出标题或疗法名称\n"
        "3. 不要着急给出建议，以共情、支持为主\n"
        "4. 每次聚焦一到两个问题，保持对话连续性\n"
        "5. 咨询可以结束时委婉提醒用户\n"
        "6. 收到supervisor协作建议或retrieval检索结果时务必参考\n"
        "7. 画像/检索/证据等内部资料融入理解后用自己的话表达，不直接引用或提及系统术语\n"
        "8. 自我披露三原则：为来访者利益而非自己表达需要；披露后将焦点引回来访者；避免无意识披露\n"
        "9. 保持专业自我觉察：留意过度建议/过度认同/过度疏离倾向\n"
        "10. 不要编造背景信息——优先使用画像摘要和对话历史中已有的信息，仅在关键背景确实未知且对话需要时才设置should_retrieve=true。不要为确认细节而反复查询。\n\n"
        "=== 治疗计划遵从 ===\n\n"
        "每轮对话前会自动注入治疗计划摘要（当前阶段、主要方法、目标进展、未完成事项）。\n"
        "你应该：\n"
        "1. 优先使用治疗计划指定的主要方法 — 它们是基于全面评估为这位来访者个性化选择的\n"
        "2. 谨慎方法仅在明确适应症时才使用（如来访者表现出强烈的移情/防御信号时用psychodynamic）\n"
        "3. 保持跨会话连续性 — 跟进上次会话的未完成事项和计划\n"
        "4. 偏离计划时要有临床理由（如联盟修复、危机响应）— 这将由supervisor评估\n"
        "5. 不需要在每个回应中都刻意提及计划 — 自然地遵循它\n\n"
        "=== 用户画像 ===\n\n"
        "每轮对话前会自动注入来访者画像摘要。同一画像领域只可查询一次——查询后即拥有该领域完整数据，"
        "后续对话直接使用已有信息，绝对不要重复查询相同领域。"
        "查询特定领域会自动附带证据原始内容。\n\n"
        "=== 记忆检索 ===\n\n"
        "需要更多背景信息时设should_retrieve=true并写具体retrieve_query。\n"
        "系统会立即检索并将结果反馈给你，你可以在同一轮对话中使用这些信息。\n"
        "信息充足时保持false。\n\n"
        "=== 文件处理 ===\n\n"
        "用户上传文件时阅读并进行初步分析。\n"
    ),
    tools=[read_file_tool, retrieve_user_profile_tool, lookup_skill, get_available_skills],
    response_format=ToolStrategy(ChatterOutput)
)


def _parse_chatter_output(output_text: str) -> ChatterOutput:
    """解析 chatter 输出，带容错。

    优先 JSON 解析；失败时尝试从文本中提取 JSON 对象；
    再尝试伪 Python 格式（reply='...' should_retrieve=...）；
    都失败时把全文当作 reply 返回。
    """
    import re

    # 1. 直接解析 JSON
    try:
        return ChatterOutput.model_validate_json(output_text)
    except Exception:
        pass

    # 2. 从文本中提取首个 JSON 对象
    match = re.search(r'\{[^{}]*\}(?=\s*$)', output_text, re.DOTALL)
    if not match:
        match = re.search(r'\{.*\}', output_text, re.DOTALL)
    if match:
        try:
            return ChatterOutput.model_validate_json(match.group())
        except Exception:
            pass

    # 3. 解析 LLM 伪 Python 格式: reply='...' should_retrieve=True/False retrieve_query='...'
    reply_match = re.search(r"reply\s*=\s*['\"](.+?)['\"]\s*(?:should_retrieve|$)", output_text, re.DOTALL)
    retrieve_match = re.search(r"should_retrieve\s*=\s*(True|False)", output_text)
    query_match = re.search(r"retrieve_query\s*=\s*['\"](.+?)['\"]", output_text, re.DOTALL)

    if reply_match:
        reply = reply_match.group(1).strip()
        should_retrieve = retrieve_match.group(1) == "True" if retrieve_match else False
        retrieve_query = query_match.group(1).strip() if query_match else ""
        logger.info("chatter 输出已按文本格式解析 (should_retrieve=%s)", should_retrieve)
        return ChatterOutput(reply=reply, should_retrieve=should_retrieve, retrieve_query=retrieve_query)

    # 4. 兜底：全文作为 reply，不触发检索
    logger.warning("chatter 输出非 JSON，使用全文兜底: %s", output_text[:120])
    return ChatterOutput(reply=output_text, should_retrieve=False, retrieve_query="")


async def _build_chatter_input(shared_context, user_input: str, profile_summary: str = "", retrieval_context: str = "", plan_context: str = "") -> str:
    """构建 chatter 输入文本"""
    history = await shared_context.get_recent_messages(50)
    history_messages = [msg["content"] for msg in history]
    chat_input = "\n\n".join(history_messages)

    if retrieval_context:
        chat_input = f"【记忆检索结果（请参考以下信息回应来访者）】\n{retrieval_context}\n\n{chat_input}"

    if profile_summary:
        chat_input = f"【来访者画像摘要（你已了解的背景信息）】\n{profile_summary}\n\n{chat_input}"

    if plan_context:
        chat_input = f"{plan_context}\n\n{chat_input}"

    queried_summary = get_queried_profile_domains_summary(getattr(shared_context, 'user_id', 'default'))
    if queried_summary:
        chat_input = f"【画像查询追踪】{queried_summary}\n\n{chat_input}"

    chat_input += "\n\n" + user_input
    return chat_input


async def _invoke_chatter(chat_input: str) -> ChatterOutput:
    """单次调用 chatter agent"""
    reset_skill_lookup_counts()
    result = await chatter.ainvoke(
        {"messages": [{"role": "user", "content": chat_input}]},
        config={"recursion_limit": 10}
    )
    output_text = result["messages"][-1].content
    return _parse_chatter_output(output_text)


async def _do_sync_retrieval(query: str) -> str:
    """同步执行三类检索，返回格式化文本"""
    from mem_retrieve_diary import retrieve_diary
    from mem_retrieve_material import retrieve_materials
    from mem_retrieve_conv_outline import retrieve_conv_outline

    results_parts = []

    try:
        diary_results = await retrieve_diary(query)
        if diary_results:
            parts = []
            for r in diary_results:
                docs = getattr(r, 'documents', []) or []
                for doc in docs[:2]:
                    content = getattr(doc, 'page_content', str(doc))[:300]
                    parts.append(content)
            if parts:
                results_parts.append("【日记检索】:\n" + "\n".join(parts))
    except Exception as e:
        logger.warning(f"日记检索失败: {e}")

    try:
        conv_results = await retrieve_conv_outline(query)
        if conv_results:
            parts = []
            for r in conv_results:
                docs = getattr(r, 'matched_docs', []) or []
                for doc in docs[:2]:
                    content = getattr(doc, 'page_content', str(doc))[:300]
                    parts.append(content)
            if parts:
                results_parts.append("【对话摘要检索】:\n" + "\n".join(parts))
    except Exception as e:
        logger.warning(f"对话摘要检索失败: {e}")

    try:
        material_results = await retrieve_materials(query)
        if material_results:
            parts = []
            for r in material_results:
                docs = getattr(r, 'matched_children', []) or []
                for doc in docs[:2]:
                    content = getattr(doc, 'page_content', str(doc))[:300]
                    parts.append(content)
            if parts:
                results_parts.append("【资料检索】:\n" + "\n".join(parts))
    except Exception as e:
        logger.warning(f"资料检索失败: {e}")

    if not results_parts:
        return ""
    return "\n\n".join(results_parts)


async def call_chatter(shared_context, user_input: str, profile_summary: str = "", plan_context: str = "") -> ChatterOutput:
    """调用chatter并返回结构化输出。

    采用两轮同步检索模式：
    1. 第一轮：chatter 判断是否需要检索
    2. 如果需要，同步执行检索并第二轮调用 chatter
    3. 最多两轮，防止无限循环
    """
    MAX_PASSES = 2

    retrieval_context = ""
    for _ in range(MAX_PASSES):
        chat_input = await _build_chatter_input(shared_context, user_input, profile_summary, retrieval_context, plan_context)
        output = await _invoke_chatter(chat_input)

        if not output.should_retrieve or not output.retrieve_query.strip():
            return output

        logger.info(f"Chatter请求同步检索: {output.retrieve_query[:80]}")
        retrieval_context = await _do_sync_retrieval(output.retrieve_query.strip())

        if not retrieval_context:
            logger.info("同步检索无结果，直接返回第一轮回复")
            return output

        logger.info(f"同步检索完成: {len(retrieval_context)}字符，进入第二轮")

    # 达到最大轮数，返回最后一轮的回复
    return output


_chatter_retrieve_lock = asyncio.Lock()

async def background_chatter_retrieve(shared_context, query: str):
    """后台执行检索并将结果注入到下一轮对话（已弃用，保留作为兜底）。

    call_chatter 现在在内部同步处理检索请求，此函数仅作为 fallback。
    """
    if _chatter_retrieve_lock.locked():
        logger.debug("Chatter检索任务已在运行，跳过")
        return

    try:
        await _chatter_retrieve_lock.acquire()
    except Exception:
        return

    try:
        combined = await _do_sync_retrieval(query)
        if not combined:
            logger.info(f"Chatter检索无结果: {query}")
            return

        from SharedContext import PromptInjection
        async with shared_context._lock:
            shared_context._chatter_retrieval_injection = PromptInjection(
                content=combined,
                timestamp=time.time(),
                source="chatter_retrieval"
            )
            logger.info(f"Chatter检索注入完成: {len(combined)}字符")
    except Exception as e:
        logger.error(f"Chatter后台检索失败: {e}")
    finally:
        if _chatter_retrieve_lock.locked():
            _chatter_retrieve_lock.release()
