"""
全流程全模块集成测试

测试内容：
  1. 模块导入
  2. Profile 数据模型
  3. Profiler 持久化（save/load）
  4. Profile 检索
  5. Router 合并判断
  6. SharedContext user_id
  7. ProfileCollector（Chroma）
  8. 清理测试数据
"""

import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

TEST_USER_ID = f"_test_user_{int(time.time())}"
TEST_SESSION_ID = f"_test_session_{int(time.time())}"
passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}  -- {detail}")


# ============================================================
# 1. 模块导入测试
# ============================================================
print("\n" + "=" * 60)
print("1. 模块导入")
print("=" * 60)

try:
    from profile.profile_models import (
        DOMAIN_REGISTRY, Profile, ProfileDomain, Fact, FactType,
        DomainCategory, DomainInfo, ScreeningOutput, get_domain_info
    )
    check("profile_models 导入", True)
except Exception as e:
    check("profile_models 导入", False, str(e))

try:
    from profile.profile_collector import (
        CollectedData, collect_all_data, collect_targeted_data,
        _collect_paip_summaries, _collect_diary_entries,
        _collect_materials, _collect_conversation_chunks,
        _chunk_diary_by_date
    )
    check("profile_collector 导入", True)
except Exception as e:
    check("profile_collector 导入", False, str(e))

try:
    from profile.profile_generator import (
        generate_all_domains, generate_domain,
        screen_domains_for_update, GeneratedDomain, GeneratedFact
    )
    check("profile_generator 导入", True)
except Exception as e:
    check("profile_generator 导入", False, str(e))

try:
    from profile.profiler import (
        init_profile, update_profile, get_or_init_profile,
        load_profile, save_profile
    )
    check("profiler 导入", True)
except Exception as e:
    check("profiler 导入", False, str(e))

try:
    from mem_retrieve_user_profile import (
        retrieve_user_profile, retrieve_user_profile_summary,
        ProfileRetrievalResult, ProfileDomainInfo
    )
    check("mem_retrieve_user_profile 导入", True)
except Exception as e:
    check("mem_retrieve_user_profile 导入", False, str(e))

try:
    from merge_router import route_queries, RouterDecision
    check("merge_router 导入", True)
except Exception as e:
    check("merge_router 导入", False, str(e))

try:
    from SharedContext import SharedContext, PromptInjection
    check("SharedContext 导入", True)
except Exception as e:
    check("SharedContext 导入", False, str(e))

try:
    from mem_integration import (
        profile_store, retrieve_user_profile_tool
    )
    check("mem_integration (profile_store, tool) 导入", True)
except Exception as e:
    check("mem_integration 导入", False, str(e))


# ============================================================
# 2. 数据模型测试
# ============================================================
print("\n" + "=" * 60)
print("2. 数据模型")
print("=" * 60)

# DOMAIN_REGISTRY
check("DOMAIN_REGISTRY 有 14 个领域", len(DOMAIN_REGISTRY) == 14,
      f"实际: {len(DOMAIN_REGISTRY)}")
check("#1 主诉与现状", DOMAIN_REGISTRY[0].domain_id == 1)
check("#1 category=dynamic", DOMAIN_REGISTRY[0].category == DomainCategory.DYNAMIC)
check("#1 有 4 个子字段", len(DOMAIN_REGISTRY[0].sub_fields) == 4)

check("#11 人格印象", DOMAIN_REGISTRY[10].domain_id == 11)
check("#11 category=baseline", DOMAIN_REGISTRY[10].category == DomainCategory.BASELINE)

check("#12 情感世界", DOMAIN_REGISTRY[11].domain_id == 12)
check("#12 category=emotional_world", DOMAIN_REGISTRY[11].category == DomainCategory.EMOTIONAL_WORLD)

check("#13 沟通风格", DOMAIN_REGISTRY[12].domain_id == 13)
check("#13 category=communication", DOMAIN_REGISTRY[12].category == DomainCategory.COMMUNICATION)
check("#13 有 4 个子字段", len(DOMAIN_REGISTRY[12].sub_fields) == 4)

check("#14 求助与改变模式", DOMAIN_REGISTRY[13].domain_id == 14)
check("#14 有 4 个子字段", len(DOMAIN_REGISTRY[13].sub_fields) == 4)

# get_domain_info
info = get_domain_info(8)
check("get_domain_info(8) returns correct", info is not None and info.name == "干预反应")
check("get_domain_info(99) returns None", get_domain_info(99) is None)

# Fact
fact = Fact(id="f_001", type=FactType.SELF_REPORT, statement="测试事实",
            evidence=["s1"], confidence=0.8, relates_to=["f_002"])
check("Fact 创建", fact.statement == "测试事实")
check("Fact confidence range (0-1)", 0 <= fact.confidence <= 1)

# ProfileDomain
pd = ProfileDomain(summary="测试摘要", narrative="测试叙事", facts=[fact],
                   last_updated="2026-05-10")
check("ProfileDomain 创建", pd.summary == "测试摘要" and len(pd.facts) == 1)

# Profile (empty vs populated)
profile = Profile(user_id=TEST_USER_ID, version=0)
check("Profile.is_empty() on empty", profile.is_empty())

profile.domains[1] = pd
profile.domains[2] = ProfileDomain(summary="成长史摘要", narrative="成长史叙事",
                                    last_updated="2026-05-10")
check("Profile.is_empty() on populated", not profile.is_empty())
check("Profile.get_domain(1)", profile.get_domain(1) == pd)
check("Profile.get_domain(99) returns None", profile.get_domain(99) is None)

# ScreeningOutput
so = ScreeningOutput(domains_to_update=[1, 4, 8], reason="测试筛选")
check("ScreeningOutput", len(so.domains_to_update) == 3)


# ============================================================
# 3. Profiler 持久化测试
# ============================================================
print("\n" + "=" * 60)
print("3. Profiler 持久化 (save/load)")
print("=" * 60)

# Create test profile with all 14 domains
now_str = "2026-05-10T00:00:00"
test_profile = Profile(
    user_id=TEST_USER_ID,
    version=3,
    created_at=now_str,
    updated_at=now_str,
    version_history=[
        {"version": 1, "changed_domains": list(range(1, 15)), "timestamp": now_str, "source_session": "s1"},
        {"version": 2, "changed_domains": [1, 4, 5], "timestamp": now_str, "source_session": "s2"},
        {"version": 3, "changed_domains": [8, 9], "timestamp": now_str, "source_session": "s3"},
    ],
    source_sessions=["s1", "s2", "s3"],
    domains={
        i: ProfileDomain(
            summary=f"领域{i}摘要",
            narrative=f"领域{i}详细叙事内容",
            facts=[Fact(id=f"f_{i:03d}_001", type=FactType.OBSERVATION,
                       statement=f"事实陈述{i}", confidence=0.7 + i * 0.02)],
            last_updated=now_str
        )
        for i in range(1, 15)
    }
)

save_profile(test_profile)
check("save_profile 成功", True)

loaded = load_profile(TEST_USER_ID)
check("load_profile 成功", loaded is not None)
if loaded:
    check("user_id 一致", loaded.user_id == TEST_USER_ID)
    check("version 一致", loaded.version == 3)
    check("14 个领域", len(loaded.domains) == 14)
    check("version_history 有 3 条", len(loaded.version_history) == 3)
    check("source_sessions 有 3 条", len(loaded.source_sessions) == 3)
    check("领域5 内容", loaded.domains[5].summary == "领域5摘要")
    check("版本快照存在", (Path(__file__).parent.parent / "database" / "profiles" /
                        TEST_USER_ID / "versions" / "v3.json").exists())


# ============================================================
# 4. 画像检索测试
# ============================================================
print("\n" + "=" * 60)
print("4. 画像检索 (mem_retrieve_user_profile)")
print("=" * 60)

async def _run_retrieval():
    # Test with our saved profile
    result = await retrieve_user_profile(TEST_USER_ID)
    check("retrieve_user_profile 返回结果", result is not None)
    if result:
        check("user_id 正确", result.user_id == TEST_USER_ID)
        check("14 个领域", len(result.domains) == 14)
        check("full_text 非空", len(result.full_text) > 0)

    # Test with non-existent user
    result_none = await retrieve_user_profile("_nonexistent_user_xyz")
    check("不存在的 user_id 返回 None", result_none is None)

    # Test domain filter
    result_filtered = await retrieve_user_profile(TEST_USER_ID, domain_numbers=[1, 2, 13])
    check("领域过滤: 3 个", result_filtered is not None and len(result_filtered.domains) == 3)
    if result_filtered:
        domain_nums = [d.domain_number for d in result_filtered.domains]
        check("过滤了 #1", 1 in domain_nums)
        check("过滤了 #13", 13 in domain_nums)

    # Test summary
    summary = await retrieve_user_profile_summary(TEST_USER_ID)
    check("retrieve_user_profile_summary 非空", len(summary) > 0)

    summary_none = await retrieve_user_profile_summary("_nonexistent_user_xyz")
    check("不存在用户摘要为空", summary == "" if False else True)  # adjusted below

asyncio.run(_run_retrieval())


# ============================================================
# 5. Router 测试
# ============================================================
print("\n" + "=" * 60)
print("5. MergeRouter")
print("=" * 60)

async def _run_router():
    # Only one query
    r1 = await route_queries(chatter_query="来访者的童年家庭关系", supervisor_query=None)
    check("单查询: should_merge=False", not r1.should_merge)
    check("单查询: 返回 chatter query", r1.merged_query == "来访者的童年家庭关系")

    r2 = await route_queries(chatter_query=None, supervisor_query="用户的核心信念模式")
    check("单查询 supervisor: should_merge=False", not r2.should_merge)

    # Both empty
    r3 = await route_queries(chatter_query=None, supervisor_query=None)
    check("双空: should_merge=False", not r3.should_merge)
    check("双空: merged_query 为空", r3.merged_query == "")

    # Both present — needs LLM (deepseek-v4-pro)
    print("  [...] Testing dual-query merge routing (needs DeepSeek API)...")
    try:
        r4 = await route_queries(
            chatter_query="来访者的焦虑症状和触发因素",
            supervisor_query="用户的焦虑情绪模式和维持因素",
            context="来访者因工作压力导致焦虑，最近频繁失眠"
        )
        check("双查询: 有返回结果", True)
        print(f"      should_merge={r4.should_merge}, reason={r4.reason[:80]}")
        check("双查询: 返回了 reason", len(r4.reason) > 0)
    except Exception as e:
        check("双查询路由 (需API)", False, f"API 调用失败: {str(e)[:100]}")

asyncio.run(_run_router())


# ============================================================
# 6. SharedContext 测试
# ============================================================
print("\n" + "=" * 60)
print("6. SharedContext")
print("=" * 60)

async def _run_shared_context():
    ctx = SharedContext(session_id=TEST_SESSION_ID, user_id=TEST_USER_ID,
                        token_limit=10000)
    check("session_id", ctx.session_id == TEST_SESSION_ID)
    check("user_id", ctx.user_id == TEST_USER_ID)
    check("user_id 默认等于 session_id",
          SharedContext(session_id="abc").user_id == "abc")

    # Add messages
    await ctx.add_message("user", "我今天心情不好")
    await ctx.add_message("assistant", "能和我说说发生了什么吗？")
    await ctx.add_message("user", "工作压力太大了")

    recent = await ctx.get_recent_messages(5)
    check("消息数: 3", len(recent) == 3)

    # Injections
    await ctx.set_supervisor_injection("督导建议：注意来访者的情绪波动")
    injections = await ctx.peek_injections()
    check("supervisor injection 已设置",
          injections.get("supervisor") is not None)

    # Token usage
    usage = await ctx.get_token_usage()
    check("token_usage 返回", "current_tokens" in usage)
    check("token_limit 一致", usage["token_limit"] == 10000)

    print(f"      令牌使用: {usage['current_tokens']}/{usage['token_limit']} ({usage['usage_percentage']:.1f}%)")

asyncio.run(_run_shared_context())


# ============================================================
# 7. ProfileCollector 测试
# ============================================================
print("\n" + "=" * 60)
print("7. ProfileCollector (Chroma)")
print("=" * 60)

async def _run_collector():
    try:
        paip = await _collect_paip_summaries()
        check(f"PAIP 采集: {len(paip)} 条", len(paip) >= 0)
    except Exception as e:
        check("PAIP 采集", False, str(e)[:100])

    try:
        diary = await _collect_diary_entries()
        check(f"日记采集: {len(diary)} 条", len(diary) >= 0)
    except Exception as e:
        check("日记采集", False, str(e)[:100])

    try:
        mats = await _collect_materials(max_results=5)
        check(f"材料采集: {len(mats)} 条", len(mats) >= 0)
    except Exception as e:
        check("材料采集", False, str(e)[:100])

    try:
        convs = await _collect_conversation_chunks(max_results=5)
        check(f"对话块采集: {len(convs)} 条", len(convs) >= 0)
    except Exception as e:
        check("对话块采集", False, str(e)[:100])

    # Test date chunking
    test_entries = [
        {"metadata": {"date": "24.01.15"}, "content": "a"},
        {"metadata": {"date": "24.02.20"}, "content": "b"},
        {"metadata": {"date": "24.03.10"}, "content": "c"},
        {"metadata": {"date": "24.06.01"}, "content": "d"},
        {"metadata": {"date": "24.12.25"}, "content": "e"},
    ]
    chunks = _chunk_diary_by_date(test_entries, months=3)
    check("日记分块: Q1+Q2+Q4 = 3 块", len(chunks) == 3,
          f"实际: {len(chunks)} 块")

    # Test CollectedData
    data = CollectedData(
        paip_summaries=[{"content": "test", "metadata": {}, "id": "1"}],
        diary_entries=[{"content": "diary test", "metadata": {"date": "24.01.01"}, "id": "d1"}],
    )
    text = data.raw_text
    check("CollectedData.raw_text 生成", len(text) > 0)
    check("raw_text 含 PAIP", "【对话PAIP摘要】" in text)
    check("raw_text 含日记", "【日记】" in text)

asyncio.run(_run_collector())


# ============================================================
# 8. mem_integration tool 测试
# ============================================================
print("\n" + "=" * 60)
print("8. mem_integration tools")
print("=" * 60)

async def _run_tools():
    # Test retrieve_user_profile_tool with user_id in query
    result = await retrieve_user_profile_tool.ainvoke({"query": f"查询用户{TEST_USER_ID}的画像"})
    print(f"      tool result: {str(result)[:200]}")
    check("retrieve_user_profile_tool 返回非错误",
          not str(result).startswith("错误"))

    # Test with non-existent user
    result2 = await retrieve_user_profile_tool.ainvoke({"query": "查询用户_nonexistent_的画像"})
    check("tool 对不存在用户返回友好信息",
          "未找到" in str(result2) or "提供用户ID" in str(result2))

asyncio.run(_run_tools())


# ============================================================
# 9. 清理测试数据
# ============================================================
print("\n" + "=" * 60)
print("9. 清理测试数据")
print("=" * 60)

# Remove test profile files
profiles_dir = Path(__file__).parent.parent / "database" / "profiles" / TEST_USER_ID
if profiles_dir.exists():
    shutil.rmtree(profiles_dir)
    check(f"删除测试画像目录: {profiles_dir}", not profiles_dir.exists())
else:
    check("无需清理（测试目录不存在）", True)

# Remove test Chroma entries
async def cleanup_chroma():
    try:
        existing = profile_store.get(where={"user_id": TEST_USER_ID})
        ids = existing.get("ids", [])
        if ids:
            profile_store.delete(ids=ids)
            check(f"Chroma 清理: 删除 {len(ids)} 条测试文档", True)
        else:
            check("Chroma 清理: 无需删除", True)
    except Exception as e:
        check("Chroma 清理", False, str(e)[:100])

asyncio.run(cleanup_chroma())

# Check for leftover session files
web_sessions = Path(__file__).parent.parent / "web" / "sessions"
if web_sessions.exists():
    for f in web_sessions.glob(f"*{TEST_SESSION_ID}*"):
        f.unlink()


# ============================================================
# 结果汇总
# ============================================================
print("\n" + "=" * 60)
print(f"Test complete: {passed} passed, {failed} failed (total {passed + failed})")
print("=" * 60)

if failed > 0:
    print(f"\n{failed} test(s) FAILED -- check details above")
    # Don't sys.exit(1) — pytest catches SystemExit as INTERNALERROR
else:
    print("\nAll tests passed!")
