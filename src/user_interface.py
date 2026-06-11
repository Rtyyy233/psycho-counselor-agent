import asyncio
import logging
import time
from typing import Dict, Optional
from langchain_core.documents import Document
from SharedContext import SharedContext
from supervisor import supervisor, SupervisionOutput, call_supervisor
from chatter import chatter, call_chatter, ChatterOutput
from session_manager import session_manager
from plan_manager import load_plan, save_plan, update_plan, build_continuity_context
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def load_command(user_input: str):
    """解析用户输入，如果是 /load 命令则返回 (True, id)，否则返回 (False, None)"""
    user_input = user_input.strip()
    if not user_input.startswith("/load"):
        return False, None
    
    # 按空格分割，例如 "/load 123" -> ["/load", "123"]
    parts = user_input.split()
    if len(parts) < 2:
        print("用法: /load <会话ID>")
        return True, None   # 命令格式正确但缺少参数
    
    load_id = parts[1].strip()
    return True, load_id

async def store_conversation_callback(conversation_text: str, metadata: Dict) -> str:
    """存储对话摘要的回调函数"""
    try:
        from mem_store_conv_outline import store_conversation_outline
        
        doc = Document(
            page_content=conversation_text,
            metadata={
                "source": f"session_{metadata['session_id']}",
                "cleaned_at": metadata["cleaned_at"],
                "message_count": metadata["message_count"],
                "token_count": metadata["token_count"],
                "target_usage": metadata.get("target_usage", 0.7)
            }
        )
        
        storage_id = await store_conversation_outline(doc)
        return storage_id
    except ImportError as e:
        raise Exception(f"无法导入存储模块: {e}")
    except Exception as e:
        raise Exception(f"存储对话摘要失败: {e}")

def parse_command(user_input: str):
    """解析用户命令，返回命令类型和参数"""
    user_input = user_input.strip()
    
    if user_input == "/exit":
        return "exit", None
    elif user_input.startswith("/load"):
        return "load", user_input[5:].strip()
    elif user_input == "/tokens":
        return "tokens", None
    elif user_input == "/help":
        return "help", None
    elif user_input.startswith("/clean"):
        parts = user_input.split()
        if len(parts) == 1:
            return "clean", {"target": 70, "auto_confirm": False}
        elif len(parts) == 2:
            try:
                target = int(parts[1])
                return "clean", {"target": target, "auto_confirm": False}
            except ValueError:
                return "clean", {"target": 70, "auto_confirm": parts[1].lower() == "auto"}
        elif len(parts) == 3:
            target = int(parts[1]) if parts[1].isdigit() else 70
            auto_confirm = parts[2].lower() == "auto"
            return "clean", {"target": target, "auto_confirm": auto_confirm}
        else:
            return "clean", {"target": 70, "auto_confirm": False}
    elif user_input.startswith("/profile"):
        parts = user_input.split()
        if len(parts) == 1:
            return "profile_view", None
        sub = parts[1].lower()
        if sub == "init":
            return "profile_init", None
        elif sub == "update":
            return "profile_update", None
        elif sub == "view" and len(parts) >= 3:
            try:
                return "profile_view_domain", int(parts[2])
            except ValueError:
                return "profile_view_domain", None
        elif sub == "help":
            return "profile_help", None
        else:
            return "profile_view_domain", None
    elif user_input.startswith("/plan"):
        parts = user_input.split()
        if len(parts) == 1:
            return "plan_view", None
        sub = parts[1].lower()
        if sub == "init":
            return "plan_init", None
        elif sub == "help":
            return "plan_help", None
        else:
            return "plan_view", None
    elif user_input.startswith("/"):
        # 未知命令
        return "unknown", user_input
    else:
        return "message", None

async def input_process(SharedContext: SharedContext) -> None:
    def log_exception(task):
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"后台任务异常: {e}")
    
    async def check_and_auto_clean():
        """检查并执行自动清理"""
        usage = await SharedContext.get_token_usage()
        
        # 自动清理阈值：95%
        AUTO_CLEAN_THRESHOLD = 0.95
        TARGET_USAGE = 0.70  # 清理到70%
        
        if usage["usage_percentage"] >= AUTO_CLEAN_THRESHOLD * 100:
            print(f"\n🤖 检测到令牌使用率过高 ({usage['usage_percentage']:.1f}%)，触发自动清理...")
            print(f"   目标: 清理到{TARGET_USAGE*100:.0f}%使用率")
            
            result = await SharedContext.cleanup_context(
                target_usage=TARGET_USAGE,
                storage_callback=store_conversation_callback
            )
            
            if result["status"] == "success":
                print(f"   ✅ 自动清理完成:")
                print(f"     清理消息: {result['cleaned_messages']}条")
                print(f"     释放令牌: {result['cleaned_tokens']:,}")
                print(f"     新使用率: {result['new_usage_percentage']:.1f}%")
                if result.get('storage_id'):
                    print(f"     摘要ID: {result['storage_id']}")
                    # 触发后台画像增量更新
                    try:
                        async def _trigger_profile_update():
                            from user_profile.profiler import update_profile
                            from mem_retrieve_user_profile import retrieve_user_profile_summary
                            summary_text = result.get('summary_text', '')
                            if summary_text:
                                await update_profile(
                                    SharedContext.user_id, summary_text, SharedContext,
                                    source_session=SharedContext.session_id
                                )
                                new_summary = await retrieve_user_profile_summary(SharedContext.user_id)
                                if new_summary:
                                    SharedContext.set_profile_summary(new_summary)
                        # 同时触发治疗计划更新
                        async def _trigger_plan_update():
                            plan = load_plan(SharedContext.user_id)
                            if plan is not None and not plan.is_empty():
                                summary_text = result.get('summary_text', '')
                                session_paip = result.get('summary_text', '')
                                messages = await SharedContext.get_recent_messages(20)
                                recent_dialogue = "\n".join([m.get("content", "") for m in messages])
                                plan_feedback = await SharedContext.get_and_clear_plan_feedback()
                                if plan_feedback:
                                    plan.supervisor_notes = plan_feedback
                                from datetime import datetime
                                await update_plan(
                                    SharedContext.user_id,
                                    SharedContext.session_id,
                                    datetime.now().strftime("%y.%m.%d"),
                                    session_paip,
                                    recent_dialogue,
                                    plan,
                                )
                                new_context = build_continuity_context(plan)
                                SharedContext.set_plan_context(new_context)
                        asyncio.create_task(_trigger_plan_update())
                        asyncio.create_task(_trigger_profile_update())
                    except Exception as e:
                        pass  # 画像更新失败不影响主流程
            else:
                print(f"   ⚠ 自动清理失败: {result.get('reason', result.get('error', '未知原因'))}")
                # 根据用户要求，存储失败时不删除消息，所以这里只是报告失败

            return True
        return False
    
    session_turn_count = 0

    async def _trigger_lightweight_profile_update():
        """每5轮触发一次轻量级 Domain 8 更新（含治疗计划匹配度数据）"""
        try:
            from user_profile.profiler import update_profile
            from mem_retrieve_user_profile import retrieve_user_profile_summary
            messages = await SharedContext.get_recent_messages(20)
            recent_text = "\n".join([m.get("content", "") for m in messages])
            # Inject plan intervention data so Domain 8 treatment_plan_alignment gets populated
            plan = load_plan(SharedContext.user_id)
            if plan is not None and not plan.is_empty():
                plan_section = "\n【治疗计划干预反应数据】\n"
                plan_section += f"主要方法: {plan.primary_approach}\n"
                if plan.intervention_responses:
                    plan_section += "干预效果记录:\n"
                    for r in plan.intervention_responses[-5:]:
                        plan_section += f"  - [{r.effect}] {r.skill_name}: {r.evidence[:100]}\n"
                recent_text = plan_section + "\n" + recent_text
            await update_profile(
                SharedContext.user_id, recent_text, SharedContext,
                source_session=SharedContext.session_id
            )
            new_summary = await retrieve_user_profile_summary(SharedContext.user_id)
            if new_summary:
                SharedContext.set_profile_summary(new_summary)
        except Exception:
            pass  # 轻量更新失败不影响主流程

    while True:
        user_input = input("type in here:")
        if not user_input:
            continue

        # 解析命令
        cmd_type, cmd_arg = parse_command(user_input)
        
        if cmd_type == "exit":
            # 保存前处理治疗计划反馈
            plan = load_plan(SharedContext.user_id)
            if plan is not None and not plan.is_empty():
                plan_feedback = await SharedContext.get_and_clear_plan_feedback()
                if plan_feedback:
                    plan.supervisor_notes = plan_feedback
                    try:
                        from datetime import datetime
                        messages = await SharedContext.get_recent_messages(20)
                        recent_dialogue = "\n".join([m.get("content", "") for m in messages])
                        await update_plan(
                            SharedContext.user_id, SharedContext.session_id,
                            datetime.now().strftime("%y.%m.%d"),
                            plan.last_session_plan, recent_dialogue, plan,
                        )
                    except Exception:
                        pass  # plan update failure shouldn't block exit
            await SharedContext.auto_save()
            print("会话已保存，再见！")
            return
        elif cmd_type == "tokens":
            # 显示令牌使用情况
            usage = await SharedContext.get_token_usage()
            print(f"\n📊 令牌使用情况:")
            print(f"   当前使用: {usage['current_tokens']:,} tokens")
            print(f"   上限: {usage['token_limit']:,} tokens")
            print(f"   剩余: {usage['remaining_tokens']:,} tokens")
            print(f"   使用率: {usage['usage_percentage']:.1f}%")
            print(f"   ⚠ 80%警告: {'是' if usage['is_near_limit_80'] else '否'}")
            print(f"   🚨 90%警告: {'是' if usage['is_near_limit_90'] else '否'}")
            print(f"   ❌ 超出限制: {'是' if usage['is_over_limit'] else '否'}")
            continue
        elif cmd_type == "load":
            if cmd_arg:
                print(f"正在加载会话: {cmd_arg}")
                new_ctx = await SharedContext.load_from_file(cmd_arg)
                if new_ctx:
                    # 将新实例的状态复制到当前实例
                    async with SharedContext._lock:
                        # 复制消息
                        SharedContext._messages.clear()
                        SharedContext._messages.extend(new_ctx._messages)
                        # 复制统计信息
                        SharedContext._stats = new_ctx._stats.copy()
                        # 复制注入状态
                        SharedContext._supervisor_injection = new_ctx._supervisor_injection
                        SharedContext._chatter_retrieval_injection = new_ctx._chatter_retrieval_injection
                        # 更新session_id
                        SharedContext.session_id = new_ctx.session_id
                        # 重置事件和标志（加载新会话后应重置）
                        SharedContext.supervisor_trigger.clear()
                        SharedContext.supervisor_spare = True
                        # 注意：token_limit和_tokenizer保持不变
                    
                    print(f"会话加载成功: {cmd_arg}")
                    # 显示加载后的令牌使用情况
                    usage = await SharedContext.get_token_usage()
                    print(f"   加载后令牌: {usage['current_tokens']:,}/{usage['token_limit']:,}")
                else:
                    print(f"加载会话失败: {cmd_arg}")
            else:
                print("用法: /load <会话ID>")
            continue
        elif cmd_type == "help":
            print("\n可用命令:")
            print("  /tokens - 显示当前令牌使用情况")
            print("  /load <ID> - 加载指定ID的会话")
            print("  /clean [目标百分比] [auto] - 清理旧消息并存储摘要")
            print("     示例: /clean 70 - 清理到70%使用率")
            print("           /clean auto - 自动确认清理到70%")
            print("           /clean 60 auto - 自动清理到60%")
            print("  /profile - 查看用户画像摘要")
            print("  /profile init - 初始化用户画像（从历史数据生成）")
            print("  /profile update - 手动更新用户画像")
            print("  /profile view <1-14> - 查看特定领域详情")
            print("  /plan - 查看治疗计划")
            print("  /plan init - 运行 PlanGenerator Agent 制定/重新制定治疗计划")
            print("  /help - 显示此帮助信息")
            print("  /exit - 退出程序")
            continue
        elif cmd_type == "clean":
            target = cmd_arg["target"]
            auto_confirm = cmd_arg["auto_confirm"]
            
            # 显示当前状态
            usage = await SharedContext.get_token_usage()
            print(f"\n🧹 清理计划:")
            print(f"   当前使用: {usage['current_tokens']:,}/{usage['token_limit']:,} tokens ({usage['usage_percentage']:.1f}%)")
            print(f"   目标使用: {target}% ({int(SharedContext.token_limit * target/100):,} tokens)")
            
            # 计算预估清理量
            tokens_to_clean = max(0, usage['current_tokens'] - int(SharedContext.token_limit * target/100))
            if tokens_to_clean == 0:
                print("   无需清理，当前使用率已低于目标")
                continue
                
            print(f"   预估清理: {tokens_to_clean:,} tokens")
            
            # 确认（除非auto模式）
            if not auto_confirm:
                confirm = input(f"   确认清理？(y/N): ").lower()
                if confirm != 'y':
                    print("   清理取消")
                    continue
            
            # 执行清理
            print("   🚀 执行清理中...")
            result = await SharedContext.cleanup_context(
                target_usage=target/100,
                storage_callback=store_conversation_callback
            )
            
            # 显示结果
            if result["status"] == "success":
                print(f"   ✅ 清理完成:")
                print(f"     清理消息: {result['cleaned_messages']}条")
                print(f"     释放令牌: {result['cleaned_tokens']:,}")
                print(f"     剩余令牌: {result['remaining_tokens']:,}")
                print(f"     剩余消息: {result['remaining_messages']}条")
                print(f"     新使用率: {result['new_usage_percentage']:.1f}%")
                if result.get('storage_id'):
                    print(f"     摘要ID: {result['storage_id']}")
            else:
                print(f"   ⚠ 清理失败: {result.get('reason', result.get('error', '未知原因'))}")
            continue
        elif cmd_type == "profile_view":
            # /profile - 查看画像摘要
            print(f"\n📋 用户画像 ({SharedContext.user_id}):")
            try:
                from mem_retrieve_user_profile import retrieve_user_profile
                result = await retrieve_user_profile(SharedContext.user_id)
                if result:
                    for d in result.domains:
                        cat_mark = {"baseline": "📌", "dynamic": "🔄", "emotional_world": "💭", "communication": "💬"}
                        mark = cat_mark.get("", "")
                        print(f"  {mark} #{d.domain_number} {d.domain_name}: {d.summary[:100]}")
                else:
                    print("  (暂无画像，使用 /profile init 创建)")
            except Exception as e:
                print(f"  ❌ 读取画像失败: {e}")
            continue
        elif cmd_type == "profile_view_domain":
            # /profile view <n>
            domain_num = cmd_arg
            if domain_num is None or domain_num < 1 or domain_num > 14:
                print("用法: /profile view <领域编号1-14>")
                continue
            print(f"\n📋 领域 #{domain_num}:")
            try:
                from mem_retrieve_user_profile import retrieve_user_profile
                result = await retrieve_user_profile(SharedContext.user_id, domain_numbers=[domain_num])
                if result and result.domains:
                    print(result.domains[0].details_text[:2000])
                else:
                    print(f"  (领域 #{domain_num} 暂无内容)")
            except Exception as e:
                print(f"  ❌ 读取失败: {e}")
            continue
        elif cmd_type == "profile_init":
            # /profile init
            print(f"\n🏗️ 开始初始化用户画像 ({SharedContext.user_id})...")
            print("   这将从所有历史对话、日记和材料中生成画像，可能需要几分钟。")
            confirm = input("   确认？(y/N): ").lower()
            if confirm != 'y':
                print("   已取消")
                continue
            try:
                from user_profile.profiler import init_profile
                from mem_retrieve_user_profile import retrieve_user_profile_summary
                profile = await init_profile(SharedContext.user_id, SharedContext,
                                              source_session=SharedContext.session_id)
                print(f"   ✅ 画像初始化完成: version={profile.version}, domains={len(profile.domains)}")
                # 刷新 chatter 用的 L1 摘要
                summary = await retrieve_user_profile_summary(SharedContext.user_id)
                if summary:
                    SharedContext.set_profile_summary(summary)
                    print(f"   📋 画像摘要已注入对话上下文")
            except Exception as e:
                print(f"   ❌ 初始化失败: {e}")
                import traceback
                traceback.print_exc()
            continue
        elif cmd_type == "profile_update":
            # /profile update
            print(f"\n🔄 手动触发画像增量更新...")
            try:
                from user_profile.profiler import update_profile
                from mem_retrieve_user_profile import retrieve_user_profile_summary
                messages = await SharedContext.get_recent_messages(50)
                recent_text = "\n".join([m.get("content", "") for m in messages])
                profile = await update_profile(SharedContext.user_id, recent_text, SharedContext,
                                                source_session=SharedContext.session_id)
                print(f"   ✅ 画像更新完成: version={profile.version}, updated_at={profile.updated_at}")
                # 刷新 chatter 用的 L1 摘要
                summary = await retrieve_user_profile_summary(SharedContext.user_id)
                if summary:
                    SharedContext.set_profile_summary(summary)
            except Exception as e:
                print(f"   ❌ 更新失败: {e}")
            continue
        elif cmd_type == "profile_help":
            print("\n📋 /profile 命令:")
            print("  /profile              - 查看画像14领域摘要")
            print("  /profile init         - 从头初始化画像（从已有日记+材料+对话）")
            print("  /profile update       - 手动触发增量更新")
            print("  /profile view <编号>  - 查看特定领域的完整内容")
            print("  /profile help         - 显示此帮助")
            continue
        elif cmd_type == "plan_view":
            # /plan — view current treatment plan
            print(f"\n📋 治疗计划 ({SharedContext.user_id}):")
            plan = load_plan(SharedContext.user_id)
            if plan is None or plan.is_empty():
                print("  (暂无治疗计划，使用 /plan init 创建)")
            else:
                print(f"  版本: v{plan.version}")
                print(f"  阶段: {plan.get_stage_label()}")
                print(f"  主要方法: {plan.primary_approach}")
                if plan.primary_approach_rationale:
                    print(f"    理由: {plan.primary_approach_rationale[:200]}")
                print(f"  辅助方法: {', '.join(plan.secondary_approaches) if plan.secondary_approaches else '无'}")
                print(f"  谨慎使用: {', '.join(plan.cautionary_approaches) if plan.cautionary_approaches else '无'}")
                if plan.case_conceptualization:
                    print(f"  个案概念化: {plan.case_conceptualization[:300]}...")
                print(f"  目标:")
                for g in plan.goals:
                    icon = {"green": "🟢", "yellow": "🟡", "red": "🔴", "white": "⚪"}.get(g.progress_status, "⚪")
                    print(f"    {icon} [{g.priority}] {g.description[:100]}")
                    print(f"       指标: {g.indicator[:80]}")
                    print(f"       时间线: {g.timeline}  进展: {g.progress:.0%}")
                if plan.pending_items:
                    print(f"  未完成事项:")
                    for item in plan.pending_items[:5]:
                        print(f"    - {item}")
                if plan.last_session_plan:
                    print(f"  上次会话计划: {plan.last_session_plan[:200]}")
                if plan.supervisor_notes:
                    print(f"  督导备注: {plan.supervisor_notes[:200]}")
            continue
        elif cmd_type == "plan_init":
            # /plan init — run PlanGenerator Agent
            print(f"\n🏗️ 开始制定治疗计划 ({SharedContext.user_id})...")
            print("   PlanGenerator Agent 将自主检索日记、材料、对话摘要和用户画像...")
            print("   这可能需要 1-3 分钟。")
            confirm = input("   确认？(y/N): ").lower()
            if confirm != 'y':
                print("   已取消")
                continue
            try:
                from plan_generator import generate_plan, plan_output_to_treatment_plan
                output = await generate_plan(SharedContext.user_id,
                                             source_session=SharedContext.session_id)
                plan = plan_output_to_treatment_plan(output, SharedContext.user_id,
                                                     source_session=SharedContext.session_id)
                save_plan(plan)
                # Refresh plan context for chatter
                plan_context = build_continuity_context(plan)
                SharedContext.set_plan_context(plan_context)
                print(f"   ✅ 治疗计划制定完成:")
                print(f"     版本: v{plan.version}")
                print(f"     阶段: {plan.get_stage_label()}")
                print(f"     主要方法: {plan.primary_approach}")
                print(f"     目标数: {len(plan.goals)}")
                if plan.cautionary_approaches:
                    print(f"     ⚠ 谨慎使用: {', '.join(plan.cautionary_approaches)}")
            except Exception as e:
                print(f"   ❌ 计划制定失败: {e}")
                import traceback
                traceback.print_exc()
            continue
        elif cmd_type == "plan_help":
            print("\n📋 /plan 命令:")
            print("  /plan              - 查看当前治疗计划")
            print("  /plan init         - 运行 PlanGenerator Agent 制定/重新制定治疗计划")
            print("  /plan help         - 显示此帮助")
            continue
        elif cmd_type == "unknown":
            print(f"未知命令: {cmd_arg}")
            print("输入 /help 查看可用命令")
            continue
        elif cmd_type == "message":
            # 普通消息处理流程
            session_turn_count += 1

            async with SharedContext._lock:
                if SharedContext._supervisor_injection:
                    user_input = "supervisor:" + SharedContext._supervisor_injection.content + "\n" + user_input

            # 添加用户消息（触发analyst和supervisor事件）
            await SharedContext.add_message("user", user_input)

            # 检查并执行自动清理（用户消息添加后）
            cleaned = await check_and_auto_clean()

            # 检查令牌使用情况（用户消息添加后）
            usage = await SharedContext.get_token_usage()
            if usage["is_near_limit_80"]:
                print(f"\n⚠ 警告：上下文令牌使用已达{usage['usage_percentage']:.1f}%")
                print(f"   当前: {usage['current_tokens']:,} tokens, 上限: {usage['token_limit']:,}")
                print(f"   建议使用 /tokens 命令查看详细使用情况")

            # 获取治疗计划上下文
            plan_context = SharedContext.get_plan_context()

            # 启动后台监督分析任务（合并了 analyst + supervisor 职责，传入 plan context）
            if SharedContext.supervisor_spare:
                task = asyncio.create_task(call_supervisor(SharedContext, plan_context, session_turn_count))
                task.add_done_callback(log_exception)

            # 获取AI回复（结构化输出，附带画像L1摘要和治疗计划上下文）
            profile_summary = SharedContext.get_profile_summary()
            output = await call_chatter(SharedContext, user_input, profile_summary, plan_context)
            print(output.reply)

            # 添加助手消息
            await SharedContext.add_message("assistant", output.reply)

            # 每 5 轮触发一次轻量级画像更新（Domain 8 干预反应保持新鲜）
            if session_turn_count % 5 == 0:
                asyncio.create_task(_trigger_lightweight_profile_update())

            # 检查是否需要中期计划更新（Supervisor 标记 should_revise_plan >= 2 次）
            if await SharedContext.get_and_reset_revise_plan_count() >= 2:
                async def _mid_session_plan_update():
                    plan = load_plan(SharedContext.user_id)
                    if plan is not None and not plan.is_empty():
                        from datetime import datetime as dt
                        feedback = SharedContext.get_and_clear_plan_feedback()
                        if feedback:
                            plan.supervisor_notes = (
                                f"{plan.supervisor_notes}\n[中期更新 {dt.now().strftime('%H:%M')}] {feedback}"
                            ).strip()
                        messages = await SharedContext.get_recent_messages(20)
                        recent = "\n".join([m.get("content", "") for m in messages])
                        await update_plan(
                            SharedContext.user_id, SharedContext.session_id,
                            dt.now().strftime("%y.%m.%d"),
                            plan.last_session_plan, recent, plan,
                        )
                        new_ctx = build_continuity_context(plan)
                        SharedContext.set_plan_context(new_ctx)
                        logger.info("中期治疗计划更新完成: user=%s version=%d",
                                    SharedContext.user_id, plan.version)
                asyncio.create_task(_mid_session_plan_update())

            # 再次检查令牌使用情况（助手消息添加后）
            usage = await SharedContext.get_token_usage()
            if usage["is_near_limit_90"]:
                print(f"\n🚨 严重警告：上下文令牌使用已达{usage['usage_percentage']:.1f}%")
                print(f"   当前: {usage['current_tokens']:,} tokens, 上限: {usage['token_limit']:,}")
                print(f"   建议考虑清理旧消息以避免超出限制")
            elif usage["is_over_limit"]:
                print(f"\n❌ 错误：上下文令牌已超出限制！")
                print(f"   当前: {usage['current_tokens']:,} tokens, 上限: {usage['token_limit']:,}")
                print(f"   必须立即清理旧消息！")

           
async def main_async():
    """异步主函数"""
    # 尝试加载DeepSeek官方tokenizer
    tokenizer = None
    tokenizer_type = "字符估算模式"
    
    try:
        # 尝试从SharedContext加载tokenizer
        tokenizer = SharedContext.load_deepseek_tokenizer()
        if tokenizer:
            tokenizer_type = "DeepSeek官方tokenizer"
            print(f"✅ DeepSeek tokenizer加载成功")
        else:
            print(f"⚠ 无法加载DeepSeek tokenizer，使用字符估算模式")
    except ImportError as e:
        print(f"⚠ {e}")
        print("  使用字符估算模式（3字符≈1token）")
    except Exception as e:
        print(f"⚠ 加载tokenizer时出错: {e}")
        print("  使用字符估算模式")
    
    # 创建带令牌管理的上下文
    state = SharedContext(
        session_id="default",
        token_limit=1000000,  # DeepSeek V4标准窗口
        tokenizer=tokenizer
    )
    
    load_dotenv()
    
    # 显示初始化信息
    print(f"\n🤖 心理咨询系统已启动")
    print(f"📝 会话ID: {state.session_id}")
    print(f"🎯 上下文令牌限制: {state.token_limit:,} tokens")
    print(f"🔧 Tokenizer模式: {tokenizer_type}")
    print(f"💡 输入 /help 查看可用命令")
    print()
    
    # 尝试加载已有画像（不自动初始化，避免启动时长时间阻塞）
    try:
        from user_profile.profiler import load_profile
        from mem_retrieve_user_profile import retrieve_user_profile_summary
        profile = load_profile(state.user_id)
        if profile is not None:
            print(f"📋 已加载用户画像: user={state.user_id}, version={profile.version}")
            # 构建并设置 L1 摘要，供 chatter 每轮自动使用
            summary = await retrieve_user_profile_summary(state.user_id)
            if summary:
                state.set_profile_summary(summary)
        else:
            print(f"📋 暂无用户画像，使用 /profile init 创建")
    except Exception as e:
        print(f"⚠ 画像加载失败: {e}")

    # 尝试加载治疗计划（不自动初始化 PlanGenerator，避免启动时长时间阻塞）
    try:
        plan = load_plan(state.user_id)
        if plan is not None and not plan.is_empty():
            print(f"📋 已加载治疗计划: user={state.user_id}, version={plan.version}, stage={plan.get_stage_label()}")
            plan_context = build_continuity_context(plan)
            state.set_plan_context(plan_context)
        else:
            print(f"📋 暂无治疗计划，使用 /plan init 创建")
    except Exception as e:
        print(f"⚠ 治疗计划加载失败: {e}")

    # 检查初始令牌使用情况
    try:
        usage = await state.get_token_usage()
        print(f"📊 初始令牌使用: {usage['current_tokens']:,}/{usage['token_limit']:,} tokens")
    except Exception as e:
        print(f"⚠ 检查初始令牌失败: {e}")

    # 启动用户界面
    await input_process(state)

if __name__ == "__main__":
    asyncio.run(main_async())