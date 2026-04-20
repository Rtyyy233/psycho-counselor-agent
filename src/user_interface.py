import asyncio
import time
from typing import Dict
from langchain_core.documents import Document
from SharedContext import SharedContext
from supervisor import supervisor,supervision,call_supervisor
from analysist import analysist,analysis,call_analysist
from chatter import chatter
from session_manager import session_manager
from dotenv import load_dotenv

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
    elif user_input.startswith("/"):
        # 未知命令
        return "unknown", user_input
    else:
        return "message", None

async def input_process(SharedContext: SharedContext):
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
            else:
                print(f"   ⚠ 自动清理失败: {result.get('reason', result.get('error', '未知原因'))}")
                # 根据用户要求，存储失败时不删除消息，所以这里只是报告失败
                
            return True
        return False
    
    while True:
        user_input = input("type in here:")
        if not user_input:
            continue
            
        # 解析命令
        cmd_type, cmd_arg = parse_command(user_input)
        
        if cmd_type == "exit":
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
                success = await SharedContext.load_from_file(cmd_arg)
                if success:
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
        elif cmd_type == "unknown":
            print(f"未知命令: {cmd_arg}")
            print("输入 /help 查看可用命令")
            continue
        elif cmd_type == "message":
            # 普通消息处理流程
            
            # start conversation
            #chatter
            
            async with SharedContext._lock:
                if SharedContext._analyst_injection:
                    user_input += "analyst:" + SharedContext._analyst_injection.content
                if SharedContext._supervisor_injection:
                    user_input += "supervisor:" + SharedContext._supervisor_injection.content
            
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
            
            # 启动后台任务进行分析（不阻塞主流程）
            if SharedContext.analysist_spare:
                task = asyncio.create_task(call_analysist(SharedContext))
                task.add_done_callback(log_exception)
            if SharedContext.supervisor_spare:
                task = asyncio.create_task(call_supervisor(SharedContext))
                task.add_done_callback(log_exception)

            # 构建聊天输入
            history = await SharedContext.get_recent_messages(50)
            history_messages = [msg["content"] for msg in history]
            chat_input = "\n\n".join(history_messages) + "\n\n" + user_input

            # 获取AI回复
            reply = await chatter.ainvoke({
                "messages":[{"role": "user", "content": chat_input}]
            })
            print(reply["messages"][-1].content)

            # 添加助手消息
            await SharedContext.add_message("assistant", reply["messages"][-1].content)
            
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
        token_limit=128000,  # DeepSeek V3标准窗口
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