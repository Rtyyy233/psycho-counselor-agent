#!/usr/bin/env python3
"""
调试版本的 input_process，添加超时和日志
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from SharedContext import SharedContext
from user_interface import load_command
import time

async def call_analysist_with_timeout(ctx, timeout=10):
    """带超时的 call_analysist"""
    try:
        print(f"  [DEBUG] 开始 call_analysist (超时: {timeout}s)")
        start = time.time()
        from analysist import call_analysist
        await asyncio.wait_for(call_analysist(ctx), timeout=timeout)
        print(f"  [DEBUG] call_analysist 完成 (耗时: {time.time()-start:.2f}s)")
    except asyncio.TimeoutError:
        print(f"  [WARN] call_analysist 超时 (>{timeout}s)")
    except Exception as e:
        print(f"  [ERROR] call_analysist 异常: {e}")

async def call_supervisor_with_timeout(ctx, timeout=10):
    """带超时的 call_supervisor"""
    try:
        print(f"  [DEBUG] 开始 call_supervisor (超时: {timeout}s)")
        start = time.time()
        from supervisor import call_supervisor
        await asyncio.wait_for(call_supervisor(ctx), timeout=timeout)
        print(f"  [DEBUG] call_supervisor 完成 (耗时: {time.time()-start:.2f}s)")
    except asyncio.TimeoutError:
        print(f"  [WARN] call_supervisor 超时 (>{timeout}s)")
    except Exception as e:
        print(f"  [ERROR] call_supervisor 异常: {e}")

async def debug_input_process(ctx):
    """调试版本的 input_process"""
    from chatter import chatter
    
    while True:
        user_input = input("type in here: ")
        print(f"[DEBUG] 用户输入: {user_input}")
        
        if user_input:
            if user_input == "/exit":
                print("[DEBUG] 退出命令")
                ctx.auto_save
                return
            
            load, load_id = load_command(user_input)
            if load and load_id is not None:
                print(f"[DEBUG] 加载命令: {load_id}")
                await ctx.load_from_file(load_id)
            
            # 检查注入内容
            async with ctx._lock:
                if ctx._analyst_injection:
                    user_input += "anlysist:" + ctx._analyst_injection.content
                    print(f"[DEBUG] 添加 analyst 注入")
                if ctx._supervisor_injection:
                    user_input += "supervisor:" + ctx._supervisor_injection.content
                    print(f"[DEBUG] 添加 supervisor 注入")
            
            # 调用 analysist 和 supervisor（带超时）
            if ctx.analysist_spare:
                print("[DEBUG] 调用 analysist...")
                await call_analysist_with_timeout(ctx, timeout=5)
            else:
                print("[DEBUG] analysist_spare=False，跳过")
                
            if ctx.supervisor_spare:
                print("[DEBUG] 调用 supervisor...")
                await call_supervisor_with_timeout(ctx, timeout=5)
            else:
                print("[DEBUG] supervisor_spare=False，跳过")
            
            # 获取历史消息
            print("[DEBUG] 获取历史消息...")
            history = await ctx.get_recent_messages(50)
            history_messages = [msg["content"] for msg in history]
            chat_input = "\n\n".join(history_messages) + "\n\n" + user_input
            print(f"[DEBUG] chat_input 长度: {len(chat_input)} 字符")
            
            # 添加用户消息
            print("[DEBUG] 添加用户消息到上下文...")
            await ctx.add_message("user", user_input)
            
            # 调用 chatter
            print("[DEBUG] 调用 chatter...")
            try:
                start = time.time()
                reply = await asyncio.wait_for(
                    chatter.ainvoke({
                        "messages": [{"role": "user", "content": chat_input}]
                    }),
                    timeout=30
                )
                print(f"[DEBUG] chatter 响应完成 (耗时: {time.time()-start:.2f}s)")
                
                assistant_reply = reply["messages"][-1].content
                print(f"[DEBUG] AI 回复: {assistant_reply[:100]}...")
                print("\n" + "="*50)
                print(f"AI: {assistant_reply}")
                print("="*50 + "\n")
                
                await ctx.add_message("assistant", assistant_reply)
                
            except asyncio.TimeoutError:
                print("[ERROR] chatter 调用超时 (30s)")
            except Exception as e:
                print(f"[ERROR] chatter 调用异常: {e}")

async def main():
    print("=== 调试模式 ===")
    print("正在初始化 SharedContext...")
    ctx = SharedContext()
    
    print("开始交互式调试。输入 /exit 退出。")
    try:
        await debug_input_process(ctx)
    except KeyboardInterrupt:
        print("\n[DEBUG] 用户中断")
    except Exception as e:
        print(f"[ERROR] 主循环异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())