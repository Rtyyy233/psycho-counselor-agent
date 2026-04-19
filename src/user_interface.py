import asyncio
from SharedContext import SharedContext
from supervisor import supervisor,supervision,call_supervisor
from analysist import analysist,analysis,call_analysist
from chatter import chatter
from SharedContext import SharedContext
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

async def input_process(SharedContext: SharedContext):
    def log_exception(task):
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"后台任务异常: {e}")
    
    while True:
        user_input = input("type in here:")
        if user_input:
            if user_input == "/exit":
                SharedContext.auto_save
                return 
            load, id = load_command(user_input)
            if load and id != None:
                await SharedContext.load_from_file(id)

            # start conversation
            #chatter

            

            async with SharedContext._lock:
                if SharedContext._analyst_injection:
                    user_input += "anlysist:" + SharedContext._analyst_injection.content
                if SharedContext._supervisor_injection:
                    user_input += "supervisor:" + SharedContext._supervisor_injection.content
            


            # 添加用户消息（触发analyst和supervisor事件）
            await SharedContext.add_message("user", user_input)

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

           
if __name__ == "__main__":
    state = SharedContext()
    load_dotenv()
    asyncio.run(input_process(state))