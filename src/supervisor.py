from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel
from langchain.agents.structured_output import ToolStrategy
from typing import Literal,TypedDict,Annotated,Optional
from SharedContext import SharedContext

base_model = ChatDeepSeek(
    model= "deepseek-chat",
    temperature= 0.1
)

class supervision(BaseModel):
    supervison_state: Literal["true", "false"] = "false"
    supervisor_injection: Optional[str] = None


supervisor = create_agent(
    model= base_model,
    system_prompt= "你是一名资深的心理咨询专家，你正在参与一个多Agent协作心理咨询系统,"\
    "静默监听你收到的对话，当你认为需要提出督导建议的时候，将supervision_state改为true,"\
    "然后在supervisor_injection中填写你给出的督导建议",
    response_format= ToolStrategy(supervision)
)

async def call_supervisor(SharedContext: SharedContext):
    await SharedContext.supervisor_trigger.wait()
    
    messages = None
    async with SharedContext._lock:
        SharedContext.supervisor_spare = False
        messages = SharedContext._messages[-10:]
    
    # 将消息列表格式化为字符串
    formatted_history = "\n".join([
        f"{msg['role']}: {msg['content']}" 
        for msg in messages
    ])
    
    injection = await supervisor.ainvoke({
        "messages": [{"role": "user", "content": formatted_history}]
    })
    
    async with SharedContext._lock:
        SharedContext.supervisor_spare = True
        # 检查并创建 PromptInjection 对象
        if SharedContext._supervisor_injection is None:
            from SharedContext import PromptInjection
            import time
            SharedContext._supervisor_injection = PromptInjection(
                content=injection["messages"][-1].content,
                timestamp=time.time(),
                source="supervisor"
            )
        else:
            SharedContext._supervisor_injection.content = injection["messages"][-1].content
        SharedContext.supervisor_trigger.clear()  # Fixed: added parentheses
    
    return