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
    
    injection = await supervisor.ainvoke({
        "messages": [{"role": "user", "content": messages}]
    })
    
    async with SharedContext._lock:
        SharedContext.supervisor_spare = True
        SharedContext._supervisor_injection.content = injection["messages"][-1].content #type:ignore
        SharedContext.supervisor_trigger.clear()  # Fixed: added parentheses
    
    return