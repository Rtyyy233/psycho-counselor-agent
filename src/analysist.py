from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel
from langchain.agents.structured_output import ToolStrategy
from typing import Literal,TypedDict,Annotated,Optional
from SharedContext import SharedContext

base_model = ChatDeepSeek(
    model= "deepseek-chat",
    temperature= 0.2
)


class analysis(BaseModel):
    analysist_state: Literal["true","false"] = "false"
    analysist_injection: Optional[str] = None


from mem_integration import retrieve_conv_outline_tool, retrieve_diary_tool,retrieve_materials_tool
from read_file import read_file

analysist = create_agent(
    model= base_model,
    system_prompt= "你是一名资深心理咨询分析师，你正在参与一个多Agent协作心理咨询系统"\
    "你将静默监听一段对话，当你认为有深度分析的必要时进行分析，"\
    "将analysist_state输出为true,并填写analysist_injection，否则保持沉默",
    tools= [
            retrieve_conv_outline_tool,
            retrieve_diary_tool,
            retrieve_materials_tool,
            read_file
    ],
    response_format= ToolStrategy(analysis)
)

async def call_analysist(SharedContext:SharedContext):
    await SharedContext.analyst_trigger.wait()

    messages = None

    async with SharedContext._lock:
        SharedContext.analysist_spare = False
        messages = SharedContext._messages[-10:]
    
    injection = await analysist.ainvoke({
        "messages": [{"role": "user", "content": messages}]
    })

    async with SharedContext._lock:
        SharedContext.analysist_spare = True
        SharedContext._analyst_injection.content = injection["messages"][-1].content #type:ignore
        SharedContext.analyst_trigger.clear()

    return

