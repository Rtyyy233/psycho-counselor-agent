from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel
from typing import Literal,TypedDict,Annotated,Optional

base_model = ChatDeepSeek(
    model= "deepseek-chat",
    temperature= 0.2
)

from mem_integration import retrieve_conv_outline_tool, retrieve_diary_tool,retrieve_materials_tool

analysist = create_agent(
    model= base_model,
    system_prompt= "你是一名资深心理咨询分析师，你正在参与一个多Agent协作心理咨询系统"\
    "你将静默监听一段对话，当你认为有深度分析的必要时进行分析，"\
    "将analysist_state输出为true,并填写analysist_injection，否则保持沉默",
    tools= [
            retrieve_conv_outline_tool,
            retrieve_diary_tool,
            retrieve_materials_tool
    ]
)

class analysis:
    analysist_state: Literal["true","false"]
    analysist_injection: Optional[str] = None
