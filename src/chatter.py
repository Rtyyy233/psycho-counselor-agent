from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel
from typing import Literal,TypedDict,Annotated,Optional

base_model = ChatDeepSeek(
    model= "deepseek-chat",
    temperature= 0.5
)

from mem_integration import read_file_tool

chatter = create_agent(
    model= base_model,
    system_prompt= "你是一名资深的心理咨询专家，你正在参与一个多Agent协作心理咨询系统," \
    "有时你会收到来自分析师analysis、督导supervisor的协作建议，这时务必参考；当你没有收到建议的时候，" \
    "你需要按照分析模式-表达共情-引导聚焦情绪的模式，作为情绪聚焦疗法的专家回应、安抚用户" \
    "当用户上传文件时，阅读文件并进行初步分析",
    tools=[read_file_tool]
)
