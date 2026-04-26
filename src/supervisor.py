from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel
from langchain.agents.structured_output import ToolStrategy
from typing import Literal,TypedDict,Annotated,Optional
from SharedContext import SharedContext
from config import LLM_MODEL

base_model = ChatDeepSeek(
    model= LLM_MODEL,
    temperature= 0.1
)

class supervision(BaseModel):
    supervison_state: Literal["true", "false"] = "false"
    supervisor_injection: Optional[str] = None


from mem_integration import read_file_tool,store_material_tool,store_diary_tool

supervisor = create_agent(
    model= base_model,
    system_prompt= "你是一名资深的心理咨询专家，你正在参与一个多Agent协作心理咨询系统,"\
    "静默监听你收到的对话，当你认为需要提出督导建议的时候，将supervision_state改为true,"\
    "然后在supervisor_injection中填写你给出的督导建议" \
    "当用户上传文件时，阅读文件，判断文件类型，然后存储文件（无法判断为日记类型的就存储为材料）"\
    "注意：1.当你收到的对话中包含多轮对话时，需要判断最新的对话是否需要调用文件阅读与存储，不要" \
    "反复调用工具阅读与存储你已经存储过的文件" \
    "2.记得保持督导的一致性，对之前的督导建议记得跟踪其中重要的内容并敦促跟进，你的职责是对重要的治疗主题维护一致性",
    response_format= ToolStrategy(supervision),
    tools=[read_file_tool,store_material_tool,store_diary_tool],
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