from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel
from langchain.agents.structured_output import ToolStrategy
from typing import Literal,TypedDict,Annotated,Optional
from SharedContext import SharedContext
from config import LLM_MODEL
import logging
import asyncio

logger = logging.getLogger(__name__)
_analyst_lock = asyncio.Lock()  # 防止多个analyst任务并发执行

base_model = ChatDeepSeek(
    model= LLM_MODEL,
    temperature= 0.2
)


class analysis(BaseModel):
    analysist_state: Literal["true","false"] = "false"
    analysist_injection: Optional[str] = None


from mem_integration import retrieve_conv_outline_tool, retrieve_diary_tool,retrieve_materials_tool, read_file_tool

analysist = create_agent(
    model= base_model,
    system_prompt= "你是一名资深心理咨询分析师，你正在参与一个多Agent协作心理咨询系统"\
    "你将静默监听一段对话，当你认为有深度分析的必要时进行分析，"\
    "将analysist_state输出为true,并填写analysist_injection，否则保持沉默。"\
    "每次对话新开始时，你可以查询过去的对话摘要来形成对用户的初步印象。"\
    "例如:1.当用户谈及过去的事情，你可以检索是否有相关的日记内容，如果检索成功，结合当前对话" \
    "分析日记内容，探索用户当前问题可能的成因，识别用户一贯的思维模式等",
    
    tools= [
            retrieve_conv_outline_tool,
            retrieve_diary_tool,
            retrieve_materials_tool,
            read_file_tool
    ],
    response_format= ToolStrategy(analysis)
)

async def call_analysist(SharedContext:SharedContext):
    # 防止多个analyst任务并发执行
    if _analyst_lock.locked():
        # 锁已被占用，直接返回
        logger.debug("Analyst任务已在运行，跳过")
        return
    
    try:
        await _analyst_lock.acquire()
    except Exception as e:
        # 获取锁失败，直接返回
        logger.debug(f"获取Analyst锁失败: {e}")
        return
    
    try:
        await SharedContext.analyst_trigger.wait()

        messages = None
        injection = None
        async with SharedContext._lock:
            SharedContext.analysist_spare = False
            messages = SharedContext._messages[-10:] if SharedContext._messages else []
        
        # 将消息列表格式化为字符串
        if not messages:
            logger.debug("没有消息可分析")
            return
        
        formatted_history = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in messages
        ])
        
        injection = await analysist.ainvoke({
            "messages": [{"role": "user", "content": formatted_history}]
        })

        # 检查响应格式
        if not injection or "messages" not in injection or not injection["messages"]:
            logger.warning("Analyst返回无效响应格式")
            return
        
        injection_content = injection["messages"][-1].content
        if not injection_content or len(injection_content.strip()) < 5:
            logger.debug("Analyst返回内容过短，忽略")
            return

        async with SharedContext._lock:
            # 检查并创建 PromptInjection 对象
            if SharedContext._analyst_injection is None:
                from SharedContext import PromptInjection
                import time
                SharedContext._analyst_injection = PromptInjection(
                    content=injection_content,
                    timestamp=time.time(),
                    source="analyst"
                )
            else:
                SharedContext._analyst_injection.content = injection_content
    except Exception as e:
        logger.error(f"Analyst调用失败: {e}")
        # 不重新抛出异常，避免影响主流程
    finally:
        async with SharedContext._lock:
            SharedContext.analysist_spare = True
        SharedContext.analyst_trigger.clear()
        # 释放锁
        if _analyst_lock.locked():
            _analyst_lock.release()

    return

