from langchain.agents import create_agent
from mem_store_diary import store_diary
from read_file import read_file
from mem_store_material import store_materials
from mem_store_conv_outline import store_conversation_outline

# 不必要的架构设计，考虑舍去
memory_manager = create_agent(
    model = "deepseek-chat",
    tools = [store_diary, read_file, store_materials, store_conversation_outline],
    system_prompt = "你是记忆系统的存储管理员，负责管理用户的记忆存储。" \
    "你需要判断你获得的文件类型，并调用对应的工具把文件存储到数据库中." \
    "如果用户上传了多个文件，你需要多次调用工具来存储这些文件"
)
