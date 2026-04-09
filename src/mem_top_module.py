from langchain.agents import create_agent
from mem_store_diary import store_diary
from read_file import read_file


memory_manager = create_agent(
    model = "deepseek-chat",
    tools = [store_diary, read_file],
    system_prompt = "你是记忆系统的管理员，负责管理用户的记忆。" \
    "你需要判断你获得的文件类型，并调用对应的工具把文件存储到数据库中." \
    "如果用户上传了多个文件，你需要多次调用工具来存储这些文件"
)
