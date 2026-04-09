from langchain.agents import create_agent
from memory_storation import store_diary, read_file


memory_manager = create_agent(
    model = "deepseek-chat",
    tools = [store_diary, read_file],
    system_prompt = "你是一个心理咨询Agent中记忆系统的管理员，负责管理用户的记忆。" \
    "可能的输入类型有：1.日记、文章、病例等用户提供的原始资料 2.用户与Agent的对话记录摘要"\
    "3.Agent对用户形成的结构化分析"\
    "你需要判断获得的输入类型，如果是文件，打开文件判断文件类型，"\
    "并调用对应的工具把文件存储到对应的数据库中." \
    "如果用户上传了多个文件，你需要多次调用工具来存储这些文件"\
    "如果文字，判断输入类型，存储到对应数据库中"
)
