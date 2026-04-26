from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
from dotenv import load_dotenv
import os


def find_project_root(start_path=Path(__file__).parent):
    for parent in [start_path] + list(start_path.parents):
        if (parent / ".env").exists():
            return parent
    return start_path

PROJECT_ROOT = find_project_root()
load_dotenv(PROJECT_ROOT / ".env")  # 显式指定 .env 路径

# 获取相对路径字符串
rel_data_dir = os.getenv("DATA_DIR", "data")
# 转换为绝对路径
abs_data_dir = PROJECT_ROOT / rel_data_dir

data_path = PROJECT_ROOT / os.getenv("DATA_DIR", "data")

DATA_DIR = data_path

embeddings = OllamaEmbeddings(model="qwen3-embedding:4b")

original_diary = Chroma(
        collection_name = "original_diary",
        embedding_function=embeddings,
        persist_directory=str(data_path) # notice the problem of hard coed path
    )

diary_annotation = Chroma(
        collection_name = "diary_annotation",
        embedding_function=embeddings,
        persist_directory=str(data_path) # notice the problem of hard coed path
    )

material_store = Chroma(
        collection_name="child_chunks",
        embedding_function=embeddings,
        persist_directory=str(DATA_DIR),
    )

parent_store = Chroma(
        collection_name="parent_chunks",
        embedding_function=embeddings,
        persist_directory=str(DATA_DIR),
    )

conv_store = Chroma(
        collection_name="conv_outline",
        embedding_function=embeddings,
        persist_directory=str(DATA_DIR),
    )

from langchain_core.tools import tool


@tool
async def retrieve_diary_tool(query: str):
    """
    依据日记检索所需的信息
    """
    try:
        from mem_retrieve_diary import retrieve_diary
        logger.info(f"检索日记查询: {query}")
        results = await retrieve_diary(query)
        
        # 格式化结果为字符串
        if not results:
            return "未找到相关日记内容。"
        
        formatted = []
        for result in results:
            docs_text = "\n".join([doc.page_content[:200] + "..." for doc in result.documents[:3]])
            formatted.append(f"步骤 {result.step_id} ({result.mode}):\n{docs_text}")
        
        return "\n\n".join(formatted)
    except ImportError as e:
        logger.error(f"导入模块失败: {e}")
        return f"错误：无法导入日记检索模块 - {e}"
    except Exception as e:
        logger.error(f"检索日记失败: {e}")
        return f"错误：检索日记失败 - {e}"

@tool
async def retrieve_materials_tool(query: str):
    """
        依据侧写情绪的材料检索所需的信息（非日记类材料）
    """
    try:
        from mem_retrieve_material import retrieve_materials
        logger.info(f"检索材料查询: {query}")
        results = await retrieve_materials(query)
        
        # 格式化结果为字符串
        if not results:
            return "未找到相关材料内容。"
        
        formatted = []
        for result in results:
            children_text = "\n".join([doc.page_content[:200] + "..." for doc in result.matched_children[:3]])
            parent_text = "\n".join([doc.page_content[:200] + "..." for doc in result.parent_contexts[:3]])
            formatted.append(f"步骤 {result.step_id} ({result.mode}):\n儿童片段:\n{children_text}\n父级上下文:\n{parent_text}")
        
        return "\n\n".join(formatted)
    except ImportError as e:
        logger.error(f"导入模块失败: {e}")
        return f"错误：无法导入材料检索模块 - {e}"
    except Exception as e:
        logger.error(f"检索材料失败: {e}")
        return f"错误：检索材料失败 - {e}"

@tool
async def retrieve_conv_outline_tool(query: str):
    """
    从历史对话的摘要中检索所需的信息（注意：并非当前对话！！！）
    """
    try:
        from mem_retrieve_conv_outline import retrieve_conv_outline
        logger.info(f"检索对话摘要查询: {query}")
        results = await retrieve_conv_outline(query)
        
        # 格式化结果为字符串
        if not results:
            return "未找到相关对话摘要内容。"
        
        formatted = []
        for result in results:
            docs_text = "\n".join([doc.page_content[:200] + "..." for doc in result.matched_docs[:3]])
            paip_text = "\n".join([f"{section.section}: {section.content[:100]}..." for section in result.paip_outlines[:3]])
            formatted.append(f"步骤 {result.step_id} ({result.mode}):\n匹配文档:\n{docs_text}\nPAIP摘要:\n{paip_text}")
        
        return "\n\n".join(formatted)
    except ImportError as e:
        logger.error(f"导入模块失败: {e}")
        return f"错误：无法导入对话摘要检索模块 - {e}"
    except Exception as e:
        logger.error(f"检索对话摘要失败: {e}")
        return f"错误：检索对话摘要失败 - {e}"



@tool(
    "memory_manager",
    description= "call memory_manager to store files uploaded by user"
)
async def call_memory_manager(file_path:str):
    try:
        from mem_store_module import memory_manager
        logger.info(f"调用memory_manager存储文件: {file_path}")
        result = await memory_manager.ainvoke({"messages":[{"role": "user", "content": file_path}] })
        logger.info(f"memory_manager执行成功: {file_path}")
        return f"文件存储成功: {file_path}"
    except ImportError as e:
        logger.error(f"导入memory_manager模块失败: {e}")
        return f"错误：无法导入memory_manager模块 - {e}"
    except Exception as e:
        logger.error(f"存储文件失败 {file_path}: {e}")
        return f"错误：存储文件失败 - {e}"


import logging
from read_file import read_file

logger = logging.getLogger(__name__)

@tool
async def read_file_tool(file_path:str):
    """read file through file path provided"""
    try:
        logger.info(f"Reading file: {file_path}")
        content = read_file(file_path)
        logger.info(f"Successfully read file: {file_path}, length: {len(content)}")
        return content
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}")
        return f"错误：文件不存在 - {e}"
    except ValueError as e:
        logger.error(f"Unsupported file type or invalid path: {file_path} - {e}")
        return f"错误：不支持的文件类型或无效路径 - {e}"
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        return f"错误：读取文件失败 - {e}"

from mem_store_diary import store_diary
from mem_store_material import store_materials

@tool
async def store_diary_tool(file_path:str):
    """store diary after reading file"""
    try:
        from mem_store_diary import store_diary
        logger.info(f"存储日记文件: {file_path}")
        result = await store_diary(file_path)
        logger.info(f"日记存储成功: {file_path}")
        return f"日记存储成功: {result}"
    except ImportError as e:
        logger.error(f"导入store_diary模块失败: {e}")
        return f"错误：无法导入日记存储模块 - {e}"
    except Exception as e:
        logger.error(f"存储日记失败 {file_path}: {e}")
        return f"错误：存储日记失败 - {e}"

@tool
async def store_material_tool(file_path:str):
    """store materials after reading file"""
    try:
        from mem_store_material import store_materials
        logger.info(f"存储材料文件: {file_path}")
        result = await store_materials(file_path)
        logger.info(f"材料存储成功: {file_path}")
        return f"材料存储成功: {result}"
    except ImportError as e:
        logger.error(f"导入store_materials模块失败: {e}")
        return f"错误：无法导入材料存储模块 - {e}"
    except Exception as e:
        logger.error(f"存储材料失败 {file_path}: {e}")
        return f"错误：存储材料失败 - {e}"