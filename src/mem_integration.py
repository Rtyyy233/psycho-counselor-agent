import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from read_file import read_file

logger = logging.getLogger(__name__)


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

profile_store = Chroma(
        collection_name="user_profile",
        embedding_function=embeddings,
        persist_directory=str(DATA_DIR),
    )


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

# 跨轮查询去重：记录本会话中已查询过的 (user_id, domain_number) 组合
_queried_profile_domains: dict[str, set[int]] = {}

# 每会话最多查询的画像领域数量
_MAX_PROFILE_DOMAIN_QUERIES = 5


def get_queried_profile_domains_summary(user_id: str) -> str:
    """返回已查询画像领域的摘要文本，用于注入 chatter 上下文。"""
    DOMAIN_NAMES = {
        1: "主诉与现状", 2: "成长与发展史", 3: "易感因素", 4: "诱发因素",
        5: "维持因素", 6: "保护因素", 7: "关系模式", 8: "干预反应",
        9: "风险评估", 10: "文化/背景因素", 11: "人格印象", 12: "情感世界",
        13: "沟通风格", 14: "求助与改变模式",
    }
    domains = _queried_profile_domains.get(user_id, set())
    if not domains:
        return "尚未查询任何画像领域。"

    names = [f"领域{d}({DOMAIN_NAMES.get(d, '')})" for d in sorted(domains)]
    count = len(domains)
    remaining = _MAX_PROFILE_DOMAIN_QUERIES - count

    if remaining <= 0:
        return (
            "已查询的画像领域: " + "、".join(names)
            + f"。查询配额已用尽（{count}/{_MAX_PROFILE_DOMAIN_QUERIES}），"
            "不得再查询任何画像领域，使用已有信息即可。"
        )

    return (
        "已查询的画像领域: " + "、".join(names)
        + f"。剩余查询配额: {remaining}次。"
        "仅在对话确实需要且该领域完全未知时才查询——大部分情况下已有信息已足够。"
    )


@tool
async def retrieve_user_profile_tool(query: str):
    """
    检索来访者的用户画像（用户画像）。当需要了解来访者的背景、人格特征、
    情绪模式、沟通风格、成长史等全面信息时使用。
    查询时请包含来访者的用户ID，例如"查询用户xxx的画像"或"获取xxx的人格特征"。
    可以指定领域编号来获取特定领域的详细内容，例如"查询用户default的画像领域2"。
    查询特定领域时会自动附带该领域fact的证据原始内容（日记、PAIP摘要等来源）。
    重要：同一领域查询一次即可，不要重复查询相同领域。
    """
    try:
        from mem_retrieve_user_profile import retrieve_user_profile

        import re

        # 提取 user_id
        user_id = ""
        for prefix in ("user:", "user_id=", "用户:", "用户ID:", "用户 "):
            ql = query.lower()
            pl = prefix.lower()
            if pl in ql:
                idx = ql.index(pl) + len(pl)
                rest = query[idx:].strip()
                m = re.match(r'[\w\-]+', rest)
                if m:
                    user_id = m.group()
                break

        if not user_id:
            user_id = "default"

        # 提取领域编号（如 "领域2", "领域 5", "domain 3"）
        domain_numbers = None
        domain_matches = re.findall(r'领域\s*(\d+)', query) or re.findall(r'domain\s*(\d+)', query, re.IGNORECASE)
        if domain_matches:
            domain_numbers = [int(n) for n in domain_matches if 1 <= int(n) <= 14]

        logger.info(f"检索用户画像查询: {query}, user_id={user_id or '(未提取)'}, domains={domain_numbers}")

        # --- 去重检查 + 配额检查 ---
        tracked = _queried_profile_domains.setdefault(user_id, set())
        if domain_numbers:
            already_queried = [d for d in domain_numbers if d in tracked]
            new_domains = [d for d in domain_numbers if d not in tracked]

            # 纯重复 → 拦截
            if already_queried and not new_domains:
                logger.info(f"用户画像领域已查询过，跳过: user={user_id}, domains={already_queried}")
                return (
                    f"【注意】领域 {','.join(map(str, already_queried))} 已在此对话中查询过。"
                    "请使用已有的画像信息，不要重复查询。"
                )

            # 配额用尽 → 拦截所有新查询
            if len(tracked) >= _MAX_PROFILE_DOMAIN_QUERIES:
                logger.info(f"用户画像查询配额用尽: user={user_id}, total={len(tracked)}")
                return (
                    f"【配额用尽】已查询 {len(tracked)} 个画像领域（上限 {_MAX_PROFILE_DOMAIN_QUERIES}），"
                    "对话中已有足够的背景信息。请直接基于已有信息回应来访者，不要再查询画像。"
                )

            # 部分新领域但未超配额 → 只查新领域
            if already_queried:
                logger.info(f"用户画像混合查询: user={user_id}, new={new_domains}, dup={already_queried}")
                # 只处理新领域，但在这里无法截断domain_numbers传给retrieve_user_profile
                # 简单处理：如果还有配额，允许查询包含已查领域，下游只是多返回一些文本
                # 实际只需记录新领域
            for d in domain_numbers:
                if len(tracked) < _MAX_PROFILE_DOMAIN_QUERIES:
                    tracked.add(d)
        else:
            # 未指定领域 = 查全部 → 检查是否已有全量
            if len(tracked) >= _MAX_PROFILE_DOMAIN_QUERIES:
                logger.info(f"用户画像查询配额用尽(全量查询): user={user_id}")
                return (
                    f"【配额用尽】已查询 {len(tracked)} 个画像领域（上限 {_MAX_PROFILE_DOMAIN_QUERIES}），"
                    "对话中已有足够的背景信息。请直接基于已有信息回应来访者。"
                )

        DISCLAIMER = "（以下为内部参考信息，请融入理解但不要直接引用原文或提及画像/证据等系统术语）\n\n"

        if user_id:
            result = await retrieve_user_profile(user_id, domain_numbers)
            if result:
                if domain_numbers and len(domain_numbers) <= 3:
                    domain_text = "\n\n".join([d.details_text[:2000] for d in result.domains])
                    return f"{DISCLAIMER}用户 {result.user_id}（领域 {','.join(map(str, domain_numbers))}）:\n{domain_text}"

                domain_summaries = "; ".join(
                    [f"#{d.domain_number} {d.domain_name}: {d.summary[:80]}"
                     for d in result.domains[:5]]
                )
                return f"{DISCLAIMER}用户 {result.user_id}:\n{domain_summaries}\n\n完整画像:\n{result.full_text[:3000]}"
            return f"未找到用户 {user_id} 的画像。"
        else:
            return "请提供用户ID以检索画像，例如'查询用户default的画像'。"

    except ImportError as e:
        logger.error(f"导入模块失败: {e}")
        return f"错误：无法导入用户画像检索模块 - {e}"
    except Exception as e:
        logger.error(f"检索用户画像失败: {e}")
        return f"错误：检索用户画像失败 - {e}"


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


@tool
async def read_file_tool(file_path:str):
    """read file through file path provided"""
    import os as _os

    # 校验路径：拒绝明显的虚构/Linux远程路径
    _suspicious_prefixes = (
        "/root/", "/home/", "/opt/", "/var/", "/tmp/", "/etc/",
        "/root/.openclaw", "/workspace/", "/paip_materials/",
    )
    _normalized = file_path.replace("\\", "/")
    for _pfx in _suspicious_prefixes:
        if _normalized.startswith(_pfx) or _pfx in _normalized:
            logger.warning(f"拒绝可疑文件路径: {file_path}")
            return (
                f"错误：路径 {file_path} 不存在，且看起来不是有效的本地文件路径。"
                "这可能是对话历史或检索结果中引用过的文件名，但原始文件已不可访问。"
                "请使用对话历史中已有的内容直接回应，不要尝试读取不存在的文件。"
            )
    try:
        logger.info(f"Reading file: {file_path}")
        content = read_file(file_path)
        logger.info(f"Successfully read file: {file_path}, length: {len(content)}")
        return content
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}")
        return (
            f"错误：文件 {file_path} 不存在。请勿根据检索结果中的文件名猜测路径——"
            "文件可能从未上传、已被移动或仅存在于历史引用中。"
            "请直接使用已有的检索内容回应，不要反复尝试读取不存在的文件。"
        )
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


