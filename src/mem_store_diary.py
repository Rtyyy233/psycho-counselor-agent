from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
)

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
import re, asyncio
from pathlib import Path
import os
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import hashlib
from langchain_experimental.text_splitter import SemanticChunker
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

EmotionType = Literal[
    "喜悦",  # 快乐、高兴、兴奋
    "悲伤",  # 难过、失落、忧郁
    "焦虑",  # 担心、紧张、不安
    "愤怒",  # 生气、恼怒、愤慨
    "恐惧",  # 害怕、惊恐、畏惧
    "惊讶",  # 意外、震惊
    "厌恶",  # 反感、讨厌
    "平静",  # 平和、宁静、放松
    "疲惫",  # 疲倦、无力、耗竭
    "孤独",  # 孤单、寂寞
    "羞耻",  # 羞愧、丢脸、尴尬
    "内疚",  # 自责、愧疚
    "希望",  # 期盼、乐观、憧憬
    "迷茫",  # 困惑、不知所措
    "满足",  # 满意、充实、欣慰
    "害羞",  # 羞涩、腼腆
    "安全感",  # 安全、安心
    "兴奋",  # 激动、亢奋
    "失望",  # 失落、沮丧
    "感激",  # 感谢、感恩
    "爱",  # 喜爱、爱情
    "恨",  # 憎恨、怨恨
    "嫉妒",  # 妒忌、羡慕
    "自豪",  # 骄傲、自豪
    "自卑",  # 自卑、自我否定
    "好奇",  # 好奇、探究
    "无聊",  # 无聊、乏味
    "放松",  # 轻松、舒缓
    "紧张",  # 紧张、紧绷
    "困惑",  # 迷惑、不解
]


def find_project_root(start_path=Path(__file__).parent):
    for parent in [start_path] + list(start_path.parents):
        if (parent / ".env").exists():
            return parent
    return start_path


PROJECT_ROOT = find_project_root()
load_dotenv(PROJECT_ROOT / ".env")
rel_data_dir = os.getenv("DATA_DIR", default="databse")
abs_data_dir = PROJECT_ROOT / rel_data_dir


class EmotionalState(BaseModel):
    """情绪状态"""

    emotion: Optional[List[str]] = Field(
        None,
        description="当前情绪名称，可包含一到三个情绪。常见情绪：喜悦、悲伤、焦虑、愤怒、恐惧、惊讶、厌恶、平静、疲惫、孤独、羞耻、内疚、希望、迷茫、满足、害羞、安全感、兴奋、失望、感激、爱、恨、嫉妒、自豪、自卑、好奇、无聊、放松、紧张、困惑等",
    )
    intensity: Optional[Literal["潜意识", "弱", "中", "强", "极强"]] = Field(
        None, description="情绪强度"
    )


class Cognition(BaseModel):
    """认知内容"""

    automatic_thought: Optional[str] = Field(None, description="自动想法原文")
    belief: Optional[str] = Field(None, description="深层信念或假设")
    reflection: Optional[str] = Field(None, description="反思、洞见、哲学思考")


class Behavior(BaseModel):
    """行为表现"""

    action: Optional[str] = Field(None, description="具体行为")
    consequence: Optional[str] = Field(None, description="行为后果（内外部）")


class SituationalTag(BaseModel):
    """情境标签（便于结构化筛选）"""

    place: Optional[str] = Field(None, description="地点")
    persons: List[str] = Field(
        default_factory=list, description="涉及人物及角色，如['母亲-权威']"
    )
    scene_type: Optional[
        Literal["工作", "家庭", "亲密关系", "社交", "独处", "学习", "其他"]
    ] = Field(None, description="场景类型")
    event_type: Optional[
        Literal[
            "创伤",
            "积极",
            "重大转折",
            "日常",
            "冲突",
            "成就",
            "失落",
            "反思",
            "情绪抒发",
        ]
    ] = Field(None, description="事件类型")
    # 此处的枚举类型有待check


class DiaryChunk(BaseModel):  # 有待check：对日记的处理模式的改进空间
    """
    日记分析模型 V2
    保留完整原始文本，并提取多维度结构化信息
    """

    # ----- 原始层（完整保留）-----
    raw_text: str = Field(description="原始日记段落（或整篇）")

    # ----- 概览层（快速浏览）-----
    outline: str = Field(description="精炼摘要（一句话概括核心内容）")
    date: str = Field(description="日记标注日期，必须严格按照如'25.03.15'的格式标注")  # pyright: ignore[reportAssignmentType]

    # ----- 内容层（结构化提取）-----
    emotions: EmotionalState = Field(
        default_factory=EmotionalState, description="主要情绪及强度"
    )  # pyright: ignore[reportArgumentType]
    cognitions: Cognition = Field(
        default_factory=Cognition, description="自动想法、信念、反思"
    )  # type: ignore
    behaviors: Behavior = Field(default_factory=Behavior, description="行为与后果")  # type: ignore

    # ----- 情境层（便于检索过滤）-----
    tags: SituationalTag = Field(
        default_factory=SituationalTag, description="结构化标签"
    )  # type: ignore


"""DiaryChunk_schema = {
    "type": "object",
    "properties": {
        "outline": {"type": "string", "description": "事件的摘要"},
        "date": {"type": "string", "description": "事件的日期"},
        "txt": {"type": "string", "description": "事件对应的日记文本"},
        "type": {"type": "string", "description": "事件的类型，如：创伤、日常、积极、重大转折"},
        "emotion": {"type": "string", "description": "事件的主要情绪、次要情绪、情绪强度"}
    },
    "required": ["outline", "date", "txt", "type", "emotion"]
}"""

"""class DiaryChunkbyEvent(BaseModel): # diary_chunker ver 1.0
    date: str = Field(description="信息块的日期")
    txt: str = Field(description="信息块对应的日记原文，不得修改、摘要或做出任何省略，必须原样复制")  # 改成JSON格式
"""


class DiaryChunkbyEvent(BaseModel):  # diary_chunker ver 2.0
    date: str = Field(description="信息块的日期,必须严格按照如'25.03.15'的格式标注")
    start_marker: str = Field(
        description="信息块开头的一段连续文本，约10-20字，必须是原文本的连续子串，"
        "必须从信息块的第一个字符开始，不得对文本进行任何修改、摘要或省略，必须原样复制"
    )
    end_marker: str = Field(
        description="信息块结尾的一段连续文本，约10-20字，必须是原文本的连续子串，"
        "必须以信息块的最后一个字符结束，不得对文本进行任何修改、摘要或省略，必须原样复制"
    )


"""DiaryChunkbyEvent_schema = {
    "type": "object",
    "properties": {
        "date": {"type": "string", "description": "事件的日期"},
        "txt": {"type": "string", "description": "事件对应的日记文本"}
    },
    "required": ["date", "txt"]
}"""

# timeout waiting to be set
diary_chunker = ChatDeepSeek(
    model="deepseek-chat",
    # response_format=DiaryChunkbyEvent_schema,
)


def diary_splitter_date(file_path):  # 待改善：1.非txt， 2.非文本数据， 3.非结构化日期
    diary = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    # 支持多种日期格式：2024-12-01, 2024/12/01, 25.12.21, 25.12.21-26.3.28, 12月1日
    date_pattern = r"\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}|\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{1,2}月\d{1,2}日|\d{2}\.\d{1,2}\.\d{1,2}(?:-\d{2}\.\d{1,2}\.\d{1,2})?"
    dates = re.findall(date_pattern, diary)
    splitted_diary = re.split(date_pattern, diary)
    diary_splitted_date = []

    if not dates:
        # 如果没有找到日期，将整个文件作为一个文档
        diary_splitted_date.append(
            Document(page_content=diary, metadata={"date": "未知日期"})
        )
        print("未找到日期格式，使用整个文件作为单个文档")
    else:
        # 确保分割后的片段数量比日期多一个
        if len(splitted_diary) > len(dates):
            # 第一个分割片段可能是日期前的文本（通常为空）
            for j in range(1, len(splitted_diary)):
                if j - 1 < len(dates):
                    diary_splitted_date.append(
                        Document(
                            page_content=splitted_diary[j],
                            metadata={"date": dates[j - 1]},
                        )
                    )
        else:
            # 回退：将每个日期与对应内容配对
            for j in range(len(dates)):
                content = splitted_diary[j + 1] if j + 1 < len(splitted_diary) else ""
                diary_splitted_date.append(
                    Document(page_content=content, metadata={"date": dates[j]})
                )

    print(f"按日期分割成功，找到 {len(diary_splitted_date)} 个文档")
    return diary_splitted_date


def diary_splitter_event(diary_splitted_date):  # 待改进：异步化处理
    text_chunker = SemanticChunker(OllamaEmbeddings(model="qwen3-embedding:4b"))
    diary_splitted_event = []

    docs = text_chunker.split_documents(diary_splitted_date)

    for k in docs:
        k.page_content += k.metadata["date"]
        diary_splitted_event.append(k)
    print(f"按事件分割成功，得到 {len(diary_splitted_event)} 个事件")
    return diary_splitted_event


"""async def diary_splitter_event(splitted_docs_date):
    chunker_event = diary_chunker.with_structured_output(DiaryChunkbyEvent)
    splitted_docs_event = []
    tasks_event = []
    tasks_structure = []
    chunks = []
    for i in splitted_docs_date:
        prompt_for_structure = [
            {
                "role": "system","content": "你是一个文本结构分析师。任务：分析用户提供的日记片段（属于同一天），识别出其中**宏观的、有意义的段落边界"\
                "划分依据可以是：主题的明显转换、时间或场景的跳跃、情绪或语调的显著变化,输出尽可能精炼而准确文本框架以便后续切分文本使用"\
                "要求：1. 不要过度拆分。每个片段内，输出的单元数量应控制在 2 到 5 个之间。连续论述同一主题的内容必须合并为一个单元。"
            },
            {
                "role":"user","content": i
            }
            
        ]
        task = diary_chunker.ainvoke(prompt_for_structure)
        tasks_structure.append(task)

    structure = await asyncio.gather(*tasks_structure)

    m = 0
    for n in splitted_docs_date:
        prompt_for_event = [
            {
                "role": "system", "content":"你会接收到一段文本和它的整体结构，你需要依照结构将文本逐部分拆分为信息块，"\
                "每个信息块可以是一个事件、一段感受或任何完整的内容单元，并返回拆分后的原文本的开头标记、结尾标记和每段文本的日期。" \
                "你不能舍弃任何原本的日记文本，每个信息块的内容必须是原文本的连续子串。并且抱着所有信息块组合起来可以得到原文本，没有丢失任何信息"
            },
            {
                "role": "user", "content": "以下是文本的整体结构：" + structure[m].content + "\n\n" "以下是文本原文：" + n
            }            
        ]
        task = chunker_event.ainvoke(prompt_for_event)
        tasks_event.append(task)
        m += 1
    
    splitted_docs_event = await asyncio.gather(*tasks_event)


    i = 0
    for j in splitted_docs_date:
    
        pos_start = j.find(splitted_docs_event[i].start_marker)
        pos_end = j.find(splitted_docs_event[i].end_marker) + len(splitted_docs_event[i].end_marker)
        # 错误处理待添加
        chunks.append(splitted_docs_event[i].date + "\n" + j[pos_start:pos_end])
        i += 1

    return {"message": chunks}"""

diary_analysist = ChatDeepSeek(
    model="deepseek-chat",
    # system_prompt="你是一个有深厚心理咨询和精神医学背景的日记分析师，你需要把接收" \
    # "的日记按事件进行拆分，并且对每个事件进行分析，提取出事件的摘要、日期、类型、主要情绪、次要情绪和情绪强度。" ,
    # response_format=DiaryChunk_schema,
)


@tool
def read_file(file_path):
    """A tool to read files of different types, including txt, pdf, md, csv and docx,but only return the first 300 characters"""
    extend = file_path.split(".")[-1].lower()
    # print(extend) # temporary code for testing the file type, to be deleted soon
    loaders = {
        "txt": TextLoader(file_path, encoding="utf-8"),
        "pdf": PyPDFLoader(file_path),
        "md": UnstructuredMarkdownLoader(file_path),
        "csv": CSVLoader(file_path),
        "docx": UnstructuredWordDocumentLoader(file_path),
    }

    if extend not in loaders:
        return ValueError("unsupported file type:" + extend)
    else:
        txt = loaders[extend].load()
        return txt[0].page_content[0:300]


@tool
async def store_diary(file_path: str) -> str:
    """A tool to store the diary into the vector database"""

    """def load_file(file_path):
        extend = file_path.split(".")[-1].lower()
        print(extend) # temporary code for testing the file type, to be deleted soon
        loaders = {
            'txt': TextLoader(file_path, encoding='utf-8'),
            'pdf': PyPDFLoader(file_path),
            'md': UnstructuredMarkdownLoader(file_path),
            'csv': CSVLoader(file_path),
            'docx': UnstructuredWordDocumentLoader(file_path),
        }

        if extend not in loaders:
            return ValueError("unsupported file type:" + extend)
        else:
            return loaders[extend].load()"""

    embeddings = OllamaEmbeddings(model="qwen3-embedding:4b")

    original_diary = Chroma(
        collection_name="original_diary",
        embedding_function=embeddings,
        persist_directory=str(data_path),  # notice the problem of hard coed path
    )

    diary_annotation = Chroma(
        collection_name="diary_annotation",
        embedding_function=embeddings,
        persist_directory=str(data_path),  # notice the problem of hard coed path
    )

    # docs = load_file(file_path)

    """text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 50, # hard codes that wait for test
        separators=["\n", " ", "" "。"]
    )"""  # a choice for a splitter

    split_docs_date = diary_splitter_date(file_path)  # load and split the file
    splitted_docs_event = diary_splitter_event(split_docs_date)

    standardized_docs = []

    original_diary_to_save = []
    diary_annotation_to_save = []

    tasks = []
    analysist = diary_analysist.with_structured_output(DiaryChunk)
    prompt = (
        "你是一个有深厚心理咨询和精神医学背景的日记分析师，你需要对接收的日记片段"
        "从认知内容、行为表现、情景标签三个大方面进行分析，如果存在，则提取出日记片段的自动想法、深层信念、"
        "反思、洞见、具体行为、行为后果、情绪状态（包括情绪名称和强度）以及结构化的情境标签（包括绝对日期、相对时间、地点、涉及人物及角色、场景类型和事件类型）。"
        "如果某个方面的信息在日记片段中没有体现，则填充'无'。请严格按照规则进行分析。"
        "情绪名称可以使用常见的中文情绪词汇，如：喜悦、悲伤、焦虑、愤怒、恐惧、惊讶、厌恶、平静、疲惫、孤独、羞耻、内疚、希望、迷茫、满足、害羞、安全感、兴奋、失望、感激、爱、恨、嫉妒、自豪、自卑、好奇、无聊、放松、紧张、困惑等。"
    )
    for i in splitted_docs_event:
        task = analysist.ainvoke(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": i.page_content},
            ]
        )
        tasks.append(task)

    standardized_docs = await asyncio.gather(*tasks)

    print("diary annotation success")

    for i in standardized_docs:
        datatable = {  # 已修改：根据DiaryChunk的结构调整
            "intensity": i.emotions.intensity,  # 仅记录主要情绪的强度，后续可以考虑记录次要情绪
            "date": i.date,  # 格式为'25.03.15'
            "scene_type": i.tags.scene_type,
            "event_type": i.tags.event_type,
            "emotion": i.emotions.emotion,  # 仅记录主要情绪，后续可以考虑记录次要情绪
        }

        index = hashlib.md5(i.raw_text.encode("utf-8")).hexdigest()

        original_diary_to_save.append(
            Document(
                page_content=i.raw_text,
                id=index,
            )
        )

        # Convert emotions list to string
        emotion_str = ", ".join(i.emotions.emotion) if i.emotions.emotion else ""

        diary_annotation_to_save.append(
            Document(
                page_content="\n".join(
                    [
                        i.outline or "",
                        emotion_str,
                        i.cognitions.automatic_thought or "",
                        i.cognitions.belief or "",
                        i.cognitions.reflection or "",
                        i.behaviors.action or "",
                        i.behaviors.consequence or "",
                        i.tags.place or "",
                        ", ".join(i.tags.persons) if i.tags.persons else "",
                    ]
                ),
                metadata=datatable,
                id=index,
            )
        )

    original_diary.add_documents(original_diary_to_save)  # 向量库批量写入

    diary_annotation.add_documents(diary_annotation_to_save)  # 向量库批量写入
    # a function to check for repetition waiting to be added

    return "Success"


# 发现的问题：概括事件的时候丢失了信息，叙事性不够强的内容被忽视了
