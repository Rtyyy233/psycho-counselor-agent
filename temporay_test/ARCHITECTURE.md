# 心理咨询师代理系统架构文档

## 系统概述

心理咨询师代理是一个基于人工智能的个人记忆分析与心理辅导系统。系统采用异步观察者模式，结合向量数据库和大型语言模型，提供实时的心理辅导对话和个性化的记忆分析。

### 核心设计理念

- **观察者模式架构**: Chatter（前台响应代理）+ Analyst（后台分析代理）+ Supervisor（后台督导代理）
- **实时与异步分离**: 用户立即获得响应，分析在后台异步进行
- **记忆增强对话**: 向量检索关联历史记忆，提供个性化回应
- **多模态存储**: 支持日记、学习材料、咨询对话大纲等多种记忆类型

## 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      用户界面层                              │
│  ┌─────────────┐    ┌─────────────┐    ┌───────────────┐  │
│  │   Web聊天界面 │    │  文件上传界面 │    │  会话管理界面  │  │
│  └──────┬──────┘    └──────┬──────┘    └──────┬────────┘  │
│         │                  │                   │           │
└─────────┼──────────────────┼───────────────────┼───────────┘
          │                  │                   │
          ▼                  ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│                    API层 (FastAPI)                          │
│  ┌─────────────┐    ┌─────────────┐    ┌───────────────┐  │
│  │ WebSocket聊天 │    │ 文件上传API │    │  会话管理API  │  │
│  │  /ws/chat    │    │  /api/upload │    │ /api/sessions│  │
│  └──────┬──────┘    └──────┬──────┘    └──────┬────────┘  │
└─────────┼──────────────────┼───────────────────┼───────────┘
          │                  │                   │
          ▼                  ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│                    业务逻辑层                                │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              观察者模式协调器                         │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────────┐        │  │
│  │  │  Chatter │◄─┤  Analyst │  │  Supervisor │        │  │
│  │  │ (前台响应)│  │(后台分析)│  │(后台督导)   │        │  │
│  │  └────┬────┘  └────┬────┘  └─────┬───────┘        │  │
│  └───────┼────────────┼──────────────┼────────────────┘  │
│          │            │               │                   │
│          ▼            ▼               ▼                   │
│  ┌─────────────┐┌─────────────┐┌───────────────┐        │
│  │  记忆检索系统 ││  记忆存储系统 ││  对话管理系统  │        │
│  └──────┬──────┘└──────┬──────┘└──────┬────────┘        │
└─────────┼──────────────┼───────────────┼─────────────────┘
          │              │               │
          ▼              ▼               ▼
┌─────────────────────────────────────────────────────────────┐
│                    数据访问层                                │
│  ┌─────────────────────────────────────────────────────┐  │
│  │               向量数据库 (Chroma)                     │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐│  │
│  │  │ original_diary│  │ diary_annotation │  │ child_chunks ││  │
│  │  │  原始日记     │  │ 结构化标注     │  │ 材料分块    ││  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘│  │
│  │  ┌─────────────┐                                   │  │
│  │  │ conv_outline │                                   │  │
│  │  │ 对话大纲     │                                   │  │
│  │  └─────────────┘                                   │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 核心组件详解

### 1. 顶层协调器 (`src/top_module.py`)

**职责**: 协调所有代理的交互，实现观察者模式
**关键类**:
- `SharedContext`: 线程安全的共享上下文，包含消息、注入和控制事件
- `PsychologicalCounselor`: 主协调器，管理对话循环和观察者任务
- `analyst_observer`: 后台分析师观察者，监听触发并注入分析结果
- `supervisor_observer`: 后台督导观察者，监听触发并注入指导建议

**数据流**:
1. 用户消息 → `SharedContext.add_message()`
2. 设置触发事件 → `ctx.on_analyst_trigger.set()` 和 `ctx.on_supervisor_trigger.set()`
3. 观察者异步处理 → 生成分析/督导注入
4. Chatter读取注入 → 生成增强回复

### 2. Chatter代理 (`src/chatter.py`)

**职责**: 提供温暖、共情的实时对话响应
**关键功能**:
- 基础系统提示：定义心理咨询师角色特性
- 注入处理：读取Analyst和Supervisor的注入并融入回复
- 上下文管理：维护最近10条消息的对话历史
- 结构化输出：使用`ChatterResponse` Pydantic模型确保响应格式

**注入机制**:
```python
# 检查并应用注入
if analyst_injection:
    enriched_prompt += f"\n\n[分析洞察]\n{analyst_injection.content}"
    ctx.analyst_injection = None  # 使用后清除

if supervisor_injection:
    enriched_prompt += f"\n\n[指导建议]\n{supervisor_injection.content}"
    ctx.supervisor_injection = None
```

### 3. Analyst代理 (`src/analysist.py`)

**职责**: 深度分析用户对话，关联历史记忆，提供洞察
**关键功能**:
- 多源检索：并行检索日记、学习材料、对话大纲
- 综合分析：使用LLM合成检索结果，生成有洞察的分析
- 触发机制：基于最近对话内容自动触发
- 错误处理：单源检索失败不影响整体分析

**检索流程**:
1. 从最近用户消息构建查询
2. 并行检索三个记忆源：`retrieve_diary`, `retrieve_materials`, `retrieve_conv_outline`
3. 使用`synthesize_analysis`函数综合结果
4. 返回分析洞察供Chatter注入

### 4. Supervisor代理 (`src/supervisoner.py`)

**职责**: 监控对话质量，提供专业督导建议
**关键功能**:
- 对话状态分析：评估当前对话进展和质量
- 专业指导：基于咨询理论提供干预建议
- 适度干预：仅在需要时提供指导，避免过度干预
- 优先级管理：区分不同重要程度的指导建议

### 5. 记忆存储系统

#### 5.1 日记存储 (`src/mem_store_diary.py`)
**功能**: 处理日记文件，提取结构化信息，存储到向量数据库
**处理流程**:
1. 文件加载 → 支持TXT、PDF、MD、CSV、DOCX格式
2. 按日期分割 → 识别日记中的日期条目
3. 语义分块 → 使用`SemanticChunker`保持语义完整性
4. LLM分析 → 提取情绪、认知、行为、场景等多维度信息
5. 向量存储 → 存储到`original_diary`和`diary_annotation`集合

**数据模型**:
- `EmotionalState`: 情绪状态（扩展至30+中文情绪词）
- `DiaryChunk`: 日记分块，包含原始文本和结构化标注
- `EmotionType`: 情绪类型Literal，支持30种常见中文情绪

#### 5.2 材料存储 (`src/mem_store_material.py`)
**功能**: 处理学习材料，创建父子分块结构
**特点**:
- 类型推断：基于内容识别材料类型（文章、论文、报告等）
- 层次分块：父分块包含整体概要，子分块包含详细内容
- 语义关联：保持分块间的语义连贯性

#### 5.3 对话大纲存储 (`src/mem_store_conv_outline.py`)
**功能**: 处理咨询对话，生成PAIP（问题-评估-干预-计划）大纲
**PAIP模型**:
- Problem: 识别核心问题
- Assessment: 多维评估
- Intervention: 干预策略
- Plan: 后续计划

### 6. 记忆检索系统

#### 6.1 日记检索 (`src/mem_retrieve_diary.py`)
**功能**: 基于多维度过滤条件检索相关日记
**过滤维度**:
- 情绪强度：潜意识、弱、中、强、极强
- 时间范围：支持日期区间检索
- 场景类型：工作、家庭、亲密关系、社交等
- 事件类型：创伤、积极、重大转折、日常等
- 情绪名称：支持30+中文情绪词的多选检索

**检索架构**:
- 使用LangGraph实现多步骤检索流程
- 支持在`original_diary`和`diary_annotation`集合间切换
- 结合语义搜索和过滤条件

#### 6.2 材料检索 (`src/mem_retrieve_material.py`)
**功能**: 检索相关学习材料，支持父级查找
**特点**:
- 语义搜索：基于查询内容查找相关材料分块
- 父级关联：找到相关子分块后，定位其父分块获取上下文
- 类型过滤：支持按材料类型筛选

#### 6.3 对话大纲检索 (`src/mem_retrieve_conv_outline.py`)
**功能**: 检索相关咨询对话PAIP大纲
**价值**: 为当前咨询提供历史案例参考和结构化框架

### 7. Web应用层 (`src/web/`)

#### 7.1 主应用 (`src/web/main.py`)
**架构**:
- FastAPI应用：提供HTTP和WebSocket接口
- 环境配置：优先加载环境变量，确保LangChain正确初始化
- 依赖注入：单例模式的会话管理器
- 错误处理：统一的异常处理中间件

**关键端点**:
- `GET /`: 主聊天界面
- `GET /upload`: 文件上传界面
- `POST /api/upload`: 文件上传API
- `WebSocket /ws/chat`: 实时聊天WebSocket连接
- `GET /api/sessions`: 会话管理API

#### 7.2 会话管理 (`src/web/session_manager.py`)
**功能**: 管理用户会话的创建、读取、更新、删除
**存储**: JSON文件存储，支持持久化和快速恢复
**特性**:
- 自动标题生成：基于第一条消息生成会话标题
- 消息历史：完整记录对话历史
- 会话摘要：可选的长对话摘要

#### 7.3 静态文件 (`src/web/static/`)
- `index.html`: 主聊天界面，基于WebSocket的实时通信
- `upload.html`: 文件上传界面，支持拖拽和进度显示
- `app.js`: 前端逻辑，处理WebSocket通信和UI交互
- `styles.css`: 响应式CSS样式

### 8. 工具与工具类

#### 8.1 工具工具类 (`src/tool_utils.py`)
**功能**: 统一管理工具调用，提供超时、重试、进度反馈
**特性**:
- 异步工具调用：`call_tool_async`函数
- 超时处理：可配置的超时时间
- 重试机制：失败自动重试
- 进度反馈：实时更新操作进度

#### 8.2 文件处理 (`src/read_file.py`)
**功能**: 多格式文件读取，统一文本提取
**支持格式**: TXT、PDF、MD、CSV、DOCX

#### 8.3 存储服务 (`src/storage_service.py`)
**功能**: 文件存储服务，处理临时文件和持久化存储

### 9. 配置管理 (`src/config.py`)
**集中化管理**:
- 文件上传配置：大小限制、允许格式、MIME类型
- 超时配置：存储、分析、工具调用、WebSocket超时
- 向量数据库配置：Chroma路径、嵌入模型
- LLM配置：DeepSeek API、LangSmith追踪
- Web服务器配置：主机、端口、调试模式
- 文件类型检测：关键词模式、日期模式

## 数据流详解

### 实时对话流程

```
1. 用户发送消息
   ↓
2. WebSocket接收 → handle_user_message()
   ↓
3. 添加到会话存储 → session_manager.add_message()
   ↓
4. 添加到共享上下文 → ctx.ctx.messages.append()
   ↓
5. 触发观察者 → ctx.ctx.on_analyst_trigger.set()
   ↓
6. 立即生成响应 → call_chatter(ctx.ctx)
   ↓
7. 响应返回用户 ← send_json(websocket, ...)
   ↓
8. 后台观察者处理 (并行)
   ├─ Analyst: 检索记忆 → 生成分析 → 注入上下文
   └─ Supervisor: 评估对话 → 生成指导 → 注入上下文
```

### 文件上传与分析流程

```
1. 用户上传文件
   ↓
2. 文件类型检测 → detect_file_type()
   ↓
3. 临时存储 → tempfile.NamedTemporaryFile
   ↓
4. 内容分析 → 检测关键词和日期模式
   ↓
5. 路由到对应存储模块
   ├─ 日记 → store_diary()
   ├─ 材料 → store_materials()
   └─ 对话大纲 → store_conversation_outline()
   ↓
6. 处理结果返回用户
```

### 记忆检索流程

```
1. Analyst触发检索
   ↓
2. 构建查询 → 最近用户消息
   ↓
3. 并行检索
   ├─ 日记检索 → retrieve_diary()
   ├─ 材料检索 → retrieve_materials()
   └─ 对话大纲检索 → retrieve_conv_outline()
   ↓
4. 结果综合 → synthesize_analysis()
   ↓
5. 生成分析洞察
   ↓
6. 注入到SharedContext → ctx.safe_set_analyst()
```

## 数据结构

### 向量数据库集合

| 集合名称 | 存储内容 | 用途 |
|---------|---------|------|
| `original_diary` | 原始日记文本分块 | 语义搜索原始内容 |
| `diary_annotation` | 结构化日记分析 | 基于情绪的过滤和检索 |
| `child_chunks` | 学习材料子分块 | 详细内容检索 |
| `conv_outline` | 咨询对话PAIP大纲 | 案例参考和框架指导 |

### 数据模型

#### EmotionalState (`src/mem_store_diary.py`)
```python
class EmotionalState(BaseModel):
    emotion: List[str]  # 情绪名称列表，支持30+中文情绪词
    intensity: Literal["潜意识", "弱", "中", "强", "极强"]
    triggers: List[str]  # 触发因素
    physical_sensations: List[str]  # 身体感受
    duration_minutes: Optional[int]  # 持续时间
```

#### DiaryChunk (`src/mem_store_diary.py`)
```python
class DiaryChunk(BaseModel):
    raw_text: str  # 原始文本
    date: Optional[str]  # 日期
    emotions: List[EmotionalState]  # 情绪状态
    cognitions: List[str]  # 认知内容
    behaviors: List[str]  # 行为表现
    scene: Scene  # 场景信息
    tags: List[str]  # 标签
```

#### PAIPOutline (`src/mem_store_conv_outline.py`)
```python
class PAIPOutline(BaseModel):
    problem: str  # 问题描述
    assessment: Assessment  # 多维评估
    intervention: List[InterventionStep]  # 干预步骤
    plan: Plan  # 后续计划
```

### 共享上下文结构

```python
class SharedContext:
    messages: List[ChatMessage]  # 对话消息
    current_topic: str  # 当前话题
    topic_history: List[str]  # 话题历史
    
    # 注入机制
    analyst_injection: Optional[PromptInjection]
    supervisor_injection: Optional[PromptInjection]
    
    # 控制事件
    on_new_message: asyncio.Event
    on_analyst_trigger: asyncio.Event
    on_supervisor_trigger: asyncio.Event
    
    # 线程安全锁
    _lock: asyncio.Lock
```

## 部署架构

### 开发环境
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  前端浏览器      │    │  FastAPI服务器   │    │  Ollama服务     │
│  (localhost)    │◄──►│  (localhost:8000)│◄──►│  (localhost:11434)│
└─────────────────┘    └────────┬────────┘    └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  Chroma数据库    │
                        │  (SQLite后端)    │
                        └─────────────────┘
```

### 生产环境建议
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   负载均衡器     │    │  FastAPI集群     │    │  Ollama集群     │
│   (Nginx)       │◄──►│  (多实例)        │◄──►│  (多GPU节点)    │
└─────────────────┘    └────────┬────────┘    └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │   PostgreSQL    │
                        │  + pgvector     │
                        └─────────────────┘
```

## 关键技术栈

### 后端技术
- **LLM框架**: LangChain + LangGraph
- **向量数据库**: Chroma (SQLite后端，生产建议使用pgvector)
- **嵌入模型**: Ollama qwen3-embedding:4b (本地) / 其他云服务
- **LLM服务**: DeepSeek API (默认) / 支持OpenAI兼容API
- **Web框架**: FastAPI + WebSocket
- **异步框架**: asyncio
- **数据验证**: Pydantic v2

### 前端技术
- **核心**: 原生HTML5/CSS3/ES6
- **通信**: WebSocket原生API
- **样式**: CSS变量 + Flexbox + 响应式设计
- **构建**: 无构建步骤，直接使用

### 开发与运维
- **环境管理**: python-dotenv
- **代码检查**: ruff
- **测试**: pytest + pytest-asyncio
- **部署**: Uvicorn (开发) / Gunicorn + Uvicorn (生产)
- **监控**: LangSmith (可选)

## 配置说明

### 环境变量 (.env)
```env
# 必需配置
DATA_DIR=database  # 向量数据库存储路径

# Web服务配置
WEB_HOST=0.0.0.0
WEB_PORT=8000

# LLM配置 (至少配置一个)
DEEPSEEK_API_KEY=your_deepseek_api_key
# 或使用本地Ollama
OLLAMA_BASE_URL=http://localhost:11434

# 可选：LangSmith追踪
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=counselor-agent-demo
```

### 关键配置常量 (`src/config.py`)
- `MAX_FILE_SIZE`: 10MB (文件上传限制)
- `STORAGE_TIMEOUT`: 60秒 (存储操作超时)
- `ANALYSIS_TIMEOUT`: 120秒 (分析操作超时)
- `SESSION_TIMEOUT`: 3600秒 (会话超时)
- `EMBEDDING_MODEL`: "qwen3-embedding:4b" (嵌入模型)

## 扩展性与维护

### 添加新记忆类型
1. 创建存储模块: `src/mem_store_newtype.py`
2. 创建检索模块: `src/mem_retrieve_newtype.py`
3. 在`analysist.py`中集成检索
4. 在`config.py`中添加类型检测关键词
5. 在`web/main.py`的文件路由中添加支持

### 添加新代理
1. 创建代理模块: `src/new_agent.py`
2. 在`top_module.py`中添加观察者函数
3. 在`SharedContext`中添加对应的注入字段和触发事件
4. 在`chatter.py`中处理新代理的注入

### 性能优化建议
1. **向量检索**: 考虑使用HNSW索引提高检索速度
2. **缓存机制**: 对常见查询结果添加缓存
3. **批处理**: 对多个文件上传使用批处理
4. **异步优化**: 使用连接池管理数据库连接

## 故障排除

### 常见问题
1. **WebSocket连接失败**: 检查防火墙和CORS设置
2. **文件上传失败**: 检查文件大小和格式限制
3. **向量检索慢**: 检查Ollama服务状态和网络连接
4. **LLM调用失败**: 检查API密钥和网络连通性

### 调试建议
1. 启用LangSmith追踪分析LLM调用链
2. 检查Chroma数据库日志
3. 监控Ollama服务资源使用
4. 使用`test_websocket_fix.py`等测试脚本验证核心功能

## 安全考虑

### 数据安全
1. **本地存储**: 默认使用本地SQLite数据库，数据不离开用户环境
2. **文件隔离**: 用户上传文件在临时目录处理，处理完成后清理
3. **会话隔离**: 不同会话间数据完全隔离

### API安全
1. **CORS配置**: 生产环境应限制允许的源
2. **速率限制**: 建议添加API速率限制
3. **认证授权**: 生产环境应添加用户认证

### 模型安全
1. **提示注入防护**: 系统提示设计考虑了角色隔离
2. **输出过滤**: Pydantic模型验证确保结构化输出
3. **错误处理**: 完善的异常处理避免信息泄露

---

*文档版本: 1.0*
*最后更新: 2025-04-18*
*对应代码版本: 修复了观察者任务启动和情绪类型验证后的稳定版本*