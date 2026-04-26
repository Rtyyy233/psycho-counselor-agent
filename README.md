# 心理咨询师代理系统 (Counselor Agent)

一个基于 **3-Agent 观察者模式** 的心理咨询 AI 代理系统。采用异步事件驱动核心，集成实时对话、后台分析、专业督导和长期记忆管理。

## 功能特性

- **多 Agent 协作**: Chatter（前台咨询师）、Analyst（后台分析师）、Supervisor/Supervisoner（后台督导）三 Agent 协同
- **长期记忆管理**: 支持日记、学习材料和对话大纲的存储与语义检索（ChromaDB + Ollama 嵌入）
- **LangGraph 检索图**: 三个检索模块各自使用 LangGraph 状态机构建多步检索流水线（LLM Planner → 路由分发 → 多模式搜索）
- **结构化咨询**: 基于 PAIP（问题-评估-干预-计划）模型的对话摘要存储
- **集中化模型配置**: 统一通过 `config.py` 管理 LLM 模型名，单点修改

## 系统架构

```
                 ┌─────────────────────────────────────┐
                 │              用户 (CLI)              │
                 └────────────┬────────────────────────┘
                              │ 消息
                              ▼
┌──────────────────────────────────────────────────────────┐
│                    SharedContext                          │
│  (线程安全异步上下文 · 消息管理 · PromptInjection 合并)    │
└────┬────────────┬──────────────────────┬─────────────────┘
     │ 用户消息    │ analyst_trigger      │ supervisor_trigger
     ▼            ▼                      ▼
┌──────────┐ ┌──────────────┐ ┌──────────────────┐
│ Chatter  │ │  Analyst     │ │  Supervisor       │
│ (前台咨询) │ │  (后台分析)   │ │  (后台督导)        │
│          │ │  检索日记/    │ │  阅读/存储文件     │
│ EFT共情   │ │  材料/对话    │ │  评估对话质量      │
│ 不暴露架构│ │  →分析注入    │ │  →督导注入        │
└──────────┘ └──────────────┘ └──────────────────┘
```

### 核心循环

1. 用户发送消息 → `SharedContext.add_message()` 追加消息并触发 `analyst_trigger` + `supervisor_trigger` 事件
2. 后台任务 `call_analysist`、`call_supervisor` 异步启动——监听触发信号，检查最近消息，设置 `PromptInjection`
3. `chatter.ainvoke()` 同步执行（接收注入内容附加到提示词），生成最终回复
4. 回复加入上下文，循环继续

## 模块说明

### 三 Agent

| Agent | 文件 | 职责 | 访问权限 |
|-------|------|------|----------|
| **Chatter** | `chatter.py` | 前台咨询师。EFT 专家风格，温暖共情，绝不暴露多 Agent 架构 | `read_file_tool` |
| **Analyst** | `analysist.py` | 后台分析师。`ToolStrategy` + `analysis` 输出模型（state + injection），并发锁防止重入 | 日记/材料/对话大纲三个检索工具 |
| **Supervisor** | `supervisor.py` | 后台督导（CLI 使用）。`ToolStrategy` + `supervision` 输出模型 | 读文件 + 存储日记/材料 |
| **Supervisoner** | `supervisoner.py` | 后台督导。`with_structured_output(SupervisorResult)` + 启发式回退 | — |

### 记忆系统

三种数据类型，每种有独立的存储模块和检索模块：

| 类型 | 存储模块 | 检索模块 | Chroma 集合 |
|------|----------|----------|-------------|
| **日记** | `mem_store_diary.py` — 日期拆分 + 语义分块 + LLM 提取情绪/认知/行为/场景标签 | `mem_retrieve_diary.py` — LangGraph 状态机（planner → 路由 → 搜索节点） | `original_diary`, `diary_annotation` |
| **材料** | `mem_store_material.py` — 类型推断 + 父子语义分块 | `mem_retrieve_material.py` — LangGraph 状态机（semantic → parent_lookup small-to-big） | `child_chunks`, `parent_chunks` |
| **对话大纲** | `mem_store_conv_outline.py` — PAIP 提取 + 父子分块 | `mem_retrieve_conv_outline.py` — LangGraph 状态机（semantic → paip_outline_lookup） | `conv_outline` |

三个检索图共享相同的控制流模式：**LLM Planner 生成多步计划 → `route_dispatch` 根据 `current_step_idx` 路由 → 执行节点 → `after_execution` 步进 → 循环/结束**。

### 配置管理

集中化在 `config.py`：

- 模块级 `load_dotenv()` 在读取任何环境变量前加载 `.env`
- 所有模块统一引用 `config.LLM_MODEL`，不再各自 `os.getenv`
- 支持 `.env` 文件统一管理模型名、API 密钥、LangSmith 等配置

## 快速开始

### 环境要求

- Python 3.9+
- [Ollama](https://ollama.ai/)（本地嵌入模型）
- DeepSeek API 密钥

### 安装步骤

1. **克隆仓库**
   ```bash
   git clone <repository-url>
   cd Counselor-Agent-main
   ```

2. **创建虚拟环境**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **配置环境变量**
   编辑 `.env` 文件，填入必要的 API 密钥：
   ```env
   DEEPSEEK_API_KEY=your_deepseek_api_key
   LLM_MODEL=deepseek-chat
   DATA_DIR=database
   ```

5. **启动 Ollama 服务并拉取嵌入模型**
   ```bash
   ollama serve
   # 另开终端
   ollama pull qwen3-embedding:4b
   ```

6. **启动应用**

   ```bash
   cd src
   python user_interface.py
   ```

## 使用方法

### 实时对话
- 在聊天界面中输入消息开始对话
- 系统自动在后台触发分析和督导，注入相关记忆和指导建议

### 会话管理
- **加载会话**: 输入 `/load <会话ID>` 加载历史对话
- **自动上下文压缩**: 接近令牌限制时自动摘要并重置上下文

## 配置说明

### 环境变量 (.env)
```env
DATA_DIR=database                    # 数据目录
LLM_MODEL=deepseek-chat              # LLM 模型名（统一入口）
DEEPSEEK_API_KEY=your_deepseek_key   # DeepSeek API 密钥
LANGSMITH_TRACING=true               # LangSmith 追踪开关
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=main
```

### 向量数据库
- **后端**: ChromaDB（SQLite）
- **嵌入模型**: `qwen3-embedding:4b`（通过 Ollama）
- **集合**: `original_diary`, `diary_annotation`, `child_chunks`, `parent_chunks`, `conv_outline`

## 项目结构

```
Counselor-Agent-main/
├── src/                        # 源代码
│   ├── config.py              # 集中化配置管理
│   ├── SharedContext.py       # 线程安全异步上下文
│   ├── user_interface.py      # CLI 主入口
│   ├── session_manager.py     # JSON 会话持久化
│   ├── conversation_manager.py# 自动摘要的对话管理
│   ├── chatter.py             # 前台咨询 Agent
│   ├── analysist.py           # 后台分析 Agent
│   ├── supervisor.py          # 后台督导 Agent (ToolStrategy)
│   ├── supervisoner.py        # 后台督导 (structured output)
│   ├── mem_integration.py     # Chroma 初始化 + 嵌入 + 工具定义
│   ├── mem_store_diary.py     # 日记存储
│   ├── mem_store_material.py  # 材料存储
│   ├── mem_store_conv_outline.py # 对话大纲存储
│   ├── mem_retrieve_diary.py  # 日记检索 (LangGraph)
│   ├── mem_retrieve_material.py # 材料检索 (LangGraph)
│   ├── mem_retrieve_conv_outline.py # 对话大纲检索 (LangGraph)
│   ├── read_file.py           # 多格式文件读取
├── database/                   # Chroma 数据（gitignored）
├── test/                       # pytest 测试文件
├── .env                        # 环境变量配置
├── requirements.txt            # Python 依赖
└── CLAUDE.md                   # 项目文档（给 Claude Code 使用）
```

## 开发指南

### 运行测试
```bash
cd src
# 全部测试
python -m pytest ../test/ -v
# 单个测试文件
python -m pytest ../test/test_analysist.py -v
# 单个测试用例
python -m pytest ../test/test_analysist.py::test_xxx -v
```

### 代码风格
```bash
cd src
ruff check .        # 检查
ruff format .       # 格式化
```

### 添加新检索逻辑
1. 在 `mem_integration.py` 中定义新的 Chroma 集合
2. 实现 `mem_store_*.py` 存储模块
3. 实现 `mem_retrieve_*.py` 检索模块（可复用 LangGraph 状态机模板）
4. 在 `analysist.py` 中注册新的检索工具

### 更换 LLM 模型
只需修改 `.env` 中的 `LLM_MODEL`，所有模块自动生效：
```env
LLM_MODEL=deepseek-chat
```

## 故障排除

### 常见问题
1. **"deepseek-reasoner does not support this tool_choice"**
   - 确保 `.env` 中 `LLM_MODEL=deepseek-chat` 而非 `deepseek-reasoner` 或旧版 `deepseek-v4-flash`
   - 清除 `__pycache__` 后重启

2. **记忆检索无结果**
   - 确认已上传相关文件
   - 检查向量数据库和 Ollama 嵌入服务状态

## 许可证

MIT License

---

*最后更新: 2026-04-26*
