# Memory Counselor Agent

一个基于向量数据库的个人记忆分析与心理辅导代理系统，支持存储和分析日记、学习材料及咨询对话大纲。

## ✨ 功能特性

- **多模态记忆存储**: 支持日记、学习材料、咨询对话大纲等多种类型的记忆存储
- **智能语义分析**: 使用LLM自动分析记忆内容，提取情绪、认知、行为等多维度信息
- **实时心理辅导**: 基于WebSocket的实时聊天界面，提供心理辅导对话
- **记忆检索增强**: 通过向量检索在对话中关联历史记忆，提供个性化回应
- **对话总结管理**: 自动总结长对话，维护上下文窗口
- **文件上传支持**: 支持TXT、PDF、MD、CSV、DOCX等多种文件格式
- **会话管理**: 多会话支持，历史对话可追溯

## 🏗️ 系统架构

### 核心架构
```
用户 ↔ Chatter (实时响应)
         ↑ 提示增强
Analyst (后台分析，触发时注入)
Supervisor (后台监督，触发时注入)
```

### 模块架构
```
memory_manager (中心代理)
├── store_diary        → src/mem_store_diary.py
├── store_materials    → src/mem_store_material.py
└── store_conversation_outline → src/mem_store_conv_outline.py
```

### 数据模型
- **日记分析** (`DiaryChunk`): `emotions`, `cognitions`, `behaviors`, `tags`, `raw_text`
- **对话大纲** (`PAIPOutline`): `problem`, `assessment`, `intervention`, `plan`

### 向量数据库 (Chroma)
- **位置**: `database/` (通过 `.env` 中的 `DATA_DIR` 配置)
- **集合**:
  - `original_diary`: 原始日记文本
  - `diary_annotation`: 结构化日记分析
  - `child_chunks`: 材料分块存储
  - `conv_outline`: 咨询对话PAIP大纲

## 🛠️ 技术栈

### 后端技术
- **LLM框架**: LangChain + DeepSeek API
- **向量数据库**: Chroma (SQLite后端)
- **嵌入模型**: Ollama (qwen3-embedding:4b)
- **Web框架**: FastAPI + WebSocket
- **数据验证**: Pydantic

### 前端技术
- **核心**: 原生HTML/CSS/JavaScript
- **通信**: WebSocket (实时聊天)
- **样式**: CSS变量 + 响应式设计

### 开发工具
- **测试**: pytest + pytest-asyncio
- **代码检查**: ruff
- **环境管理**: python-dotenv

## 📁 项目结构

```
.
├── src/                    # 源代码目录
│   ├── web/               # Web应用
│   │   ├── static/        # 静态文件
│   │   │   ├── index.html # 主界面
│   │   │   ├── upload.html# 文件上传界面
│   │   │   ├── styles.css # 样式表
│   │   │   └── app.js     # 前端逻辑
│   │   ├── main.py        # FastAPI应用
│   │   ├── session_manager.py # 会话管理
│   │   └── requirements.txt # Web依赖
│   ├── analysist.py       # 分析师模块
│   ├── chatter.py         # 聊天器模块
│   ├── conversation_manager.py # 对话管理
│   ├── mem_collections.py # 向量集合定义
│   ├── mem_retrieve_*.py  # 记忆检索模块
│   ├── mem_store_*.py     # 记忆存储模块
│   ├── mem_store_module.py # 记忆管理代理
│   ├── read_file.py       # 文件读取工具
│   ├── supervisoner.py    # 督导模块
│   └── top_module.py      # 顶层协调模块
├── test/                  # 测试目录
│   ├── test_analysist.py
│   ├── test_chatter.py
│   ├── test_conversation_manager.py
│   ├── test_mem_retrieve_*.py
│   ├── test_mem_store_*.py
│   ├── test_supervisoner.py
│   └── test_top_module.py
├── database/              # 向量数据库存储
├── web/sessions/          # 会话JSON存储
├── .env                   # 环境配置
└── README.md              # 本文档
```

## 🚀 快速开始

### 系统要求

- **Python**: 3.9 或更高版本
- **内存**: 至少 8GB RAM (推荐16GB用于向量操作)
- **存储**: 至少 2GB 可用空间
- **网络**: 可访问DeepSeek API (或本地LLM服务)

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd Counselor-Agent-main

# 创建虚拟环境 (推荐使用Python 3.9+)
python -m venv .venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 安装项目依赖
pip install -r requirements.txt

# 安装Web额外依赖
pip install fastapi uvicorn[standard] websockets
```

### 2. 配置环境变量

创建 `.env` 文件:
```env
# 数据库配置
DATA_DIR = database

# Web服务配置
WEB_HOST = 0.0.0.0
WEB_PORT = 8000

# LLM API配置 (根据需要)
# DEEPSEEK_API_KEY = your_deepseek_api_key_here
# OPENAI_API_KEY = your_openai_api_key_here
# OLLAMA_BASE_URL = http://localhost:11434

# LangSmith追踪 (可选)
# LANGCHAIN_TRACING_V2 = true
# LANGCHAIN_API_KEY = your_langsmith_api_key
# LANGCHAIN_PROJECT = counselor-agent-demo
```

### 3. 安装和配置外部服务

#### Ollama (用于本地嵌入)
```bash
# 安装Ollama (参考 https://ollama.com/)
# 拉取嵌入模型
ollama pull qwen3-embedding:4b

# 启动Ollama服务 (如果未自动启动)
ollama serve
```

#### DeepSeek API (可选，用于LLM)
- 注册并获取API密钥: https://platform.deepseek.com/
- 设置环境变量: `DEEPSEEK_API_KEY=your_key_here`
- 或使用其他兼容的LangChain LLM集成

### 4. 运行应用

```bash
# 切换到src目录
cd src

# 启动Web应用
python -m web.main

# 或直接运行
python web/main.py
```

### 5. 访问应用

- **聊天界面**: http://localhost:8000
- **文件上传**: http://localhost:8000/upload
- **API文档**: http://localhost:8000/docs

## 📖 详细使用指南

### 记忆存储

系统支持三种类型的记忆存储：

1. **日记存储**: 自动按日期分割 → 语义分块 → LLM分析 → 存储原始文本+结构化标注
2. **材料存储**: 类型推断 → 父子语义分块 → 存储到 `child_chunks` 集合
3. **对话大纲存储**: 生成PAIP摘要 → 存储原始分块+PAIP章节

### 文件上传

通过上传页面或API支持以下格式：
- `.txt` - 纯文本文件
- `.pdf` - PDF文档
- `.md` - Markdown文件
- `.csv` - CSV表格
- `.docx` - Word文档

### 实时聊天

- **WebSocket连接**: `/ws/chat?session_id={session_id}`
- **消息类型**: `message`, `ping`, `pong`
- **上下文管理**: 自动总结长对话，维护token限制

### API接口

#### HTTP API
- `GET /api/sessions` - 列出所有会话
- `POST /api/sessions` - 创建新会话
- `GET /api/sessions/{id}` - 获取会话详情
- `DELETE /api/sessions/{id}` - 删除会话
- `PATCH /api/sessions/{id}/title` - 更新会话标题
- `POST /api/sessions/{id}/clear` - 清空会话消息
- `POST /api/upload` - 上传记忆文件

#### WebSocket API
- `ws://localhost:8000/ws/chat` - 实时聊天连接
- 消息格式: `{"type": "message", "content": "用户输入"}`

### 使用示例

#### 1. 通过API上传日记文件
```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@my_diary.txt"
```

#### 2. 创建新的咨询会话
```bash
curl -X POST http://localhost:8000/api/sessions \
  -H "Content-Type: application/json" \
  -d '{"title": "关于工作压力的咨询"}'
```

#### 3. WebSocket聊天示例 (JavaScript)
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/chat');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'message') {
    console.log('助手:', data.content);
  }
};

// 发送用户消息
ws.send(JSON.stringify({
  type: 'message',
  content: '最近工作压力很大，睡不着觉'
}));
```

#### 4. Python客户端示例
```python
import asyncio
import websockets
import json

async def chat_example():
    async with websockets.connect('ws://localhost:8000/ws/chat') as ws:
        # 发送消息
        await ws.send(json.dumps({
            'type': 'message',
            'content': '我感到焦虑不安'
        }))
        
        # 接收响应
        response = await ws.recv()
        print('Response:', response)

asyncio.run(chat_example())
```

## 🔧 开发指南

### 运行测试

```bash
# 单元测试 (模拟外部依赖)
pytest test/test_mem_store_material.py -v
pytest test/test_mem_store_conv_outline.py -v

# 集成测试 (需要Ollama + DeepSeek服务)
pytest test/test_mem_store_material.py -v -m integration
pytest test/test_mem_store_conv_outline.py -v -m integration
```

### 代码规范

- **导入顺序**: 标准库 → 第三方库 → 本地模块
- **异步模式**: 使用 `asyncio.get_running_loop().run_in_executor()` 处理同步Chroma操作
- **ID生成**: `UniqueIDGenerator` 生成格式: `{prefix}_YYYYMMDD_HHMMSS_mmm_seq`
- **工具装饰器**: 存储函数使用 `@tool` 装饰器

### 添加新模块

1. 在 `src/` 目录下创建新模块
2. 遵循现有模式使用 `@tool` 装饰器
3. 在 `mem_store_module.py` 中注册到 `memory_manager`
4. 添加相应的测试文件

## 🧪 测试策略

### 测试层级
- **单元测试**: 模拟外部依赖，快速验证逻辑
- **集成测试**: 依赖真实服务，验证端到端流程
- **标记系统**: 使用 `@pytest.mark.integration` 标记集成测试

### 测试覆盖
- 记忆存储/检索功能
- 对话管理和总结
- 代理协调逻辑
- Web API接口

## 🔍 故障排除

### 常见问题

1. **端口占用**
   ```bash
   netstat -ano | findstr :8000
   taskkill /PID <PID> /F
   ```

2. **Ollama服务未启动**
   ```bash
   ollama serve
   ```

3. **导入错误**
   - 确保在 `src/` 目录下运行
   - 检查Python路径配置

4. **WebSocket连接失败**
   - 检查防火墙设置
   - 验证WebSocket协议支持

5. **依赖安装失败**
   ```bash
   # 尝试升级pip
   python -m pip install --upgrade pip
   
   # 分步安装依赖
   pip install langchain langchain-chroma langchain-ollama
   pip install langchain-deepseek langchain-community
   pip install fastapi uvicorn[standard] websockets
   
   # 如果遇到特定包问题，尝试指定版本
   pip install chromadb==0.4.22
   ```

### 日志查看

应用启动日志显示配置信息和服务状态：
```
INFO: Uvicorn running on http://0.0.0.0:8000
INFO: Application startup complete
```

## 🤝 贡献指南

1. **分支策略**: 功能分支 → 开发测试 → PR到main分支
2. **提交信息**: 遵循Conventional Commits规范
3. **代码审查**: 所有PR需要至少一个审核者
4. **测试要求**: 新功能需包含单元测试

### 开发流程
```bash
# 1. 创建功能分支
git checkout -b feature/新功能

# 2. 开发与测试
# 3. 提交更改
git add .
git commit -m "feat: 添加新功能"

# 4. 推送并创建PR
git push origin feature/新功能
```

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [LangChain](https://github.com/langchain-ai/langchain) - LLM应用框架
- [Chroma](https://github.com/chroma-core/chroma) - 向量数据库
- [Ollama](https://github.com/ollama/ollama) - 本地LLM服务
- [FastAPI](https://github.com/tiangolo/fastapi) - 现代Web框架

---

**提示**: 生产环境部署请确保配置适当的安全措施，包括HTTPS、身份验证和速率限制。