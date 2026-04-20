# 心理咨询师代理系统 (Counselor Agent)

一个基于观察者模式的心理咨询AI代理系统，集成了实时对话、背景分析、专业督导和长期记忆管理。

## 功能特性

- **多代理协作**: Chatter（前台对话）、Analyst（后台分析）、Supervisor（后台督导）协同工作
- **长期记忆管理**: 支持日记、学习材料和对话大纲的存储与检索
- **情绪识别**: 扩展的中文情绪词汇库（30+情绪类型），支持强度分级
- **结构化咨询**: 基于PAIP（问题-评估-干预-计划）模型的对话大纲
- **Web界面**: 提供WebSocket实时聊天和文件上传功能
- **向量数据库**: 使用ChromaDB存储和检索向量化记忆

## 系统架构

```
用户 ↔ Chatter (实时响应)
         ↑ 提示增强
   Analyst (后台分析，触发时注入)
   Supervisor (后台督导，触发时注入)
```

### 核心模块

- **协调层**: `SharedContext` (线程安全上下文管理)，`PsychologicalCounselor` (主协调器)
- **代理层**: 
  - `chatter.py` - 前台对话代理，实时响应用户
  - `analysist.py` - 后台分析代理，综合分析记忆
  - `supervisor.py` - 后台督导代理，提供专业指导
- **记忆系统**:
  - 存储: `mem_store_diary.py`, `mem_store_material.py`, `mem_store_conv_outline.py`
  - 检索: `mem_retrieve_diary.py`, `mem_retrieve_material.py`, `mem_retrieve_conv_outline.py`
- **Web应用**: FastAPI服务器，WebSocket聊天，文件上传API
- **会话管理**: JSON文件存储，自动标题生成

## 快速开始

### 环境要求

- Python 3.9+
- [Ollama](https://ollama.ai/) (用于本地嵌入模型)
- DeepSeek API密钥（或其他兼容LLM服务）

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

4. **环境配置**
   复制`.env.example`为`.env`并配置API密钥：
   ```bash
   cp .env.example .env
   # 编辑.env文件，填入您的API密钥
   ```

5. **启动Ollama服务**
   ```bash
   ollama serve
   # 在另一个终端中拉取嵌入模型
   ollama pull nomic-embed-text
   ```

6. **启动应用**

   **CLI模式** (当前可用):
   ```bash
   cd src
   python user_interface.py
   ```




## 使用方法

### 实时对话
1. 在聊天界面中输入消息开始对话
2. 系统会自动分析对话内容并注入相关记忆

### 文件上传
支持三种文件类型：
- **日记文件**: 自动提取情绪、触发因素、身体感受
- **学习材料**: 创建语义分块，支持父子关联检索
- **对话记录**: 生成PAIP结构化大纲

### 会话管理
- **加载会话**: 输入 `/load <会话ID>` 加载历史对话
- **自动保存**: 对话自动保存到 `database/sessions/`
- **标题生成**: 自动为会话生成描述性标题

## 配置说明

### 环境变量
```env
DATA_DIR=database                    # 数据目录
LANGSMITH_TRACING=true               # LangSmith追踪开关
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_key # LangSmith API密钥
LANGSMITH_PROJECT=main               # LangSmith项目名
DEEPSEEK_API_KEY=your_deepseek_key   # DeepSeek API密钥
```

### 向量数据库
- 位置: `database/chroma.sqlite3`
- 集合: `original_diary`, `diary_annotation`, `child_chunks`, `conv_outline`
- 嵌入模型: `nomic-embed-text` (通过Ollama)

## 开发指南

### 项目结构
```
Counselor-Agent-main/
├── src/                    # 源代码
│   ├── analysist.py       # 分析代理
│   ├── chatter.py         # 对话代理
│   ├── supervisor.py      # 督导代理
│   ├── SharedContext.py   # 共享上下文
│   ├── user_interface.py  # 主入口点（CLI）
│   └── mem_*/             # 记忆存储/检索模块
├── web/                   # Web相关文件
│   └── sessions/          # 会话存储目录
├── database/              # 数据存储
│   ├── chroma.sqlite3     # 向量数据库
│   └── sessions/          # 会话存储
├── test/                  # 测试文件
├── .env.example           # 环境变量示例
└── requirements.txt       # Python依赖
```

### 扩展系统
- **添加新情绪类型**: 修改 `src/mem_store_diary.py` 中的 `EMOTION_VOCABULARY`
- **添加新记忆类型**: 创建对应的 `mem_store_*.py` 和 `mem_retrieve_*.py`
- **自定义代理**: 继承基础代理类并注册到协调器

## 故障排除

### 常见问题
1. **WebSocket连接失败**
   - 检查Ollama服务是否运行: `ollama serve`
   - 验证API密钥配置

2. **文件上传错误**
   - 确保文件格式正确（日记、材料、对话）
   - 检查数据库连接状态

3. **记忆检索无结果**
   - 确认已上传相关文件
   - 检查向量数据库初始化

### 日志查看
```bash
# 查看应用日志
cd src
python -m web.main 2>&1 | tee app.log
```

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 贡献指南

欢迎提交Issue和Pull Request。请确保：
1. 遵循现有代码风格
2. 添加适当的测试
3. 更新相关文档

## 联系方式

如有问题或建议，请通过GitHub Issues提交。

---

*最后更新: 2025-04-20*
