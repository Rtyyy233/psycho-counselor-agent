# 心理咨询师代理系统 - 架构概览

## 核心架构：观察者模式

```
用户 ↔ Chatter (实时响应)
         ↑ 提示增强
   Analyst (后台分析，触发时注入)
   Supervisor (后台督导，触发时注入)
```

## 关键修复总结

### 已解决的问题
1. ✅ **情绪类型验证错误** - 扩展EmotionType从14种到30+中文情绪词
2. ✅ **WebSocket观察者任务启动缺失** - 在websocket_chat中启动analyst_observer和supervisor_observer
3. ✅ **分析结果注入机制失效** - 确保观察者任务正确运行并注入结果
4. ✅ **LangSmith API密钥配置** - 已配置有效API密钥（追踪暂时禁用）
5. ✅ **相对导入错误** - 修复web模块的相对导入问题

### 架构核心修改
- `src/web/main.py:295-298` - 添加观察者任务启动代码
- `src/web/main.py:334-338` - 添加WebSocket断开时的任务取消
- `src/mem_store_diary.py:45-76` - 扩展情绪类型定义
- `src/mem_store_diary.py:95-98` - 修改EmotionalState.emotion为List[str]类型
- `.env:6` - 配置LangSmith API密钥（追踪暂时禁用）

## 文件级架构

### 核心协调层
- `src/top_module.py` - 顶层协调器，实现观察者模式
  - `SharedContext`: 线程安全的共享上下文（消息、注入、事件）
  - `PsychologicalCounselor`: 主协调器类
  - `analyst_observer()`: 后台分析师观察者
  - `supervisor_observer()`: 后台督导观察者

### 代理层
- `src/chatter.py` - 前台对话代理
  - 实时响应用户消息
  - 读取并应用Analyst/Supervisor的注入
  - 使用结构化输出确保响应质量

- `src/analysist.py` - 后台分析代理
  - 并行检索日记、材料、对话大纲
  - 综合分析结果生成洞察
  - 通过SharedContext注入分析结果

- `src/supervisoner.py` - 后台督导代理
  - 监控对话质量
  - 提供专业指导建议
  - 适度干预避免过度指导

### 记忆存储系统
- `src/mem_store_diary.py` - 日记存储
  - 处理日记文件，提取结构化信息
  - 扩展情绪词汇（30+中文情绪）
  - 存储到`original_diary`和`diary_annotation`集合

- `src/mem_store_material.py` - 材料存储
  - 处理学习材料，创建父子分块结构
  - 类型推断和语义分块

- `src/mem_store_conv_outline.py` - 对话大纲存储
  - 生成PAIP（问题-评估-干预-计划）大纲
  - 结构化咨询对话记录

### 记忆检索系统
- `src/mem_retrieve_diary.py` - 日记检索
  - 多维度过滤（情绪、时间、场景、事件类型）
  - LangGraph多步骤检索流程

- `src/mem_retrieve_material.py` - 材料检索
  - 语义搜索 + 父级关联查找

- `src/mem_retrieve_conv_outline.py` - 对话大纲检索
  - PAIP大纲检索，提供案例参考

### Web应用层
- `src/web/main.py` - FastAPI主应用
  - WebSocket聊天端点 (`/ws/chat`)
  - 文件上传API (`/api/upload`)
  - 会话管理API
  - **关键修复**: 观察者任务启动(295-298行)和清理(334-338行)

- `src/web/session_manager.py` - 会话管理
  - JSON文件存储会话数据
  - 自动标题生成和消息历史

- `src/web/static/` - 前端静态文件
  - `index.html` - 主聊天界面
  - `upload.html` - 文件上传界面
  - `app.js` - WebSocket前端逻辑
  - `styles.css` - 响应式样式

### 支持模块
- `src/config.py` - 集中化配置管理
- `src/tool_utils.py` - 工具调用工具类
- `src/conversation_manager.py` - 对话总结管理
- `src/mem_collections.py` - 向量集合定义

## 数据流

### 实时对话流程
1. 用户消息 → WebSocket → `handle_user_message()`
2. 添加到会话存储和共享上下文
3. **设置触发事件** → `ctx.on_analyst_trigger.set()`
4. 立即调用`call_chatter()`生成响应
5. 响应返回用户
6. **后台并行处理**: Analyst检索记忆 → 生成分析 → 注入上下文

### 文件上传流程
1. 用户上传文件 → 检测类型（日记/材料/对话）
2. 路由到对应存储模块
3. 处理并存储到向量数据库
4. 结果返回用户

## 向量数据库结构

| 集合 | 内容 | 用途 |
|------|------|------|
| `original_diary` | 原始日记文本 | 语义搜索 |
| `diary_annotation` | 结构化分析 | 情绪过滤 |
| `child_chunks` | 材料子分块 | 详细内容检索 |
| `conv_outline` | PAIP大纲 | 案例参考 |

## 关键数据结构

### SharedContext (`src/top_module.py:40-82`)
```python
class SharedContext:
    messages: List[ChatMessage]          # 对话历史
    analyst_injection: Optional[PromptInjection]    # Analyst注入
    supervisor_injection: Optional[PromptInjection] # Supervisor注入
    on_analyst_trigger: asyncio.Event    # Analyst触发事件
    on_supervisor_trigger: asyncio.Event # Supervisor触发事件
    _lock: asyncio.Lock                  # 线程安全锁
```

### EmotionalState (`src/mem_store_diary.py:95-108`)
```python
class EmotionalState(BaseModel):
    emotion: List[str]  # 支持30+中文情绪词
    intensity: Literal["潜意识", "弱", "中", "强", "极强"]
    triggers: List[str]     # 触发因素
    physical_sensations: List[str]  # 身体感受
```

## 运行状态验证

### 测试验证通过的功能
1. ✅ WebSocket观察者任务正确启动和停止
2. ✅ Analyst分析结果能够正确注入
3. ✅ Chatter能够读取和应用注入
4. ✅ 情绪类型验证接受扩展词汇
5. ✅ 完整的对话流程端到端工作

### 当前运行状态
- Web服务器: http://localhost:8000 (运行中)
- 文件上传: http://localhost:8000/upload (可用)
- 数据库: `database/chroma.sqlite3` (已初始化)
- LangSmith: API密钥已配置，追踪暂时禁用

## 部署配置

### 环境要求
- Python 3.9+
- Ollama服务 (用于本地嵌入)
- DeepSeek API密钥 (或兼容的LLM服务)

### 启动命令
```bash
# 激活虚拟环境
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 启动Web服务器
cd src
python -m web.main
```

### 访问地址
- 聊天界面: http://localhost:8000
- 文件上传: http://localhost:8000/upload
- API文档: http://localhost:8000/docs

## 后续优化建议

### 短期优化
1. **性能监控**: 添加请求耗时和资源使用监控
2. **错误恢复**: 增强观察者任务的错误恢复机制
3. **缓存策略**: 对常见查询结果添加缓存

### 长期规划
1. **多用户支持**: 添加用户认证和权限管理
2. **插件系统**: 支持自定义记忆类型和代理
3. **生产部署**: Docker容器化 + PostgreSQL + pgvector

---

*架构文档版本: 1.1*
*对应代码版本: 修复了核心问题后的稳定版本*
*最后验证: 2025-04-18*