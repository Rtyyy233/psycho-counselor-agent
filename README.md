# 心理咨询师代理系统 (Counselor Agent)

一个基于 **双 Agent 观察者模式** 的心理咨询 AI 代理系统。Chatter（前台咨询师）与 Supervisor（统一分析师+督导）协同工作，集成渐进式披露技能体系、14 领域用户画像、跨会话治疗连续性和 Chroma 向量数据库。

## 功能特性

- **双 Agent 协作**: Chatter（整合式心理咨询师，15 项治疗技能）+ Supervisor（分析师+督导合一，5 项监测技能）
- **渐进式披露技能体系**: L1 元数据（技能目录）始终在系统提示中 → L2 完整指南按需加载（`lookup_skill`）→ L3 资源按需引用。基于 skillkit 实现
- **治疗决策一致性**: 治疗计划（TreatmentPlan）贯穿全流程——PlanGenerator Agent 自适应检索制定计划 → Chatter 每轮接收动态技能目录 → Supervisor 监测方法一致性 → 会话结束后自动评估进展和阶段切换
- **四阶段治疗模型**: 建立期 → 工作期-认知行为 → 工作期-情感深化 → 整合/收尾期。每阶段有默认方法推荐，阶段切换基于 Lambert & Hawkins 进展评估框架
- **言语强度校准**: 三层防护（Chatter 校准表 + Supervisor over-activation 检测 + 技能目录激活风险标记），解决纯文字交流中情绪词汇被过度解读为临床严重度的系统性偏差
- **14 领域用户画像**: 4 大类别（baseline/dynamic/emotional_world/communication），3 轮 LLM 生成策略，增量更新机制
- **跨会话连续性**: 治疗计划、未完成事项、上次会话 Plan 自动注入新会话
- **Agentic RAG**: 三个检索模块各使用 LangGraph 状态机构建多步检索流水线（LLM Planner → 路由分发 → 多模式搜索）
- **结构化摘要**: 基于 PAIP（问题-评估-干预-计划）模型的对话摘要存储

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
   └────────────┬──────────────────────────┬──────────────────┘
                │ 用户消息                  │ supervisor_trigger
                ▼                          ▼
   ┌──────────────────┐       ┌──────────────────────────┐
   │     Chatter       │       │       Supervisor          │
   │  (前台咨询师)      │       │  (分析师+督导合一)         │
   │                  │       │                          │
   │  15 个治疗技能    │◄──────│  5 个监测技能              │
   │  人本主义底色     │ 注入   │  3 层监测框架              │
   │  4 阶段遵从       │       │  治疗计划一致性监测         │
   │  言语强度校准     │       │  言语强度错判检测           │
   └──────────────────┘       └──────────────────────────┘
            │                            │
            ▼                            ▼
   ┌──────────────────────────────────────────────┐
   │               记忆系统 + 画像系统               │
   │  日记 · 材料 · 对话大纲 (ChromaDB)             │
   │  用户画像 (14 领域 JSON + 版本快照)             │
   │  治疗计划 (4 阶段 + 目标进展追踪)               │
   └──────────────────────────────────────────────┘
```

### 核心循环

1. 用户发送消息 → `SharedContext.add_message()` 追加消息并触发 `supervisor_trigger` 事件
2. 后台任务 `call_supervisor` 异步启动——监听触发信号，加载治疗计划和画像上下文，检查对话状态，可能设置 `PromptInjection`
3. `call_chatter` 同步执行——接收治疗计划上下文、画像摘要、检索结果，基于动态技能目录生成回复
4. 回复加入上下文，循环继续。每 5 轮触发轻量画像更新。Supervisor 标记 `should_revise_plan ≥ 2` 时触发中期计划更新

## 模块说明

### 双 Agent

| Agent | 文件 | 职责 | 工具 |
|-------|------|------|------|
| **Chatter** | `chatter.py` | 整合式咨询师，人本主义底色，15 项治疗技能按需加载，言语强度校准，治疗计划遵从 | `read_file_tool`, `retrieve_user_profile_tool`, `lookup_skill`, `get_available_skills` |
| **Supervisor** | `supervisor.py` | 分析师+督导合一，3 层监测（基础/按需/危机），治疗计划一致性监测，言语强度错判检测，跨会话模式识别 | 全部 3 个检索工具 + `read_file_tool` + 2 个存储工具 + `lookup_skill` + `get_available_skills` |

### 技能系统 (`skill_loader.py`)

渐进式披露三层架构：L1 元数据（始终在 prompt）→ L2 完整指南（`lookup_skill` 按需加载）→ L3 资源（按需引用）。

- **Chatter 技能 (15)**: person-centered, existential, psychodynamic, adlerian, gestalt, cbt, behavioral-third-wave, choice-reality, sfbt, narrative, feminist, family-systems, alliance-repair, clinical-interviewing, crisis-intervention
- **Supervisor 技能 (5)**: alliance-monitoring, process-quality, countertransference, pattern-recognition, crisis-detection

支持基于治疗计划的动态技能目录——主要方法标 ★、谨慎方法标 ⚠、高风险技能标注激活风险警告。

### 治疗计划系统

| 模块 | 职责 |
|------|------|
| `treatment_plan.py` | 数据模型：TreatmentPlan, TreatmentGoal, SkillUsageRecord, InterventionResponse。4 阶段 × 15 技能默认映射，Lambert & Hawkins 绿黄红编码 |
| `plan_generator.py` | 计划生成：14 个并行 broad queries → 最多 5 轮自适应补充检索 → 单次 LLM 结构化输出 |
| `plan_manager.py` | 计划维护：JSON 持久化，`update_plan` 会话后评估，`should_transition_stage` 阶段切换，`build_continuity_context` 连续性文本 |

### 用户画像系统 (`user_profile/`)

14 领域 × 4 类别（baseline/dynamic/emotional_world/communication）。3 轮 LLM 生成策略，增量更新机制。

| 模块 | 职责 |
|------|------|
| `profile_models.py` | 数据模型：Fact (6 类型), ProfileDomain, Profile (含版本历史), 14 领域注册表 |
| `profile_collector.py` | 数据采集：4 种策略（PAIP 扫描、日记扫描、材料语义搜索、对话语义搜索） |
| `profile_generator.py` | LLM 生成：3 轮初始化 + 增量领域更新 + 更新筛选 |
| `profiler.py` | 编排器：`init_profile`, `update_profile`, `get_or_init_profile` |
| `mem_retrieve_user_profile.py` | 检索：文件优先（JSON）+ Chroma 回退 |

### 记忆系统

三种数据类型，每种有独立的存储和检索模块：

| 类型 | 存储 | 检索 | Chroma 集合 |
|------|------|------|-------------|
| 日记 | `mem_store_diary.py` | `mem_retrieve_diary.py` (LangGraph) | `original_diary`, `diary_annotation` |
| 材料 | `mem_store_material.py` | `mem_retrieve_material.py` (LangGraph + parent_lookup) | `child_chunks`, `parent_chunks` |
| 对话大纲 | `mem_store_conv_outline.py` | `mem_retrieve_conv_outline.py` (LangGraph + PAIP 四段重组) | `conv_outline` |

三个检索图共享相同的控制流：LLM Planner → `route_dispatch` → 执行节点 → `after_execution` 步进。

### 其他模块

| 模块 | 职责 |
|------|------|
| `merge_router.py` | LLM 检索查询去重：合并 Chatter 和 Supervisor 的语义相似查询 |
| `session_manager.py` | JSON 会话持久化 |
| `SharedContext.py` | 线程安全异步上下文容器，含 PromptInjection、Token 追踪、自动清理 |
| `config.py` | 集中化配置（路径、模型名、超时、文件类型检测） |
| `read_file.py` | 多格式文件读取（txt/pdf/md/csv/docx） |

## 快速开始

### 环境要求

- Python 3.11+
- [Ollama](https://ollama.ai/)（本地嵌入模型 `qwen3-embedding:4b`）
- DeepSeek API 密钥

### 安装步骤

```bash
git clone https://github.com/Rtyyy233/psycho-counselor-agent.git
cd Counselor-Agent-main
pip install -r requirements.txt
```

编辑 `.env`：
```env
DEEPSEEK_API_KEY=your_deepseek_api_key
LLM_MODEL=deepseek-v4-flash
DATA_DIR=database
```

启动 Ollama：
```bash
ollama serve
ollama pull qwen3-embedding:4b
```

启动：
```bash
cd src
python user_interface.py
```

## 使用方法

### 命令列表

| 命令 | 功能 |
|------|------|
| 直接输入 | 开始对话，Chatter 回复 + Supervisor 后台监测 |
| `/tokens` | 查看令牌使用情况 |
| `/load <ID>` | 加载历史会话 |
| `/clean [百分比] [auto]` | 清理旧消息并存储 PAIP 摘要 |
| `/profile` | 查看用户画像 14 领域摘要 |
| `/profile init` | 初始化用户画像（从历史数据生成） |
| `/profile update` | 手动增量更新画像 |
| `/profile view <1-14>` | 查看特定画像领域详情 |
| `/plan` | 查看当前治疗计划 |
| `/plan init` | 运行 PlanGenerator 制定/重新制定治疗计划 |
| `/help` | 显示帮助 |
| `/exit` | 退出并保存 |

### 治疗计划工作流

```
/plan init
  → PlanGenerator 检索日记/材料/PAIP/画像
  → 生成个性化治疗计划（阶段 + 目标 + 方法优先级）
  → 自动注入后续对话

每次对话中:
  → Chatter 收到动态技能目录（★主要 / 辅助 / ⚠谨慎）
  → Supervisor 监测方法一致性
  → 每 5 轮 + 会话结束自动更新画像

/exit
  → 存储 PAIP 摘要
  → 评估目标进展
  → 更新治疗计划和画像
```

## 项目结构

```
Counselor-Agent-main/
├── src/
│   ├── user_interface.py       # CLI 主入口
│   ├── config.py               # 集中化配置
│   ├── SharedContext.py        # 线程安全异步上下文
│   ├── session_manager.py      # JSON 会话持久化
│   ├── conversation_manager.py # 自动摘要的对话管理
│   ├── chatter.py              # 前台咨询 Agent（整合式，15 技能）
│   ├── supervisor.py           # 分析师+督导合一 Agent（5 技能，3 层监测）
│   ├── analysist.py            # [废弃] 向后兼容 shim → 重新导出 supervisor
│   ├── supervisoner.py         # [备选] 另一套 Supervisor 实现（structured output）
│   ├── skill_loader.py         # 渐进式披露技能管理（skillkit）
│   ├── merge_router.py         # LLM 检索查询去重
│   ├── treatment_plan.py       # 治疗计划数据模型
│   ├── plan_generator.py       # PlanGenerator：自适应多轮检索生成计划
│   ├── plan_manager.py         # PlanManager：持久化、更新、阶段切换
│   ├── mem_integration.py      # Chroma 初始化 + 嵌入 + 工具定义
│   ├── mem_store_diary.py      # 日记存储
│   ├── mem_store_material.py   # 材料存储（父子语义分块）
│   ├── mem_store_conv_outline.py # 对话大纲存储（PAIP 模型）
│   ├── mem_retrieve_diary.py   # 日记检索 (LangGraph 状态机)
│   ├── mem_retrieve_material.py # 材料检索 (LangGraph + small-to-big)
│   ├── mem_retrieve_conv_outline.py # 对话大纲检索 (LangGraph + PAIP 重组)
│   ├── mem_retrieve_user_profile.py # 用户画像检索 (文件优先 + Chroma 回退)
│   ├── read_file.py            # 多格式文件读取
│   ├── user_profile/           # 用户画像系统
│   │   ├── schema.md           #   14 领域详细 Schema
│   │   ├── profile_models.py   #   数据模型 + 领域注册表
│   │   ├── profile_collector.py #   4 策略数据采集
│   │   ├── profile_generator.py #   3 轮 LLM 生成 + 增量更新
│   │   └── profiler.py         #   编排器
│   ├── skills/                 # 技能知识库（渐进式披露）
│   │   ├── chatter/            #   15 个治疗技能 SKILL.md
│   │   └── supervisor/         #   5 个督导技能 SKILL.md
│   ├── _book_extract/          # 教材提取文本（RAG）
│   │   ├── interviewing/       #   Sommers-Flanagan 临床面谈
│   │   └── supervision/        #   Falender & Shafranske 临床督导
│   └── web/                    # Web 服务（FastAPI + WebSocket）
├── database/                   # Chroma 数据 + 画像/计划 JSON（gitignored）
├── test/                       # pytest 测试文件
├── .env                        # 环境变量（gitignored）
├── requirements.txt
├── CLAUDE.md                   # 项目文档（给 Claude Code 使用）
└── README.md
```

## 开发指南

### 运行测试
```bash
cd src
python -m pytest ../test/ -v                          # 全部测试
python -m pytest ../test/test_treatment_plan.py -v     # 治疗计划模型
python -m pytest ../test/test_plan_manager.py -v       # 计划管理器
python -m pytest ../test/test_treatment_continuity.py -v # 治疗连续性集成测试
```

### 代码风格
```bash
ruff check src/    # 检查
ruff format src/   # 格式化
```

### 更换 LLM 模型
修改 `.env` 中 `LLM_MODEL` 即可，所有模块自动生效。

## 许可证

MIT License
