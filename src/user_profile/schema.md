# 用户画像 Schema (用户画像数据模式)

## 概述

用户画像是对来访者（TA）的系统性心理特征记录，包含 **14 个领域**，分为四大类。遵循渐进式呈现原则：顶层摘要暴露高频需求，下层文件按需加载。

## Schema 概览

| # | 领域 | 类别 | 核心问题 |
|---|------|------|---------|
| 1 | 主诉与现状 | dynamic | TA 目前最大的困扰是什么？ |
| 2 | 成长与发展史 | baseline | 什么经历塑造了今天的 TA？ |
| 3 | 易感因素 | baseline | 什么让 TA 对当前问题更脆弱？ |
| 4 | 诱发因素 | dynamic | 为什么是现在来求助？ |
| 5 | 维持因素 | dynamic | 什么在让问题持续？ |
| 6 | 保护因素 | dynamic | 什么在支撑着 TA？ |
| 7 | 关系模式 | baseline | TA 如何与重要他人互动？ |
| 8 | 干预反应 | dynamic | 什么对 TA 有效 / 无效？ |
| 9 | 风险评估 | dynamic | TA 有哪些需要警惕的风险？ |
| 10 | 文化/背景因素 | baseline | 文化/家庭/社会如何影响 TA？ |
| 11 | 人格印象 | baseline | TA 是个什么样的人？ |
| 12 | 情感世界 | emotional_world | TA 的深层渴望与恐惧是什么？ |
| 13 | 沟通风格 | communication | TA 习惯如何表达和接收？ |
| 14 | 求助与改变模式 | dynamic | TA 如何面对改变？ |

## 类别说明

- **baseline**（基线信息）：相对稳定、缓慢变化的信息。首次密集采集，后续增量更新。加载策略：会话开始时预加载。
- **dynamic**（动态信息）：随咨询进程变化的信息。每次会话后检查更新。加载策略：按需加载相关领域。
- **emotional_world**（情感世界）：跨越稳定特质与动态情感的领域。加载策略：当对话触及情感议题时加载。
- **communication**（沟通风格）：关于表达与接收风格的元信息。加载策略：对话早期采集，后续用于调整咨询师回应风格。

## 领域详细结构

### 1. 主诉与现状 (Presenting Complaint & Current Status)

- **Category**: dynamic
- **Sub-fields**:
  - `presenting_problem`: TA 用自己的语言说的核心困扰
  - `symptom_profile`: 症状表现、频率、强度
  - `functional_impact`: 对工作/社交/日常生活的影响
  - `stage_assessment`: 基于 Prochaska 改变阶段的评估（前沉思/沉思/准备/行动/维持），用于指导治疗计划的阶段设定

### 2. 成长与发展史 (Growth & Developmental History)

- **Category**: baseline
- **Description**: TA 的生命故事脉络（客观事实）

### 3. 易感因素 (Predisposing Factors)

- **Category**: baseline
- **Description**: 什么让 TA 更脆弱（因果推断）
- **Note**: 与 #2 的区别：#2 记录"发生了什么"，#3 记录"这些经历如何让 TA 更脆弱"

### 4. 诱发因素 (Precipitating Factors)

- **Category**: dynamic
- **Description**: 这次为什么现在求助

### 5. 维持因素 (Perpetuating Factors)

- **Category**: dynamic
- **Description**: 什么在让问题持续

### 6. 保护因素 (Protective Factors)

- **Category**: dynamic
- **Description**: 什么在兜底、什么在撑着 TA

### 7. 关系模式 (Relational Patterns)

- **Category**: baseline
- **Description**: TA 怎么和重要他人互动

### 8. 干预反应 (Intervention Response)

- **Category**: dynamic
- **Sub-fields**:
  - `effective_interventions`: 什么技术/回应方式有效
  - `ineffective_or_harmful`: 什么触发了防御或恶化
  - `alliance_quality`: 咨访关系特点（信任程度、依赖模式、边界反应）
  - `treatment_plan_alignment`: 治疗计划与该来访者的匹配度评估 — 哪些方法值得坚持，哪些需要调整

### 9. 风险评估 (Risk Assessment)

- **Category**: dynamic
- **Sub-fields**:
  - `self_harm_risk`: 自伤风险及信号
  - `suicide_risk`: 自杀风险及信号
  - `violence_risk`: 攻击风险
  - `dropout_risk`: 脱落风险及信号
  - `crisis_history`: 过往危机事件及应对

### 10. 文化/背景因素 (Cultural & Contextual Factors)

- **Category**: baseline
- **Description**: 文化、家庭、社会环境的影响

### 11. 人格印象 (Personality Impression)

- **Category**: baseline
- **Description**: TA 是个什么样的人：性格底色、价值观、自我认同、骄傲与羞耻。侧重"相对稳定的特质"

### 12. 情感世界 (Emotional World)

- **Category**: emotional_world
- **Description**: 深层渴望与恐惧、情感表达方式、意义感来源。侧重"动态的情感动力"
- **Note**: 与 #11 边界：#11 是稳定性特质，#12 是动态情感

### 13. 沟通风格 (Communication Style)

- **Category**: communication
- **Sub-fields**:
  - `language_style`: 语言风格、用词习惯
  - `metaphor_preference`: 偏好/避讳的隐喻和意象
  - `pacing_preference`: 对话节奏（快/慢、直接/迂回）
  - `response_receptivity`: 对什么回应方式接受/防御

### 14. 求助与改变模式 (Help-Seeking & Change Pattern)

- **Category**: dynamic
- **Sub-fields**:
  - `change_stage`: 改变阶段（前意向/意向/准备/行动/维持）
  - `help_seeking_style`: 求助风格（分析型/宣泄型/行动型）
  - `resistance_pattern`: 对什么类型的引导容易产生阻抗
  - `counseling_expectation`: 对咨询关系的期待

## 事实格式 (Fact Schema)

每条事实记录遵循统一结构：

```yaml
- id: fact_{nnn}          # 全局唯一 ID，按序递增
  type: <fact_type>        # 事实类型（见下）
  statement: <string>      # 事实陈述
  evidence: [source_ids]   # 来源会话 ID 列表
  confidence: <float>      # 置信度 0.0-1.0
  relates_to: [fact_ids]   # 关联的其他事实 ID
```

### 事实类型

| 类型 | 说明 |
|------|------|
| `observation` | 咨询师直接观察到的事实 |
| `self_report` | TA 自我报告的内容 |
| `inference` | 基于证据的推断 |
| `pattern` | 识别出的重复模式 |
| `risk` | 风险评估发现 |
| `treatment_response` | 干预效果观察 |
