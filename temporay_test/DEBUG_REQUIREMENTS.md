# Debug需求文档：Analyst注入失败问题

## 问题描述
用户报告Analyst的分析结果未能成功注入到Chatter的提示词中。尽管已经实施了多个修复（包括重试机制、超时设置和错误处理），但注入仍然失败。

## 影响
- Chatter无法获得Analyst的深度分析洞察
- 心理咨询师代理失去关键的记忆检索和分析功能
- 系统退化为基本对话机器人，失去核心价值

## 目标
1. 确定注入失败的根本原因
2. 修复Analyst到Chatter的注入管道
3. 确保注入状态在WebSocket重新连接后仍然保持
4. 建立可靠的端到端测试验证

## 假设
1. Analyst能够成功生成分析结果
2. SharedContext对象在组件间正确共享
3. 时序问题是主要障碍
4. WebSocket连接管理可能导致状态重置

## 调查步骤

### 阶段1：诊断当前状态
1. **验证Analyst输出**
   - Analyst是否成功调用？
   - 分析结果是否非空？
   - Analyst是否调用`safe_set_analyst()`？

2. **验证SharedContext状态**
   - SharedContext实例是否正确传递？
   - injection字段是否被设置？
   - 锁机制是否正常工作？

3. **验证Chatter读取**
   - Chatter是否看到injection？
   - enriched_prompt是否包含分析内容？
   - retry机制是否生效？

4. **验证时序问题**
   - Analyst与Chatter的调用顺序
   - 异步等待时间是否足够

5. **验证WebSocket影响**
   - 连接重置是否影响SharedContext
   - session_manager是否保存注入状态

### 阶段2：代码分析
1. **审查关键文件**
   - `src/chatter.py` - 注入读取逻辑
   - `src/analysist.py` - 分析生成逻辑
   - `src/top_module.py` - SharedContext定义
   - `src/web/main.py` - WebSocket处理
   - `src/web/session_manager.py` - 会话状态管理

2. **识别架构问题**
   - 组件间的数据流
   - 状态管理策略
   - 错误处理机制

### 阶段3：实施修复
1. **解决已识别的问题**
2. **添加监控和日志**
3. **实现状态持久化**
4. **优化时序协调**

## 测试计划
1. 单元测试：各个组件的独立功能
2. 集成测试：Analyst-Chatter数据流
3. 端到端测试：完整对话流程
4. 负载测试：多用户并发场景

## 成功标准
1. Analyst分析结果成功注入Chatter提示词
2. 注入在WebSocket重新连接后仍然有效
3. 系统响应时间可接受（<3秒）
4. 错误率低于1%

## 风险与缓解
1. **时序竞争条件**：增加同步机制或状态确认
2. **内存泄漏**：监控SharedContext生命周期
3. **性能下降**：优化检索操作，添加缓存
4. **状态不一致**：实现状态校验和恢复

## 交付物
1. 根本原因分析报告
2. 修复的代码变更
3. 测试用例和验证脚本
4. 监控和警报配置

## 时间估计
- 诊断和分析：2小时
- 修复实施：3小时
- 测试和验证：2小时
- 文档更新：1小时
总计：8小时