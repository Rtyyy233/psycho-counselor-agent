#!/usr/bin/env python3
"""
集成测试：上下文清理功能
测试SharedContext.cleanup_context与store_conversation_outline的集成
"""

import asyncio
import sys
import os
import pytest
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# 尝试导入所需模块
try:
    from SharedContext import SharedContext
    from langchain_core.documents import Document
    HAS_DEPENDENCIES = True
except ImportError as e:
    print(f"导入依赖失败: {e}")
    HAS_DEPENDENCIES = False

try:
    from mem_store_conv_outline import store_conversation_outline
    HAS_STORE_MODULE = True
except ImportError:
    HAS_STORE_MODULE = False

# 如果缺少核心依赖，跳过所有测试
pytestmark = pytest.mark.skipif(
    not HAS_DEPENDENCIES, 
    reason="缺少核心依赖（SharedContext等）"
)

# 测试数据
TEST_MESSAGES = [
    ("user", "你好，我感觉最近压力很大，工作上的事情总是处理不完。"),
    ("assistant", "我理解你的感受。工作压力确实是现代人常见的问题。你能具体说说是什么工作让你感到压力吗？"),
    ("user", "我在一家互联网公司做产品经理，最近有一个重要的项目要上线，每天都要加班到很晚。"),
    ("assistant", "听起来确实很辛苦。产品经理这个角色需要协调多方，压力自然不小。除了加班，还有哪些方面让你感到压力？"),
    ("user", "团队沟通不太顺畅，开发进度经常延迟，我感觉自己要对所有问题负责。"),
    ("assistant", "这种责任感确实会带来很大压力。你觉得团队沟通问题主要出在哪里？是信息传递不畅，还是大家对目标理解不一致？"),
    ("user", "两方面都有。我发了需求文档，但开发人员说看不懂；进度会议上大家也都沉默不语。"),
    ("assistant", "这确实是个挑战。也许我们可以探讨一些改善沟通的方法，比如更直观的需求表达方式，或者建立更开放的讨论氛围。"),
    ("user", "我也尝试过画原型图，但效果还是不太好。有时候我觉得自己不适合这个职位。"),
    ("assistant", "请不要轻易否定自己。每个角色都有一个学习曲线，你已经在积极寻找解决方案了，这很了不起。"),
]

async def create_test_context(token_limit=1000, add_messages=True):
    """创建测试用的SharedContext实例"""
    ctx = SharedContext(session_id="test_integration", token_limit=token_limit)
    
    if add_messages:
        for role, content in TEST_MESSAGES:
            await ctx.add_message(role, content)
    
    return ctx

@pytest.mark.asyncio
async def test_cleanup_context_basic():
    """测试基本清理功能（无存储回调）"""
    ctx = await create_test_context(token_limit=100)
    
    # 获取初始状态
    initial_count = len(ctx._messages)
    initial_tokens = await ctx.calculate_token_count()
    
    # 执行清理（目标使用率50%，无存储回调）
    result = await ctx.cleanup_context(target_usage=0.5, storage_callback=None)
    
    # 验证结果
    assert result["status"] == "success"
    assert "cleaned_messages" in result
    assert "cleaned_tokens" in result
    assert "remaining_tokens" in result
    
    # 验证消息确实被删除
    final_count = len(ctx._messages)
    assert final_count < initial_count
    
    # 验证令牌数减少
    final_tokens = await ctx.calculate_token_count()
    assert final_tokens < initial_tokens
    
    print(f"基本清理测试通过: 清理了{result['cleaned_messages']}条消息，{result['cleaned_tokens']}令牌")

@pytest.mark.asyncio
async def test_cleanup_context_with_storage():
    """测试清理功能与存储回调的集成"""
    # 如果存储模块不可用，跳过测试
    if not HAS_STORE_MODULE:
        pytest.skip("存储模块不可用，跳过存储集成测试")
    
    ctx = await create_test_context(token_limit=100)
    
    # 创建存储回调
    async def mock_storage_callback(conversation_text: str, metadata: dict) -> str:
        """模拟存储回调，验证参数"""
        assert conversation_text
        assert len(conversation_text) > 0
        assert "session_id" in metadata
        assert "cleaned_at" in metadata
        assert "message_count" in metadata
        assert "token_count" in metadata
        
        # 验证对话文本格式
        assert "user:" in conversation_text
        assert "assistant:" in conversation_text
        
        return "mock_storage_id_123"
    
    # 执行清理
    result = await ctx.cleanup_context(
        target_usage=0.5, 
        storage_callback=mock_storage_callback
    )
    
    # 验证结果
    assert result["status"] == "success"
    assert result.get("storage_id") == "mock_storage_id_123"
    
    print(f"存储集成测试通过: 存储ID={result['storage_id']}")

@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_STORE_MODULE, reason="存储模块不可用")
async def test_cleanup_context_real_store():
    """测试与实际store_conversation_outline的集成"""
    ctx = await create_test_context(token_limit=100)
    
    initial_count = len(ctx._messages)
    
    # 创建实际的存储回调
    async def real_storage_callback(conversation_text: str, metadata: dict) -> str:
        doc = Document(
            page_content=conversation_text,
            metadata={
                "source": f"test_session_{metadata['session_id']}",
                "cleaned_at": metadata["cleaned_at"],
                "message_count": metadata["message_count"],
                "token_count": metadata["token_count"],
                "test": True  # 标记为测试数据
            }
        )
        
        try:
            storage_id = await store_conversation_outline(doc)
            return storage_id
        except Exception as e:
            pytest.fail(f"存储对话摘要失败: {e}")
    
    # 执行清理
    result = await ctx.cleanup_context(
        target_usage=0.5,
        storage_callback=real_storage_callback
    )
    
    # 验证结果
    assert result["status"] == "success"
    assert result.get("storage_id") is not None
    assert result["storage_id"].startswith("conv_")
    
    print(f"实际存储集成测试通过: 存储ID={result['storage_id']}")

@pytest.mark.asyncio
async def test_cleanup_storage_failure():
    """测试存储失败时的处理（不应删除消息）"""
    ctx = await create_test_context(token_limit=100)
    
    initial_count = len(ctx._messages)
    
    # 创建会失败的存储回调
    async def failing_storage_callback(conversation_text: str, metadata: dict) -> str:
        raise Exception("模拟存储失败")
    
    # 执行清理（应该失败）
    result = await ctx.cleanup_context(
        target_usage=0.5,
        storage_callback=failing_storage_callback
    )
    
    # 验证存储失败
    assert result["status"] == "storage_failed"
    assert "error" in result
    assert "模拟存储失败" in result["error"]
    
    # 验证消息没有被删除
    final_count = len(ctx._messages)
    assert final_count == initial_count, "存储失败时消息不应被删除"
    
    print(f"存储失败处理测试通过: 错误信息='{result['error']}'")

@pytest.mark.asyncio
async def test_auto_cleanup_threshold_logic():
    """测试自动清理阈值逻辑"""
    ctx = await create_test_context(token_limit=100, add_messages=False)
    
    # 添加少量消息，使用率低于阈值
    for i in range(3):
        await ctx.add_message("user", f"测试消息{i}")
    
    usage = await ctx.get_token_usage()
    print(f"初始使用率: {usage['usage_percentage']}%")
    
    # 设置高阈值（比如90%），清理到70%
    # 由于使用率低，应该不会触发清理
    result = await ctx.cleanup_context(target_usage=0.7, storage_callback=None)
    
    if usage["usage_percentage"] < 70:
        assert result["status"] == "no_need"
        print("低于阈值时正确跳过清理")
    else:
        assert result["status"] == "success"
        print("高于阈值时正确执行清理")

@pytest.mark.asyncio
async def test_get_oldest_messages_without_tokenizer():
    """测试无tokenizer时的字符估算功能"""
    # 创建无tokenizer的上下文
    ctx = SharedContext(session_id="test_no_tokenizer", token_limit=1000, tokenizer=None)
    
    # 添加一些消息
    for i in range(5):
        await ctx.add_message("user", f"这是一条测试消息，用于测试字符估算功能。消息编号：{i}")
    
    # 测试get_oldest_messages
    messages, tokens = await ctx.get_oldest_messages(target_tokens=50)
    
    # 即使没有tokenizer，也应该返回结果
    assert len(messages) > 0
    assert tokens > 0
    
    print(f"字符估算测试通过: 获取到{len(messages)}条消息，估算{tokens}令牌")

if __name__ == "__main__":
    """直接运行集成测试"""
    import sys
    
    async def run_all_tests():
        test_results = []
        
        # 运行基本清理测试
        try:
            await test_cleanup_context_basic()
            test_results.append(("基本清理功能", "✅ 通过"))
        except Exception as e:
            test_results.append(("基本清理功能", f"❌ 失败: {e}"))
        
        # 运行存储集成测试（如果可用）
        if HAS_STORE_MODULE:
            try:
                await test_cleanup_context_with_storage()
                test_results.append(("存储回调集成", "✅ 通过"))
            except Exception as e:
                test_results.append(("存储回调集成", f"❌ 失败: {e}"))
        else:
            test_results.append(("存储回调集成", "⚠ 跳过（模块不可用）"))
        
        # 运行其他测试...
        try:
            await test_cleanup_storage_failure()
            test_results.append(("存储失败处理", "✅ 通过"))
        except Exception as e:
            test_results.append(("存储失败处理", f"❌ 失败: {e}"))
        
        try:
            await test_auto_cleanup_threshold_logic()
            test_results.append(("自动清理阈值", "✅ 通过"))
        except Exception as e:
            test_results.append(("自动清理阈值", f"❌ 失败: {e}"))
        
        try:
            await test_get_oldest_messages_without_tokenizer()
            test_results.append(("字符估算功能", "✅ 通过"))
        except Exception as e:
            test_results.append(("字符估算功能", f"❌ 失败: {e}"))
        
        # 打印结果汇总
        print("\n" + "="*60)
        print("集成测试结果汇总:")
        print("="*60)
        for test_name, result in test_results:
            print(f"{test_name:20} {result}")
        
        # 检查是否有失败
        failures = [name for name, result in test_results if "❌" in result]
        if failures:
            print(f"\n❌ 有{len(failures)}个测试失败: {', '.join(failures)}")
            sys.exit(1)
        else:
            print("\n✅ 所有测试通过！")
            sys.exit(0)
    
    asyncio.run(run_all_tests())