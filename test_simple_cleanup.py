#!/usr/bin/env python3
"""简单测试清理功能"""

import asyncio
import sys
sys.path.insert(0, 'src')

async def test_basic():
    print("测试基本清理功能...")
    
    from SharedContext import SharedContext
    
    # 创建测试上下文
    ctx = SharedContext(session_id="simple_test", token_limit=500)
    
    # 添加一些消息
    messages = [
        ("user", "第一条消息"),
        ("assistant", "第一条回复"),
        ("user", "第二条消息，稍微长一些"),
        ("assistant", "第二条回复，也更长一些"),
        ("user", "第三条消息，这是最后一条测试消息"),
    ]
    
    for role, content in messages:
        await ctx.add_message(role, content)
    
    print(f"添加了{len(messages)}条消息")
    print(f"当前消息数: {len(ctx._messages)}")
    
    # 计算令牌
    tokens = await ctx.calculate_token_count()
    print(f"估算令牌数: {tokens}")
    
    # 测试清理（无存储）
    print("\n执行清理测试...")
    result = await ctx.cleanup_context(target_usage=0.3, storage_callback=None)
    
    print(f"清理结果: {result['status']}")
    if result['status'] == 'success':
        print(f"清理了{result['cleaned_messages']}条消息")
        print(f"释放了{result['cleaned_tokens']}令牌")
        print(f"剩余{result['remaining_messages']}条消息")
        print(f"新使用率: {result['new_usage_percentage']:.1f}%")
    
    # 测试get_oldest_messages
    print("\n测试get_oldest_messages...")
    oldest_msgs, oldest_tokens = await ctx.get_oldest_messages(target_tokens=20)
    print(f"获取到{len(oldest_msgs)}条旧消息，约{oldest_tokens}令牌")
    
    print("\n✅ 基本测试完成")

async def test_storage_callback():
    print("\n测试存储回调集成...")
    
    from SharedContext import SharedContext
    
    ctx = SharedContext(session_id="callback_test", token_limit=500)
    
    # 添加消息
    for i in range(3):
        await ctx.add_message("user", f"测试消息{i}")
        await ctx.add_message("assistant", f"测试回复{i}")
    
    # 模拟存储回调
    async def mock_callback(text, metadata):
        print(f"  回调被调用，文本长度: {len(text)}")
        print(f"  元数据: {metadata}")
        return "mock_storage_id_123"
    
    result = await ctx.cleanup_context(
        target_usage=0.5,
        storage_callback=mock_callback
    )
    
    print(f"清理结果: {result['status']}")
    if result.get('storage_id'):
        print(f"存储ID: {result['storage_id']}")
    
    print("\n✅ 存储回调测试完成")

if __name__ == "__main__":
    print("="*60)
    print("简单清理功能测试")
    print("="*60)
    
    try:
        asyncio.run(test_basic())
        asyncio.run(test_storage_callback())
        print("\n" + "="*60)
        print("✅ 所有测试通过！")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)