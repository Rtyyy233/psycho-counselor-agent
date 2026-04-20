import asyncio
import sys
sys.path.insert(0, 'src')

async def test():
    from SharedContext import SharedContext
    
    # 创建简单上下文
    ctx = SharedContext(session_id="quick_test", token_limit=1000)
    
    # 添加多条消息
    for i in range(5):
        await ctx.add_message("user", f"消息{i} " * 10)  # 长消息
        await ctx.add_message("assistant", f"回复{i} " * 10)
    
    print(f"消息数: {len(ctx._messages)}")
    
    # 计算令牌
    tokens = await ctx.calculate_token_count()
    print(f"令牌数: {tokens}")
    
    # 获取使用情况
    usage = await ctx.get_token_usage()
    print(f"使用率: {usage['usage_percentage']:.1f}%")
    
    # 测试清理（无存储回调）
    print("\n测试清理功能...")
    result = await ctx.cleanup_context(target_usage=0.5, storage_callback=None)
    
    print(f"清理状态: {result['status']}")
    if result['status'] == 'success':
        print(f"清理消息数: {result['cleaned_messages']}")
        print(f"清理令牌数: {result['cleaned_tokens']}")
        print(f"剩余消息数: {result['remaining_messages']}")
        print(f"新使用率: {result['new_usage_percentage']:.1f}%")
    
    # 最终状态
    final_usage = await ctx.get_token_usage()
    print(f"\n最终使用率: {final_usage['usage_percentage']:.1f}%")
    
    print("测试完成")

if __name__ == "__main__":
    asyncio.run(test())