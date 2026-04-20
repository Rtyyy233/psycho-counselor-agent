#!/usr/bin/env python3
"""
测试修复后的检索函数
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_retrieve_diary():
    """测试 retrieve_diary 函数"""
    print("测试 retrieve_diary...")
    try:
        from mem_retrieve_diary import retrieve_diary
        
        # 简单查询
        result = await retrieve_diary("测试查询")
        print(f"✓ retrieve_diary 调用成功")
        print(f"  返回类型: {type(result)}")
        print(f"  结果数量: {len(result)}")
        return True
    except Exception as e:
        print(f"✗ retrieve_diary 调用失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_retrieve_materials():
    """测试 retrieve_materials 函数"""
    print("\n测试 retrieve_materials...")
    try:
        from mem_retrieve_material import retrieve_materials
        
        # 简单查询
        result = await retrieve_materials("测试查询")
        print(f"✓ retrieve_materials 调用成功")
        print(f"  返回类型: {type(result)}")
        print(f"  结果数量: {len(result)}")
        return True
    except Exception as e:
        print(f"✗ retrieve_materials 调用失败: {type(e).__name__}: {e}")
        return False

async def test_retrieve_conv_outline():
    """测试 retrieve_conv_outline 函数"""
    print("\n测试 retrieve_conv_outline...")
    try:
        from mem_retrieve_conv_outline import retrieve_conv_outline
        
        # 简单查询
        result = await retrieve_conv_outline("测试查询")
        print(f"✓ retrieve_conv_outline 调用成功")
        print(f"  返回类型: {type(result)}")
        print(f"  结果数量: {len(result)}")
        return True
    except Exception as e:
        print(f"✗ retrieve_conv_outline 调用失败: {type(e).__name__}: {e}")
        return False

async def test_analysist_tool():
    """测试 analysist 是否能够调用工具而不出错"""
    print("\n测试 analysist 工具调用...")
    try:
        from analysist import analysist
        
        # 模拟简单消息
        test_messages = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，有什么可以帮助你的？"},
        ]
        
        formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in test_messages])
        
        # 设置超时，避免长时间阻塞
        result = await asyncio.wait_for(
            analysist.ainvoke({
                "messages": [{"role": "user", "content": formatted_history}]
            }),
            timeout=15.0
        )
        
        print(f"✓ analysist.ainvoke 调用成功")
        return True
        
    except asyncio.TimeoutError:
        print(f"✗ analysist.ainvoke 调用超时 (15秒)")
        return False
    except Exception as e:
        print(f"✗ analysist.ainvoke 调用失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    print("=" * 60)
    print("LangGraph 修复验证测试")
    print("=" * 60)
    
    # 测试检索函数
    results = await asyncio.gather(
        test_retrieve_diary(),
        test_retrieve_materials(),
        test_retrieve_conv_outline(),
        return_exceptions=True
    )
    
    # 测试 analysist（可能调用工具）
    analysist_result = await test_analysist_tool()
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    tests = [
        ("retrieve_diary", results[0]),
        ("retrieve_materials", results[1]),
        ("retrieve_conv_outline", results[2]),
        ("analysist_tool", analysist_result)
    ]
    
    all_passed = True
    for name, result in tests:
        if isinstance(result, Exception):
            print(f"✗ {name}: 异常 {type(result).__name__}: {result}")
            all_passed = False
        elif result is True:
            print(f"✓ {name}: 通过")
        else:
            print(f"✗ {name}: 失败")
            all_passed = False
    
    if all_passed:
        print("\n✅ 所有测试通过！LangGraph 修复成功。")
    else:
        print("\n❌ 部分测试失败。需要进一步调试。")

if __name__ == "__main__":
    asyncio.run(main())