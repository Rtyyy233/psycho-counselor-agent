#!/usr/bin/env python3
"""
诊断脚本：定位检索工具中的 LangGraph 节点错误
错误信息：Expected dict, got semantic_search_node
"""

import asyncio
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_retrieve_diary_tool():
    """测试 retrieve_diary_tool 工具"""
    print("=" * 60)
    print("测试 retrieve_diary_tool...")
    print("=" * 60)
    
    try:
        from mem_integration import retrieve_diary_tool
        
        # 模拟一个简单查询
        query = "测试查询"
        print(f"调用 retrieve_diary_tool('{query}')...")
        
        result = await retrieve_diary_tool.invoke({"query": query})
        print(f"✓ retrieve_diary_tool 调用成功")
        print(f"  返回类型: {type(result)}")
        print(f"  返回值: {result}")
        
        return True
    except Exception as e:
        print(f"✗ retrieve_diary_tool 调用失败")
        print(f"  错误类型: {type(e).__name__}")
        print(f"  错误信息: {e}")
        print("\n详细堆栈:")
        traceback.print_exc()
        return False

async def test_retrieve_materials_tool():
    """测试 retrieve_materials_tool 工具"""
    print("\n" + "=" * 60)
    print("测试 retrieve_materials_tool...")
    print("=" * 60)
    
    try:
        from mem_integration import retrieve_materials_tool
        
        # 模拟一个简单查询
        query = "测试材料查询"
        print(f"调用 retrieve_materials_tool('{query}')...")
        
        result = await retrieve_materials_tool.invoke({"query": query})
        print(f"✓ retrieve_materials_tool 调用成功")
        print(f"  返回类型: {type(result)}")
        print(f"  返回值: {result}")
        
        return True
    except Exception as e:
        print(f"✗ retrieve_materials_tool 调用失败")
        print(f"  错误类型: {type(e).__name__}")
        print(f"  错误信息: {e}")
        
        # 检查是否是我们寻找的错误
        if "Expected dict, got" in str(e):
            print(f"  🚨 发现目标错误: {e}")
            print(f"  错误可能来自: retrieve_materials_tool → retrieve_materials()")
        
        print("\n详细堆栈:")
        traceback.print_exc()
        return False

async def test_retrieve_conv_outline_tool():
    """测试 retrieve_conv_outline_tool 工具"""
    print("\n" + "=" * 60)
    print("测试 retrieve_conv_outline_tool...")
    print("=" * 60)
    
    try:
        from mem_integration import retrieve_conv_outline_tool
        
        # 模拟一个简单查询
        query = "测试对话摘要查询"
        print(f"调用 retrieve_conv_outline_tool('{query}')...")
        
        result = await retrieve_conv_outline_tool.invoke({"query": query})
        print(f"✓ retrieve_conv_outline_tool 调用成功")
        print(f"  返回类型: {type(result)}")
        print(f"  返回值: {result}")
        
        return True
    except Exception as e:
        print(f"✗ retrieve_conv_outline_tool 调用失败")
        print(f"  错误类型: {type(e).__name__}")
        print(f"  错误信息: {e}")
        
        # 检查是否是我们寻找的错误
        if "Expected dict, got" in str(e):
            print(f"  🚨 发现目标错误: {e}")
            print(f"  错误可能来自: retrieve_conv_outline_tool → retrieve_conv_outline()")
        
        print("\n详细堆栈:")
        traceback.print_exc()
        return False

async def test_direct_retrieve_functions():
    """直接测试底层检索函数"""
    print("\n" + "=" * 60)
    print("直接测试底层检索函数...")
    print("=" * 60)
    
    functions_to_test = [
        ("retrieve_diary", "从日记检索"),
        ("retrieve_materials", "从材料检索"), 
        ("retrieve_conv_outline", "从对话摘要检索")
    ]
    
    for func_name, description in functions_to_test:
        print(f"\n--- 测试 {description} ({func_name}) ---")
        try:
            if func_name == "retrieve_diary":
                from mem_retrieve_diary import retrieve_diary
                result = await retrieve_diary("测试查询")
            elif func_name == "retrieve_materials":
                from mem_retrieve_material import retrieve_materials
                result = await retrieve_materials("测试查询")
            elif func_name == "retrieve_conv_outline":
                from mem_retrieve_conv_outline import retrieve_conv_outline
                result = await retrieve_conv_outline("测试查询")
            
            print(f"  ✓ {func_name} 调用成功")
            print(f"    返回类型: {type(result)}")
            if hasattr(result, '__len__'):
                print(f"    结果数量: {len(result)}")
            
        except Exception as e:
            print(f"  ✗ {func_name} 调用失败: {type(e).__name__}: {e}")
            
            # 检查是否是我们寻找的错误
            if "Expected dict, got" in str(e):
                print(f"    🚨 发现目标错误!")
                print(f"    错误来源: {func_name}")
                
                # 进一步分析错误
                error_str = str(e)
                if "semantic_search_node" in error_str:
                    print(f"    问题节点: semantic_search_node")
                elif "metadata_filter_node" in error_str:
                    print(f"    问题节点: metadata_filter_node")
                elif "id_lookup_node" in error_str:
                    print(f"    问题节点: id_lookup_node")
                elif "rerank_node" in error_str:
                    print(f"    问题节点: rerank_node")
                elif "paip_outline_lookup_node" in error_str:
                    print(f"    问题节点: paip_outline_lookup_node")
            
            # 打印简化堆栈
            tb = traceback.extract_tb(e.__traceback__)
            print(f"    错误位置: {tb[-1].filename}:{tb[-1].lineno}")

async def test_analysist_invocation():
    """模拟 analysist 调用环境"""
    print("\n" + "=" * 60)
    print("模拟 analysist 调用环境...")
    print("=" * 60)
    
    try:
        from analysist import analysist
        
        # 创建模拟消息（类似于 call_analysist 中的 formatted_history）
        test_messages = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，有什么可以帮助你的？"},
            {"role": "user", "content": "我感到很焦虑"}
        ]
        
        formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in test_messages])
        
        print(f"调用 analysist.ainvoke() 使用消息: {formatted_history[:50]}...")
        
        # 设置较短的超时，避免长时间阻塞
        result = await asyncio.wait_for(
            analysist.ainvoke({
                "messages": [{"role": "user", "content": formatted_history}]
            }),
            timeout=10.0
        )
        
        print(f"✓ analysist.ainvoke() 调用成功")
        return True
        
    except asyncio.TimeoutError:
        print(f"✗ analysist.ainvoke() 调用超时 (10秒)")
        return False
    except Exception as e:
        print(f"✗ analysist.ainvoke() 调用失败")
        print(f"  错误类型: {type(e).__name__}")
        print(f"  错误信息: {e}")
        
        if "Expected dict, got" in str(e):
            print(f"  🚨 发现目标错误!")
            
            # 分析错误信息
            error_str = str(e)
            if "semantic_search_node" in error_str:
                print(f"  问题工具: 可能调用了含有 semantic_search_node 的检索工具")
            elif "retrieve_" in error_str:
                # 尝试从错误信息中提取工具名称
                import re
                tool_match = re.search(r"retrieve_\w+", error_str)
                if tool_match:
                    print(f"  可能的问题工具: {tool_match.group()}")
        
        print("\n详细堆栈:")
        traceback.print_exc()
        return False

async def check_langgraph_nodes():
    """检查 LangGraph 节点函数定义"""
    print("\n" + "=" * 60)
    print("检查 LangGraph 节点函数定义...")
    print("=" * 60)
    
    # 检查 mem_retrieve_diary.py 中的节点函数
    print("\n1. 检查 mem_retrieve_diary.py:")
    try:
        from mem_retrieve_diary import semantic_search_node
        print(f"  semantic_search_node: {semantic_search_node}")
        print(f"  类型: {type(semantic_search_node)}")
        print(f"  是否为协程: {asyncio.iscoroutinefunction(semantic_search_node)}")
    except Exception as e:
        print(f"  导入失败: {e}")
    
    # 检查 mem_retrieve_conv_outline.py 中的节点函数
    print("\n2. 检查 mem_retrieve_conv_outline.py:")
    try:
        from mem_retrieve_conv_outline import semantic_search_node
        print(f"  semantic_search_node: {semantic_search_node}")
        print(f"  类型: {type(semantic_search_node)}")
        print(f"  是否为协程: {asyncio.iscoroutinefunction(semantic_search_node)}")
    except Exception as e:
        print(f"  导入失败: {e}")
    
    # 检查 mem_retrieve_material.py 中的节点函数
    print("\n3. 检查 mem_retrieve_material.py:")
    try:
        from mem_retrieve_material import semantic_search_children
        print(f"  semantic_search_children: {semantic_search_children}")
        print(f"  类型: {type(semantic_search_children)}")
        print(f"  是否为协程: {asyncio.iscoroutinefunction(semantic_search_children)}")
    except Exception as e:
        print(f"  导入失败: {e}")

async def main():
    """主诊断函数"""
    print("=" * 80)
    print("LangGraph 节点错误诊断工具")
    print("错误: 'Expected dict, got semantic_search_node'")
    print("=" * 80)
    
    # 检查环境
    print("\n[1] 检查 Python 环境和依赖...")
    print(f"Python 版本: {sys.version}")
    print(f"工作目录: {Path.cwd()}")
    
    # 检查节点函数定义
    await check_langgraph_nodes()
    
    # 测试直接检索函数
    await test_direct_retrieve_functions()
    
    # 测试工具函数
    print("\n[2] 测试工具函数调用...")
    tool_results = await asyncio.gather(
        test_retrieve_diary_tool(),
        test_retrieve_materials_tool(),
        test_retrieve_conv_outline_tool(),
        return_exceptions=True
    )
    
    # 模拟实际调用环境
    print("\n[3] 模拟实际调用环境...")
    await test_analysist_invocation()
    
    # 诊断总结
    print("\n" + "=" * 80)
    print("诊断总结")
    print("=" * 80)
    
    print("\n建议下一步:")
    print("1. 如果找到了具体的出错工具，查看对应的检索函数")
    print("2. 检查对应的 semantic_search_node 或其他节点函数")
    print("3. 确保节点函数返回字典(state)，而不是函数对象")
    print("4. 检查 LangGraph 图形构建是否正确")
    
    print("\n可能的问题位置:")
    print("- mem_retrieve_diary.py: semantic_search_node 函数")
    print("- mem_retrieve_conv_outline.py: semantic_search_node 函数") 
    print("- mem_retrieve_material.py: semantic_search_children 函数")
    print("- 对应的 graph.add_node() 调用")

if __name__ == "__main__":
    asyncio.run(main())