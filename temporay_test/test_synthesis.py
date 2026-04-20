import asyncio
import sys
sys.path.insert(0, 'src')

from analysist import synthesize_analysis

async def test_synthesis():
    print("Testing synthesis with empty results...")
    result = await synthesize_analysis(
        query="测试查询",
        diary_results=[],
        material_results=[],
        conv_results=[]
    )
    print(f"Result: {result}")
    
    print("\nTesting synthesis with some dummy results...")
    dummy_results = [[{"page_content": "这是一个测试文档内容", "metadata": {}}]]
    result2 = await synthesize_analysis(
        query="测试查询",
        diary_results=dummy_results,
        material_results=dummy_results,
        conv_results=dummy_results
    )
    print(f"Result2 length: {len(result2)}")

if __name__ == "__main__":
    asyncio.run(test_synthesis())