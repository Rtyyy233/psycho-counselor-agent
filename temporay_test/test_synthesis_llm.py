import asyncio
import time
import sys
sys.path.insert(0, 'src')

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import SystemMessage, HumanMessage

async def test_llm_with_actual_prompt():
    # Create a prompt similar to what synthesize_analysis creates
    query = "我感到很不安"
    context = """[日记] 用户提到最近工作压力大，睡眠不好。
[材料] 关于焦虑管理的材料，建议深呼吸和正念练习。
[咨询记录] 上次咨询讨论了应对压力的策略，用户尝试了正念但觉得难以坚持。"""
    
    prompt = f"""基于以下相关记忆，分析用户的查询并提供专业心理分析：

用户查询：{query}

相关记忆片段：
{context}

请提供：
1. 对用户当前情绪状态和认知模式的分析
2. 记忆片段中发现的模式和关联
3. 建议的下一步探索方向

请用中文回复，保持专业且富有洞察力。"""
    
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0.2)
    
    print(f"Testing LLM with prompt length: {len(prompt)}")
    print("Prompt preview:", prompt[:200], "...")
    
    start = time.time()
    try:
        response = await asyncio.wait_for(
            llm.ainvoke([
                SystemMessage(
                    content="你是一位专业的心理分析专家。根据提供的记忆片段进行分析。"
                ),
                HumanMessage(content=prompt),
            ]),
            timeout=10.0
        )
        elapsed = time.time() - start
        print(f"LLM responded in {elapsed:.2f} seconds")
        print(f"Response length: {len(response.content) if hasattr(response, 'content') else len(str(response))}")
        print(f"Response preview: {response.content[:200] if hasattr(response, 'content') else str(response)[:200]}")
    except asyncio.TimeoutError:
        elapsed = time.time() - start
        print(f"LLM timed out after {elapsed:.2f} seconds")
    except Exception as e:
        elapsed = time.time() - start
        print(f"LLM error after {elapsed:.2f} seconds: {e}")

if __name__ == "__main__":
    asyncio.run(test_llm_with_actual_prompt())