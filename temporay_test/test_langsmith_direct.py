#!/usr/bin/env python3
"""
直接测试LangSmith追踪功能
"""
import os
import sys
import tempfile
from pathlib import Path

# 设置路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# 检查环境变量
print("=== LangSmith Environment Check ===")
print(f"Current directory: {os.getcwd()}")
print(f".env exists: {(project_root / '.env').exists()}")

# 从.env文件读取内容
env_path = project_root / ".env"
if env_path.exists():
    with open(env_path, 'r', encoding='utf-8') as f:
        env_content = f.read()
        print(f"\n.env content:\n{env_content}")
        
        # 检查关键变量
        lines = env_content.splitlines()
        langchain_lines = [line for line in lines if 'LANGCHAIN' in line.upper()]
        print(f"\nLangSmith variables in .env:")
        for line in langchain_lines:
            print(f"  {line.strip()}")

print("\n" + "="*60)
print("IMPORTANT NOTE:")
print("For LangSmith to work, you MUST replace 'your_langsmith_api_key_here'")
print("with your actual LangSmith API key from https://smith.langchain.com")
print("="*60 + "\n")

# 测试直接导入并运行store_diary工具
try:
    print("\n=== Testing LangChain Import ===")
    
    # 检查当前进程的环境变量
    print("Environment variables in current process:")
    for var in ['LANGCHAIN_TRACING_V2', 'LANGCHAIN_API_KEY', 'LANGCHAIN_PROJECT']:
        value = os.environ.get(var, 'Not set')
        print(f"  {var}: {value}")
        if var == 'LANGCHAIN_API_KEY' and value == 'your_langsmith_api_key_here':
            print(f"    WARNING: API key is still the placeholder! Replace it in .env file")
    
    # 创建一个测试文件
    test_content = "2024-12-01\n测试LangSmith追踪功能。"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(test_content)
        temp_path = f.name
    
    try:
        print(f"\nCreated test file: {temp_path}")
        
        # 尝试导入工具
        print("\n=== Testing Tool Import ===")
        from mem_store_diary import store_diary
        print("Successfully imported store_diary tool")
        
        # 检查工具信息
        print(f"Tool name: {store_diary.name}")
        print(f"Tool description: {store_diary.description}")
        
        print("\n=== LangSmith Status ===")
        print("If LangSmith is properly configured:")
        print("1. Make sure LANGCHAIN_API_KEY is your REAL API key (not placeholder)")
        print("2. Make sure LANGCHAIN_TRACING_V2=true")
        print("3. Upload a file via http://localhost:8000/upload")
        print("4. Check https://smith.langchain.com/ for traces")
        print("\nYou can also test by running: python test_tool_call.py")
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
            print(f"\nCleaned up test file")
            
except ImportError as e:
    print(f"Import error: {e}")
    print("\nTroubleshooting steps:")
    print("1. Make sure src/ is in sys.path")
    print("2. Check if all dependencies are installed: pip install -r requirements.txt")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Summary ===")
print("LangSmith tracing requires:")
print("1. Valid API key in .env (LANGCHAIN_API_KEY=your_real_key)")
print("2. Tracing enabled (LANGCHAIN_TRACING_V2=true)")
print("3. Environment variables set BEFORE LangChain imports")
print("4. Internet connection to LangSmith servers")