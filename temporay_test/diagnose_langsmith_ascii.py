#!/usr/bin/env python3
"""
诊断LangSmith追踪问题 - ASCII版本
"""
import os
import sys
import traceback
from pathlib import Path

def check_env_file():
    """检查.env文件"""
    print("=" * 60)
    print("1. Check .env file")
    print("=" * 60)
    
    env_path = Path(".env")
    if not env_path.exists():
        print("ERROR: .env file not found")
        return False
    
    with open(env_path, 'r', encoding='utf-8') as f:
        content = f.read()
        print(f".env content:\n{content}")
    
    # 检查关键变量
    lines = content.splitlines()
    issues = []
    
    # 检查LANGCHAIN_TRACING_V2
    tracing_line = [line for line in lines if line.strip().startswith("LANGCHAIN_TRACING_V2")]
    if not tracing_line:
        issues.append("ERROR: Missing LANGCHAIN_TRACING_V2")
    else:
        value = tracing_line[0].split('=', 1)[1].strip()
        if value.lower() != 'true':
            issues.append(f"ERROR: LANGCHAIN_TRACING_V2 should be 'true', got '{value}'")
        else:
            print("OK: LANGCHAIN_TRACING_V2 = true")
    
    # 检查LANGCHAIN_API_KEY
    api_key_line = [line for line in lines if line.strip().startswith("LANGCHAIN_API_KEY")]
    if not api_key_line:
        issues.append("ERROR: Missing LANGCHAIN_API_KEY")
    else:
        value = api_key_line[0].split('=', 1)[1].strip()
        if 'your_langsmith_api_key_here' in value or not value:
            issues.append("ERROR: LANGCHAIN_API_KEY is still placeholder or empty")
            print(f"WARNING: API key: {value}")
            print("  Get real API key from https://smith.langchain.com")
        else:
            print(f"OK: LANGCHAIN_API_KEY is set (length: {len(value)} chars)")
    
    # 检查LANGCHAIN_PROJECT
    project_line = [line for line in lines if line.strip().startswith("LANGCHAIN_PROJECT")]
    if not project_line:
        issues.append("ERROR: Missing LANGCHAIN_PROJECT")
    else:
        value = project_line[0].split('=', 1)[1].strip()
        print(f"OK: LANGCHAIN_PROJECT = {value}")
    
    if issues:
        print("\nISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("\nOK: .env file is correctly configured")
        return True

def check_environment_variables():
    """检查环境变量"""
    print("\n" + "=" * 60)
    print("2. Check environment variables")
    print("=" * 60)
    
    required_vars = [
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_API_KEY", 
        "LANGCHAIN_PROJECT"
    ]
    
    issues = []
    for var in required_vars:
        value = os.environ.get(var)
        if not value:
            issues.append(f"ERROR: {var} not set")
        else:
            print(f"OK: {var} = {value[:20] if len(value) > 20 else value}")
    
    if issues:
        print("\nENVIRONMENT ISSUES:")
        for issue in issues:
            print(f"  {issue}")
        print("\nPossible causes:")
        print("  - Environment variables set in wrong order")
        print("  - load_dotenv() called too late")
        print("  - Python process not restarted after env changes")
        return False
    else:
        print("\nOK: Environment variables are set correctly")
        return True

def check_import_order():
    """检查导入顺序"""
    print("\n" + "=" * 60)
    print("3. Check import order")
    print("=" * 60)
    
    # 检查 main.py 中的导入顺序
    main_path = Path("src/web/main.py")
    if not main_path.exists():
        print(f"ERROR: File not found: {main_path}")
        return False
    
    with open(main_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 查找关键位置
    env_set_line = -1
    langchain_import_line = -1
    
    for i, line in enumerate(lines):
        if 'LANGCHAIN_TRACING_V2' in line or 'LANGCHAIN_API_KEY' in line:
            if env_set_line == -1:
                env_set_line = i
        if 'from langchain' in line or 'import langchain' in line:
            if langchain_import_line == -1:
                langchain_import_line = i
    
    print(f"Env vars set at line: {env_set_line + 1}")
    print(f"LangChain import at line: {langchain_import_line + 1}")
    
    if env_set_line == -1:
        print("ERROR: Environment variables not found")
        return False
    elif langchain_import_line == -1:
        print("ERROR: LangChain imports not found")
        return False
    elif env_set_line > langchain_import_line:
        print("ERROR: Environment variables set AFTER LangChain imports")
        print("  Must set env vars BEFORE all LangChain imports")
        return False
    else:
        print("OK: Environment variables set BEFORE LangChain imports")
        return True

def main():
    print("\n" + "=" * 60)
    print("LangSmith Tracing Diagnostics")
    print("=" * 60)
    
    print("\nImportant: LangSmith tracing requires:")
    print("1. Valid API key (get from https://smith.langchain.com)")
    print("2. LANGCHAIN_TRACING_V2=true")
    print("3. Environment variables set BEFORE LangChain imports")
    print("4. Internet connection")
    
    results = []
    
    # 运行检查
    results.append(("Check .env file", check_env_file()))
    results.append(("Check environment variables", check_environment_variables()))
    results.append(("Check import order", check_import_order()))
    
    print("\n" + "=" * 60)
    print("Diagnosis Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL CHECKS PASSED")
        print("\nNext steps:")
        print("1. Make sure API key is real (not placeholder)")
        print("2. Restart web server: taskkill /F /PID [pid]; python -m src.web.main")
        print("3. Upload file test: http://localhost:8000/upload")
        print("4. Check traces: https://smith.langchain.com/")
    else:
        print("ISSUES FOUND - NEEDS FIXING")
        print("\nCommon solutions:")
        print("1. Replace API key placeholder:")
        print("   Edit .env file, replace 'your_langsmith_api_key_here' with real key")
        print("")
        print("2. Ensure correct import order:")
        print("   Environment variables must be set BEFORE all 'from langchain' imports")
        print("")
        print("3. Restart Python process:")
        print("   Restart web server after changing .env file")
        print("")
        print("4. Check network:")
        print("   Ensure you can access https://api.smith.langchain.com")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())