#!/usr/bin/env python3
"""
诊断LangSmith追踪问题
"""
import os
import sys
import traceback
from pathlib import Path

def check_env_file():
    """检查.env文件"""
    print("=" * 60)
    print("1. 检查.env文件")
    print("=" * 60)
    
    env_path = Path(".env")
    if not env_path.exists():
        print("❌ .env文件不存在")
        return False
    
    with open(env_path, 'r', encoding='utf-8') as f:
        content = f.read()
        print(f".env内容:\n{content}")
    
    # 检查关键变量
    lines = content.splitlines()
    issues = []
    
    # 检查LANGCHAIN_TRACING_V2
    tracing_line = [line for line in lines if line.strip().startswith("LANGCHAIN_TRACING_V2")]
    if not tracing_line:
        issues.append("❌ 缺少 LANGCHAIN_TRACING_V2")
    else:
        value = tracing_line[0].split('=', 1)[1].strip()
        if value.lower() != 'true':
            issues.append(f"❌ LANGCHAIN_TRACING_V2 应为 'true', 当前为 '{value}'")
        else:
            print("✅ LANGCHAIN_TRACING_V2 = true")
    
    # 检查LANGCHAIN_API_KEY
    api_key_line = [line for line in lines if line.strip().startswith("LANGCHAIN_API_KEY")]
    if not api_key_line:
        issues.append("❌ 缺少 LANGCHAIN_API_KEY")
    else:
        value = api_key_line[0].split('=', 1)[1].strip()
        if 'your_langsmith_api_key_here' in value or not value:
            issues.append("❌ LANGCHAIN_API_KEY 仍然是占位符或为空")
            print(f"⚠️  API密钥: {value}")
            print("   需要从 https://smith.langchain.com 获取真实API密钥")
        else:
            print(f"✅ LANGCHAIN_API_KEY 已设置 (长度: {len(value)} 字符)")
    
    # 检查LANGCHAIN_PROJECT
    project_line = [line for line in lines if line.strip().startswith("LANGCHAIN_PROJECT")]
    if not project_line:
        issues.append("❌ 缺少 LANGCHAIN_PROJECT")
    else:
        value = project_line[0].split('=', 1)[1].strip()
        print(f"✅ LANGCHAIN_PROJECT = {value}")
    
    if issues:
        print("\n⚠️ 发现问题:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("\n✅ .env文件配置正确")
        return True

def check_environment_variables():
    """检查环境变量"""
    print("\n" + "=" * 60)
    print("2. 检查环境变量")
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
            issues.append(f"❌ {var} 未设置")
        else:
            print(f"✅ {var} = {value[:20] if len(value) > 20 else value}")
    
    if issues:
        print("\n⚠️ 环境变量问题:")
        for issue in issues:
            print(f"  {issue}")
        print("\n可能原因:")
        print("  - 环境变量设置顺序错误")
        print("  - load_dotenv() 调用太晚")
        print("  - Python进程重启后环境变量未重新加载")
        return False
    else:
        print("\n✅ 环境变量设置正确")
        return True

def check_import_order():
    """检查导入顺序"""
    print("\n" + "=" * 60)
    print("3. 检查导入顺序")
    print("=" * 60)
    
    # 检查 main.py 中的导入顺序
    main_path = Path("src/web/main.py")
    if not main_path.exists():
        print(f"❌ 文件不存在: {main_path}")
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
    
    print(f"环境变量设置位置: 第 {env_set_line + 1} 行")
    print(f"LangChain导入位置: 第 {langchain_import_line + 1} 行")
    
    if env_set_line == -1:
        print("❌ 未找到环境变量设置")
        return False
    elif langchain_import_line == -1:
        print("❌ 未找到LangChain导入")
        return False
    elif env_set_line > langchain_import_line:
        print("❌ 环境变量设置在LangChain导入之后")
        print("   必须在所有LangChain导入之前设置环境变量")
        return False
    else:
        print("✅ 环境变量设置在LangChain导入之前")
        return True

def test_langchain_import():
    """测试LangChain导入"""
    print("\n" + "=" * 60)
    print("4. 测试LangChain导入")
    print("=" * 60)
    
    try:
        # 模拟正确的导入顺序
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root / "src"))
        
        # 先设置环境变量
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = "test_key"  # 测试用
        
        # 然后导入
        from langchain_core.messages import HumanMessage
        print("✅ 成功导入 langchain_core.messages")
        
        from langchain_deepseek import ChatDeepSeek
        print("✅ 成功导入 langchain_deepseek")
        
        print("\n✅ LangChain导入测试成功")
        return True
        
    except Exception as e:
        print(f"❌ LangChain导入失败: {e}")
        traceback.print_exc()
        return False

def main():
    print("\n" + "=" * 60)
    print("LangSmith追踪问题诊断")
    print("=" * 60)
    
    print("\n重要提示: LangSmith追踪需要:")
    print("1. 有效的API密钥 (从 https://smith.langchain.com 获取)")
    print("2. LANGCHAIN_TRACING_V2=true")
    print("3. 环境变量在导入LangChain之前设置")
    print("4. 互联网连接")
    
    results = []
    
    # 运行检查
    results.append(("检查.env文件", check_env_file()))
    results.append(("检查环境变量", check_environment_variables()))
    results.append(("检查导入顺序", check_import_order()))
    results.append(("测试LangChain导入", test_langchain_import()))
    
    print("\n" + "=" * 60)
    print("诊断总结")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有检查通过")
        print("\n下一步:")
        print("1. 确保API密钥已替换为真实密钥")
        print("2. 重启Web服务器: taskkill /F /PID [pid]; python -m src.web.main")
        print("3. 上传文件测试: http://localhost:8000/upload")
        print("4. 查看追踪: https://smith.langchain.com/")
    else:
        print("❌ 发现问题需要修复")
        print("\n常见问题解决方案:")
        print("1. 替换API密钥占位符:")
        print("   编辑 .env 文件，将 'your_langsmith_api_key_here' 替换为真实密钥")
        print("")
        print("2. 确保导入顺序正确:")
        print("   环境变量设置必须在所有 'from langchain' 导入之前")
        print("")
        print("3. 重启Python进程:")
        print("   环境变量更改后需要重启Web服务器")
        print("")
        print("4. 检查网络连接:")
        print("   确保可以访问 https://api.smith.langchain.com")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())