#!/usr/bin/env python3
"""
测试web模块导入
"""

import sys
import os

# 模拟运行环境
sys.path.insert(0, "src")
print(f"sys.path: {sys.path[:3]}")

try:
    # 尝试导入
    from web.main import app

    print("✓ 成功导入app")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    import traceback

    traceback.print_exc()

print("\n尝试相对导入...")
try:
    # 检查web目录是否被识别为包
    import web

    print(f"web: {web}")
    print(f"web.__file__: {web.__file__}")
    print(f"web.__path__: {web.__path__}")

    from web.session_manager import SessionManager

    print("✓ 成功导入SessionManager")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    import traceback

    traceback.print_exc()

print("\n作为模块运行测试...")
try:
    # 模拟模块运行
    import importlib

    spec = importlib.util.spec_from_file_location("web.main", "src/web/main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["web.main"] = module
    spec.loader.exec_module(module)
    print("✓ 作为模块导入成功")
except Exception as e:
    print(f"✗ 模块导入失败: {e}")
    import traceback

    traceback.print_exc()
