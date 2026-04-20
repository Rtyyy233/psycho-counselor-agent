#!/usr/bin/env python3
"""
检查模型定义
"""

import sys

sys.path.insert(0, "src")

from mem_store_diary import EmotionalState, DiaryChunk, EmotionType
import inspect

print("=== EmotionalState模型 ===")
print(f"EmotionalState.__annotations__: {EmotionalState.__annotations__}")
print(f"EmotionalState.__fields__: {list(EmotionalState.__fields__.keys())}")
emotion_field = EmotionalState.__fields__.get("emotion")
if emotion_field:
    print(f"emotion字段类型: {emotion_field.type_}")
    print(f"emotion字段outer_type_: {emotion_field.outer_type_}")
    print(f"emotion字段.annotation: {emotion_field.annotation}")

print("\n=== DiaryChunk模型 ===")
print(f"DiaryChunk.__annotations__: {DiaryChunk.__annotations__}")
emotions_field = DiaryChunk.__fields__.get("emotions")
if emotions_field:
    print(f"emotions字段类型: {emotions_field.type_}")

print("\n=== EmotionType定义 ===")
print(f"EmotionType: {EmotionType}")
print(f"EmotionType.__args__: {getattr(EmotionType, '__args__', 'N/A')}")

# 测试实例化
print("\n=== 测试实例化 ===")
try:
    es = EmotionalState(emotion=["喜悦", "放松", "安全感"])
    print(f"成功创建EmotionalState: {es}")
    print(f"emotion值: {es.emotion}")
except Exception as e:
    print(f"创建EmotionalState失败: {e}")

try:
    dc = DiaryChunk(
        raw_text="测试",
        outline="测试",
        date="25.03.15",
        emotions=EmotionalState(emotion=["喜悦", "放松", "安全感"]),
        cognitions=...,
        behaviors=...,
        tags=...,
    )
    print(f"成功创建DiaryChunk")
except Exception as e:
    print(f"创建DiaryChunk失败: {e}")
