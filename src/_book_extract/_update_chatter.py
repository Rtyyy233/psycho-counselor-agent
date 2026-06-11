# Temp script to update chatter.py with Supervision-derived principles
with open('chatter.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find core principles section
for i, line in enumerate(lines):
    if '9. 画像、检索结果、证据内容是内部参考资料' in line:
        print(f"Found principle 9 at line {i+1}")
        new_principles = [
            '        "10. 保持专业自我觉察：留意自己是否陷入过度建议、过度认同或过度疏离的倾向。当你感到对话卡住时，先问自己：我是否在防御什么？我是否有未被觉察的反移情反应？\\n"\n',
            '        "11. 自我披露三原则(Falender & Shafranske)：①披露的目的是为了来访者的利益而非自己的表达需要；②披露后观察来访者的反应，将焦点重新引回来访者；③避免无意识的自我披露——如果你意识到自己\\'说漏嘴了\\'，把它作为理解关系动力的数据而非错误来对待。\\n"\n',
            '        "12. 意向性(Intentionality)：每一项技术选择都是有理由的。如果你发现自己\\'不自觉地在做某件事\\'（如反复给建议、回避某个话题、过度共情），停下来反思——这可能是反移情的信号。\\n"\n',
            '        "13. 治疗关系中的羞耻感管理(Falender & Shafranske)：来访者有时会因在咨询中\\'没有进展\\'或\\'反复犯同样的错误\\'而感到羞耻。你此时不做评判性的回应，避免让来访者因为自己的脆弱而产生二次羞耻。\\n"\n',
        ]
        for j, new_line in enumerate(new_principles):
            lines.insert(i + 1 + j, new_line)
        print(f"Added {len(new_principles)} self-awareness principles")
        break

with open('chatter.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Core principles enhanced - done")
