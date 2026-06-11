"""
Rebuild chatter.py system_prompt section with correct formatting.
Reads the current (possibly corrupted) chatter.py, extracts the system prompt
text content, and rewrites it with proper Python string formatting.
"""
import re

def make_line(text):
    """Format a system prompt line as a Python string literal."""
    return f'        "{text}\\n"\n'

def make_section(text):
    """Make a section with trailing double newline."""
    return f'        "{text}\\n\\n"\n'

with open('chatter.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the system_prompt section
prompt_start = content.find('system_prompt=(')
assert prompt_start >= 0, "Cannot find system_prompt start"

# Find: the first occurrence of '        "' after prompt_start
first_line = content.find('\n        "', prompt_start)
assert first_line >= 0, "Cannot find first prompt line"

# Find the end: '    ),' followed by newline and '    tools='
end_line = content.find('\n    tools=[read_file_tool', first_line)
assert end_line >= 0, "Cannot find prompt end"

# Read the raw lines from the prompt section
raw_section = content[first_line:end_line]

# Parse: split into logical lines, merging broken lines
# A broken line pattern is: content\n"\n        "content
# Fix by removing \n"\n        " (joining across the break)
fixed_section = re.sub(r'\n"\n        "', '', raw_section)

# Also fix stray " on its own line
fixed_section = re.sub(r'\n"\n', '\n', fixed_section)

# Now fixed_section has correct format lines
# Let's verify by checking the resulting content
lines = fixed_section.strip().split('\n')
valid_lines = []
for line in lines:
    stripped = line.strip()
    if stripped.startswith('"') and '\\n' in stripped:
        valid_lines.append(stripped)
    elif stripped == '':
        valid_lines.append('')

print(f"Extracted {len(valid_lines)} valid prompt lines")

# Build the clean file
before = content[:first_line]
after = content[end_line:]

# Check what's at the start
# The after should start with '\n    tools='
print(f"After section starts with: {repr(after[:50])}")

# Write fixed content
new_content = before + fixed_section + after

# Verify syntax
import ast
try:
    ast.parse(new_content)
    print("Syntax check: OK!")
    with open('chatter.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("File written successfully")
except SyntaxError as e:
    print(f"Syntax error at line {e.lineno}: {e.msg}")
    # Show context
    lines = new_content.split('\n')
    for i in range(max(0, e.lineno-3), min(len(lines), e.lineno+2)):
        print(f"  {i+1}: {lines[i][:120]}")
