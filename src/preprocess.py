import re
import os

file_path = os.path.join("data", "frankenstein.txt")

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# Remove Gutenberg header and footer more flexibly
start_match = re.search(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.IGNORECASE)
end_match = re.search(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.IGNORECASE)

if start_match and end_match:
    text = text[start_match.end():end_match.start()]

# Split by Letter or Chapter headings
sections = re.split(r"\n\s*(Letter\s+\d+|Chapter\s+\d+)\s*\n", text, flags=re.IGNORECASE)

clean_sections = []
for i in range(1, len(sections), 2):
    title = sections[i].strip()
    content = sections[i + 1].strip() if i + 1 < len(sections) else ""
    if len(content) > 300:
        clean_sections.append((title, content))

print(f"Total sections found: {len(clean_sections)}")

for i, (title, content) in enumerate(clean_sections[:5], start=1):
    print(f"\n--- Section {i}: {title} ---")
    print(content[:500])

# Save preview
output_file = os.path.join("outputs", "chapter_preview.txt")
with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"Total sections found: {len(clean_sections)}\n\n")
    for i, (title, content) in enumerate(clean_sections[:5], start=1):
        f.write(f"--- Section {i}: {title} ---\n")
        f.write(content[:1000])
        f.write("\n\n")

print(f"\nPreview saved to: {output_file}")