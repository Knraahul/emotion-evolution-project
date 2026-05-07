import re
import os
import csv
from nltk.sentiment import SentimentIntensityAnalyzer

file_path = os.path.join("data", "frankenstein.txt")

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

start_match = re.search(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.IGNORECASE)
end_match = re.search(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.IGNORECASE)

if start_match and end_match:
    text = text[start_match.end():end_match.start()]

sections = re.split(r"\n\s*(Letter\s+\d+|Chapter\s+\d+)\s*\n", text, flags=re.IGNORECASE)

clean_sections = []
for i in range(1, len(sections), 2):
    title = sections[i].strip()
    content = sections[i + 1].strip() if i + 1 < len(sections) else ""
    if len(content) > 300:
        clean_sections.append((title, content))

sia = SentimentIntensityAnalyzer()

results = []
for idx, (title, content) in enumerate(clean_sections, start=1):
    scores = sia.polarity_scores(content)
    results.append({
        "section_number": idx,
        "title": title,
        "negative": scores["neg"],
        "neutral": scores["neu"],
        "positive": scores["pos"],
        "compound": scores["compound"]
    })

print("\nFirst 5 results:\n")
for row in results[:5]:
    print(row)

output_file = os.path.join("outputs", "sentiment_results.csv")
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["section_number", "title", "negative", "neutral", "positive", "compound"]
    )
    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved to: {output_file}")