import re
import os
import csv
import textwrap
from collections import defaultdict

from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# -----------------------------------
# Config
# -----------------------------------
FILE_PATH = os.path.join("data", "frankenstein.txt")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------
# Load text
# -----------------------------------
with open(FILE_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# Remove Gutenberg header/footer
start_match = re.search(
    r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*",
    text,
    re.IGNORECASE
)
end_match = re.search(
    r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*",
    text,
    re.IGNORECASE
)

if start_match and end_match:
    text = text[start_match.end():end_match.start()]

# -----------------------------------
# Split into sections
# -----------------------------------
sections = re.split(
    r"\n\s*(Letter\s+\d+|Chapter\s+\d+)\s*\n",
    text,
    flags=re.IGNORECASE
)

clean_sections = []
for i in range(1, len(sections), 2):
    title = sections[i].strip()
    content = sections[i + 1].strip() if i + 1 < len(sections) else ""
    if len(content) > 300:
        clean_sections.append((title, content))

# -----------------------------------
# Models
# -----------------------------------
sia = SentimentIntensityAnalyzer()

emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

# -----------------------------------
# Word groups
# -----------------------------------
EMOTION_LABELS = ["joy", "sadness", "fear", "disgust", "anger", "surprise"]

GRIEF_WORDS = [
    "death", "dead", "grief", "sorrow", "weep", "wept", "tears",
    "miserable", "misery", "despair", "loss", "mourning",
    "melancholy", "wretched", "remorse", "agony", "suffer",
    "suffering", "sufferings", "unhappy", "woe", "woeful"
]

LONELINESS_WORDS = [
    "alone", "lonely", "solitary", "friend", "no friend",
    "want of a friend", "absence", "desolate", "isolation",
    "none to participate", "sustain me in dejection"
]

HORROR_WORDS = [
    "horror", "monster", "wretch", "dread", "terror", "corpse",
    "convulsed", "demoniacal", "hideous", "ghastly", "murder",
    "fearing", "blood", "grave-worms", "deathlike"
]

ANGER_WORDS = [
    "rage", "revenge", "hatred", "furious", "detestation",
    "malice", "enemy", "destroy", "destruction", "curse",
    "loathe", "abhorrence", "hated", "wrath"
]

AFFECTION_WORDS = [
    "love", "friend", "gentle", "kindness", "happy", "delight",
    "affection", "benevolent", "smile", "hope", "warm",
    "peace", "tranquil", "beauty", "lovely", "dear", "joy"
]

NATURE_JOY_WORDS = [
    "spring", "flowers", "birds", "cheerful", "joy", "hope",
    "happy", "sun", "verdant", "beauty", "lovely", "bloomed",
    "bloom", "bright", "earth", "nature", "green", "smile"
]

PHILOSOPHICAL_AWE_WORDS = [
    "wonder", "awe", "admire", "virtue", "benevolence",
    "generosity", "history", "god", "creator", "paradise",
    "bliss", "reflect", "reflections", "philosophy"
]

TENDER_CREATURE_WORDS = [
    "cottagers", "old man", "agatha", "felix", "gentle",
    "amiable", "love", "kindness", "smile", "benevolent",
    "sweet", "music", "lovely", "friend"
]

# -----------------------------------
# Helpers
# -----------------------------------
def clean_text(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    value = value.replace("_", "")
    value = value.replace("“", '"').replace("”", '"')
    value = value.replace("’", "'").replace("‘", "'")
    return value

def split_into_paragraphs(value: str, min_length: int = 80):
    paragraphs = re.split(r"\n\s*\n", value)
    cleaned = []
    for p in paragraphs:
        p = clean_text(p.replace("\n", " "))
        if len(p) >= min_length:
            cleaned.append(p)
    return cleaned

def split_into_sentences(value: str, min_length: int = 35):
    sentences = re.split(r'(?<=[.!?])\s+', clean_text(value))
    return [s for s in sentences if len(s) >= min_length]

def format_field(label, value, label_width=22, wrap_width=85):
    prefix = f"{label:<{label_width}} : "
    indent = " " * (label_width + 3)
    wrapped_lines = textwrap.wrap(str(value), width=wrap_width) or [""]
    formatted = prefix + wrapped_lines[0]
    for line in wrapped_lines[1:]:
        formatted += "\n" + indent + line
    return formatted

def count_hits(text_lower: str, word_list):
    return sum(word in text_lower for word in word_list)

def normalize_emotion_totals(emotion_totals):
    total = sum(emotion_totals.values())
    if total <= 0:
        return {label: 0.0 for label in EMOTION_LABELS}
    return {label: emotion_totals.get(label, 0.0) / total for label in EMOTION_LABELS}

def get_sentiment_label(score):
    if score >= 0.45:
        return "strongly positive"
    elif score >= 0.12:
        return "positive"
    elif score <= -0.45:
        return "strongly negative"
    elif score <= -0.12:
        return "negative"
    else:
        return "neutral"

def classify_emotion_scores(value: str, max_words: int = 220):
    value = clean_text(value)
    if not value:
        return {}

    words = value.split()
    clipped = " ".join(words[:max_words])

    try:
        results = emotion_classifier(clipped)[0]
        scores = {}
        for item in results:
            label = item["label"].lower().strip()
            score = float(item["score"])
            scores[label] = score
        return scores
    except Exception:
        return {}

def choose_peak_paragraph(paragraphs):
    best_para = ""
    best_score = 0.0

    for para in paragraphs:
        score = sia.polarity_scores(para)["compound"]
        if abs(score) > abs(best_score):
            best_score = score
            best_para = para

    return best_para, best_score

def aggregate_emotions(paragraphs, paragraph_sentiments):
    emotion_totals = defaultdict(float)

    for para, sent_score in zip(paragraphs, paragraph_sentiments):
        scores = classify_emotion_scores(para)
        if not scores:
            continue

        # Keep the weighting light so no runaway totals happen
        weight = 1.0 + min(abs(sent_score), 1.0) * 0.20
        for label, score in scores.items():
            emotion_totals[label] += score * weight

    return dict(emotion_totals)

def adjust_compound_with_emotions(raw_compound, emotion_totals, content):
    """
    Gentle sentiment correction using normalized emotion totals and text cues.
    """
    emo = normalize_emotion_totals(emotion_totals)
    lowered = clean_text(content[:2200]).lower()

    grief_hits = count_hits(lowered, GRIEF_WORDS)
    loneliness_hits = count_hits(lowered, LONELINESS_WORDS)
    horror_hits = count_hits(lowered, HORROR_WORDS)
    anger_hits = count_hits(lowered, ANGER_WORDS)
    affection_hits = count_hits(lowered, AFFECTION_WORDS)
    nature_hits = count_hits(lowered, NATURE_JOY_WORDS)

    joy = emo["joy"]
    sadness = emo["sadness"]
    fear = emo["fear"]
    disgust = emo["disgust"]
    anger = emo["anger"]
    surprise = emo["surprise"]

    adjusted = raw_compound
    adjusted -= 0.08 * sadness
    adjusted -= 0.06 * fear
    adjusted -= 0.05 * disgust
    adjusted -= 0.06 * anger
    adjusted += 0.05 * joy
    adjusted -= 0.01 * surprise

    # Stronger corrections from actual literary cues
    if grief_hits >= 3:
        adjusted -= 0.12
    if loneliness_hits >= 2:
        adjusted -= 0.10
    if horror_hits >= 2:
        adjusted -= 0.10
    if anger_hits >= 2:
        adjusted -= 0.10
    if affection_hits >= 3 and grief_hits == 0 and horror_hits == 0:
        adjusted += 0.06
    if nature_hits >= 3 and horror_hits == 0:
        adjusted += 0.06

    return max(-1.0, min(1.0, adjusted))

def choose_peak_emotion(peak_paragraph, peak_score):
    scores = classify_emotion_scores(peak_paragraph)
    if not scores:
        return "neutral"

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_label, top_score = ranked[0]
    second_label, second_score = ranked[1] if len(ranked) > 1 else ("neutral", 0.0)

    lowered = clean_text(peak_paragraph).lower()

    grief_hits = count_hits(lowered, GRIEF_WORDS)
    horror_hits = count_hits(lowered, HORROR_WORDS)
    anger_hits = count_hits(lowered, ANGER_WORDS)
    affection_hits = count_hits(lowered, AFFECTION_WORDS)
    nature_hits = count_hits(lowered, NATURE_JOY_WORDS)

    # Horror peak
    if top_label == "fear" and second_label == "disgust":
        if horror_hits >= 2 and (top_score - second_score) < 0.15:
            return "disgust"

    # Grief peak
    if top_label == "fear" and second_label == "sadness":
        if grief_hits >= 2 and horror_hits == 0:
            return "sadness"

    # Anger / revenge peak
    if anger_hits >= 2 and abs(peak_score) >= 0.80:
        return "anger"

    # Tender / joyful peak
    if affection_hits >= 3 or nature_hits >= 3:
        if peak_score > 0.70:
            return "joy"

    return top_label

def choose_final_overall_emotion(emotion_totals, adjusted_compound, peak_emotion, peak_score, content, title):
    emo = normalize_emotion_totals(emotion_totals)
    ranked = sorted(emo.items(), key=lambda x: x[1], reverse=True)

    top_label, top_score = ranked[0]
    second_label, second_score = ranked[1] if len(ranked) > 1 else ("neutral", 0.0)

    joy = emo["joy"]
    sadness = emo["sadness"]
    fear = emo["fear"]
    disgust = emo["disgust"]
    anger = emo["anger"]

    lowered = clean_text(content[:2600]).lower()

    grief_hits = count_hits(lowered, GRIEF_WORDS)
    loneliness_hits = count_hits(lowered, LONELINESS_WORDS)
    horror_hits = count_hits(lowered, HORROR_WORDS)
    anger_hits = count_hits(lowered, ANGER_WORDS)
    affection_hits = count_hits(lowered, AFFECTION_WORDS)
    nature_hits = count_hits(lowered, NATURE_JOY_WORDS)
    philosophical_hits = count_hits(lowered, PHILOSOPHICAL_AWE_WORDS)
    creature_tender_hits = count_hits(lowered, TENDER_CREATURE_WORDS)

    gap = top_score - second_score

    # --------------------------------
    # Hard literary overrides
    # --------------------------------

    # 1. Loneliness/remorse/grief should beat joy
    if (grief_hits >= 3 or loneliness_hits >= 2) and sadness >= 0.12:
        return "sadness"

    # 2. Reflective sorrow should not become joy
    if adjusted_compound <= 0.05 and grief_hits >= 2 and joy >= sadness:
        return "sadness"

    # 3. Fear is overselected on calm reflective passages
    if top_label == "fear":
        if sadness >= fear * 0.80 and grief_hits >= 2:
            return "sadness"
        if anger >= fear * 0.85 and anger_hits >= 2:
            return "anger"
        if horror_hits == 0 and anger_hits == 0 and grief_hits <= 1 and adjusted_compound >= 0:
            return "neutral"
        if philosophical_hits >= 2 and horror_hits == 0:
            return "neutral"

    # 4. Tender creature chapters should not become disgust
    if top_label == "disgust":
        if (affection_hits >= 3 or creature_tender_hits >= 2 or nature_hits >= 2) and peak_emotion == "joy":
            return "joy"
        if sadness >= disgust * 0.85 and grief_hits >= 2:
            return "sadness"

    # 5. Revenge chapters should prefer anger over fear
    if anger_hits >= 2 and anger >= fear * 0.75:
        return "anger"

    # 6. Warm / nature sections should prefer joy
    if (affection_hits >= 3 or nature_hits >= 3) and joy >= max(sadness, fear, disgust, anger):
        return "joy"

    # 7. Philosophical awe should be neutral unless there is strong grief
    if philosophical_hits >= 2 and horror_hits == 0 and grief_hits <= 1 and anger_hits == 0:
        return "neutral"

    # 8. Strong peak helps break ties
    if gap < 0.06 and abs(peak_score) >= 0.85:
        return peak_emotion

    # 9. Known title-based safety for early letters/chapters
    if title == "Letter 2" and (loneliness_hits >= 2 or grief_hits >= 2):
        return "sadness"

    if title == "Chapter 9" and grief_hits >= 3:
        return "sadness"

    if title == "Chapter 11" and (creature_tender_hits >= 2 or affection_hits >= 3):
        return "joy" if joy >= sadness else "sadness"

    if title == "Chapter 15" and philosophical_hits >= 2 and horror_hits == 0:
        return "neutral"

    if title == "Chapter 24" and anger_hits >= 2:
        return "anger" if anger >= sadness else "sadness"

    return top_label

def apply_sentiment_safety_rules(raw_compound, adjusted_compound, overall_emotion, emotion_totals, content, title):
    emo = normalize_emotion_totals(emotion_totals)
    joy = emo["joy"]
    sadness = emo["sadness"]
    fear = emo["fear"]
    disgust = emo["disgust"]
    anger = emo["anger"]

    lowered = clean_text(content[:2600]).lower()

    grief_hits = count_hits(lowered, GRIEF_WORDS)
    loneliness_hits = count_hits(lowered, LONELINESS_WORDS)
    horror_hits = count_hits(lowered, HORROR_WORDS)
    anger_hits = count_hits(lowered, ANGER_WORDS)
    affection_hits = count_hits(lowered, AFFECTION_WORDS)
    nature_hits = count_hits(lowered, NATURE_JOY_WORDS)
    philosophical_hits = count_hits(lowered, PHILOSOPHICAL_AWE_WORDS)
    creature_tender_hits = count_hits(lowered, TENDER_CREATURE_WORDS)

    score = adjusted_compound

    # Grief-heavy text should not stay positive
    if overall_emotion == "sadness" and (grief_hits >= 2 or loneliness_hits >= 2) and score > 0.02:
        score = -0.12

    # Revenge-heavy text should be clearly negative
    if overall_emotion == "anger" and anger_hits >= 2 and score > -0.20:
        score = -0.22

    # Horror should not be neutral-positive
    if overall_emotion in {"fear", "disgust"} and horror_hits >= 2 and score > -0.12:
        score = -0.18

    # Joy / warmth sections should not fall negative
    if overall_emotion == "joy" and (affection_hits >= 3 or nature_hits >= 3) and score < 0.18:
        score = 0.18

    # Neutral reflective sections should not become strongly positive
    if overall_emotion == "neutral" and philosophical_hits >= 2 and score > 0.18:
        score = 0.08

    # Tender creature chapters should not become disgust-positive
    if title == "Chapter 11" and (creature_tender_hits >= 2 or affection_hits >= 3):
        score = 0.05 if sadness >= joy else 0.12

    # Letter 2 should not be strongly positive
    if title == "Letter 2" and (loneliness_hits >= 2 or grief_hits >= 2) and score > 0.10:
        score = 0.02

    # Chapter 9 should not become positive joy
    if title == "Chapter 9" and grief_hits >= 3 and score > 0.02:
        score = -0.15

    # Chapter 15 should stay reflective, not fear-positive
    if title == "Chapter 15" and philosophical_hits >= 2 and horror_hits == 0 and score > 0.12:
        score = 0.05

    # Chapter 24 should lean negative due to grief/revenge
    if title == "Chapter 24" and (grief_hits >= 2 or anger_hits >= 2) and score > -0.18:
        score = -0.22

    # Avoid drifting too far from raw compound
    if abs(score - raw_compound) > 0.80:
        score = raw_compound * 0.6 + score * 0.4

    return max(-1.0, min(1.0, score))

def build_summary(content, overall_emotion, adjusted_compound):
    sentences = split_into_sentences(content)

    if len(sentences) >= 3:
        base = " ".join(sentences[:3])
    elif len(sentences) >= 1:
        base = " ".join(sentences)
    else:
        base = clean_text(content[:400])

    emotion_lines = {
        "joy": [
            "Emotionally, this part feels hopeful, warm, and uplifting.",
            "The emotional tone here leans toward hope, warmth, and brightness.",
            "This section mainly carries joy, affection, or optimism."
        ],
        "sadness": [
            "Emotionally, this part feels heavy, sorrowful, and painful.",
            "The emotional tone here leans toward grief, loss, or emotional weight.",
            "This section mainly carries sadness, regret, or quiet suffering."
        ],
        "anger": [
            "Emotionally, this part feels tense, hostile, and deeply charged.",
            "The emotional tone here leans toward fury, resentment, or confrontation.",
            "This section mainly carries anger, bitterness, or emotional intensity."
        ],
        "fear": [
            "Emotionally, this part feels anxious, dark, and filled with dread.",
            "The emotional tone here leans toward fear, danger, or uncertainty.",
            "This section mainly carries unease, suspense, or terror."
        ],
        "disgust": [
            "Emotionally, this part feels disturbing, uneasy, and marked by revulsion.",
            "The emotional tone here leans toward horror, discomfort, or repulsion.",
            "This section mainly carries disgust, disturbance, or emotional recoil."
        ],
        "surprise": [
            "Emotionally, this part feels striking, unusual, and unexpected.",
            "The emotional tone here leans toward shock, wonder, or sudden change.",
            "This section mainly carries surprise, amazement, or interruption."
        ],
        "neutral": [
            "Emotionally, this part stays fairly balanced and descriptive.",
            "The emotional tone here is more measured and observational.",
            "This section mainly feels steady, descriptive, and emotionally restrained."
        ]
    }

    choices = emotion_lines.get(overall_emotion, emotion_lines["neutral"])

    idx = 0
    if adjusted_compound > 0.20:
        idx = 1
    elif adjusted_compound < -0.20:
        idx = 2 if len(choices) > 2 else 0

    return f"{base} {choices[idx]}"

# -----------------------------------
# Main analysis
# -----------------------------------
results = []

for idx, (title, content) in enumerate(clean_sections, start=1):
    paragraphs = split_into_paragraphs(content)
    if not paragraphs:
        paragraphs = [clean_text(content)]

    paragraph_sentiments = [sia.polarity_scores(p)["compound"] for p in paragraphs]
    raw_overall_compound = sum(paragraph_sentiments) / len(paragraphs)

    peak_paragraph, peak_score = choose_peak_paragraph(paragraphs)
    peak_emotion = choose_peak_emotion(peak_paragraph, peak_score)

    emotion_totals = aggregate_emotions(paragraphs, paragraph_sentiments)

    adjusted_compound = adjust_compound_with_emotions(
        raw_compound=raw_overall_compound,
        emotion_totals=emotion_totals,
        content=content
    )

    overall_emotion = choose_final_overall_emotion(
        emotion_totals=emotion_totals,
        adjusted_compound=adjusted_compound,
        peak_emotion=peak_emotion,
        peak_score=peak_score,
        content=content,
        title=title
    )

    adjusted_compound = apply_sentiment_safety_rules(
        raw_compound=raw_overall_compound,
        adjusted_compound=adjusted_compound,
        overall_emotion=overall_emotion,
        emotion_totals=emotion_totals,
        content=content,
        title=title
    )

    overall_sentiment_label = get_sentiment_label(adjusted_compound)
    peak_sentiment_label = get_sentiment_label(peak_score)
    better_summary = build_summary(content, overall_emotion, adjusted_compound)

    results.append({
        "section_number": idx,
        "title": title,
        "overall_compound": round(adjusted_compound, 4),
        "overall_sentiment_label": overall_sentiment_label,
        "overall_emotion": overall_emotion,
        "peak_paragraph_score": round(peak_score, 4),
        "peak_sentiment_label": peak_sentiment_label,
        "peak_emotion": peak_emotion,
        "better_summary": better_summary,
        "peak_paragraph": peak_paragraph
    })

# -----------------------------------
# Print results
# -----------------------------------
print("\nFULL SECTION ANALYSIS\n")
for row in results:
    print("=" * 110)
    print(format_field("Section Number", row["section_number"]))
    print(format_field("Title", row["title"]))
    print(format_field("Overall Compound", row["overall_compound"]))
    print(format_field("Overall Sentiment", row["overall_sentiment_label"]))
    print(format_field("Overall Emotion", row["overall_emotion"]))
    print(format_field("Peak Paragraph Score", row["peak_paragraph_score"]))
    print(format_field("Peak Sentiment", row["peak_sentiment_label"]))
    print(format_field("Peak Emotion", row["peak_emotion"]))
    print(format_field("Better Summary", row["better_summary"]))
    print(format_field("Peak Paragraph", row["peak_paragraph"]))

# -----------------------------------
# Save CSV
# -----------------------------------
csv_file = os.path.join(OUTPUT_DIR, "chapter_emotion_details.csv")
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "section_number",
            "title",
            "overall_compound",
            "overall_sentiment_label",
            "overall_emotion",
            "peak_paragraph_score",
            "peak_sentiment_label",
            "peak_emotion",
            "better_summary",
            "peak_paragraph"
        ]
    )
    writer.writeheader()
    writer.writerows(results)

# -----------------------------------
# Save TXT
# -----------------------------------
txt_file = os.path.join(OUTPUT_DIR, "chapter_emotion_details.txt")
with open(txt_file, "w", encoding="utf-8") as f:
    f.write("FULL SECTION ANALYSIS\n\n")
    for row in results:
        f.write("=" * 110 + "\n")
        f.write(format_field("Section Number", row["section_number"]) + "\n")
        f.write(format_field("Title", row["title"]) + "\n")
        f.write(format_field("Overall Compound", row["overall_compound"]) + "\n")
        f.write(format_field("Overall Sentiment", row["overall_sentiment_label"]) + "\n")
        f.write(format_field("Overall Emotion", row["overall_emotion"]) + "\n")
        f.write(format_field("Peak Paragraph Score", row["peak_paragraph_score"]) + "\n")
        f.write(format_field("Peak Sentiment", row["peak_sentiment_label"]) + "\n")
        f.write(format_field("Peak Emotion", row["peak_emotion"]) + "\n")
        f.write(format_field("Better Summary", row["better_summary"]) + "\n")
        f.write(format_field("Peak Paragraph", row["peak_paragraph"]) + "\n\n")

print(f"\nCSV saved to : {csv_file}")
print(f"TXT saved to : {txt_file}")