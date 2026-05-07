import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------
# Paths
# -----------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV = os.path.join(BASE_DIR, "outputs", "chapter_emotion_details.csv")
SENTIMENT_PLOT = os.path.join(BASE_DIR, "outputs", "sentiment_flow.png")
EMOTION_PLOT = os.path.join(BASE_DIR, "outputs", "emotion_flow.png")

# -----------------------------------
# Load CSV
# -----------------------------------
df = pd.read_csv(INPUT_CSV)

# -----------------------------------
# Sentiment Flow Plot
# -----------------------------------
plt.figure(figsize=(14, 6))
plt.plot(df["section_number"], df["overall_compound"], marker="o")
plt.axhline(y=0, linestyle="--")
plt.xlabel("Section Number")
plt.ylabel("Overall Compound Score")
plt.title("Emotional Flow of Frankenstein - Sentiment by Section")
plt.grid(True)
plt.tight_layout()
plt.savefig(SENTIMENT_PLOT, dpi=300)
plt.show()

# -----------------------------------
# Emotion Flow Plot
# -----------------------------------
emotion_map = {
    "joy": 1,
    "neutral": 2,
    "surprise": 3,
    "sadness": 4,
    "fear": 5,
    "disgust": 6,
    "anger": 7
}

df["emotion_num"] = df["overall_emotion"].map(emotion_map)

plt.figure(figsize=(14, 6))
plt.plot(df["section_number"], df["emotion_num"], marker="o")
plt.yticks(list(emotion_map.values()), list(emotion_map.keys()))
plt.xlabel("Section Number")
plt.ylabel("Dominant Emotion")
plt.title("Emotional Journey Across Frankenstein")
plt.grid(True)
plt.tight_layout()
plt.savefig(EMOTION_PLOT, dpi=300)
plt.show()

print(f"Sentiment graph saved to: {SENTIMENT_PLOT}")
print(f"Emotion graph saved to: {EMOTION_PLOT}")