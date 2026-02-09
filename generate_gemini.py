# generate_gemini.py
from google import genai
import json
import time
import re
import textwrap

# ================= CONFIG =================
API_KEY = "enter_api_key"
MODEL = "models/gemma-3-1b-it"

TARGET_PER_TOPIC = 100
SLEEP_SECONDS = 2

TOPICS = [
    "Social class and inequality",
    "Moral responsibility",
    "Childhood and innocence",
    "Justice and law",
    "Individual versus society"
]
# =========================================

client = genai.Client(api_key=API_KEY)


# ---------- Robust paragraph reconstruction ----------
def split_into_paragraphs(text, min_words=80, max_words=180):
    """
    Converts raw model output into paragraph-sized chunks
    by grouping sentences. Works even if formatting is bad.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    paragraphs = []
    current = []

    for s in sentences:
        if not s.strip():
            continue

        current.append(s)
        wc = sum(len(x.split()) for x in current)

        if min_words <= wc <= max_words:
            paragraphs.append(" ".join(current).strip())
            current = []

        elif wc > max_words:
            current = []

    return paragraphs


# ---------- LLM call ----------
def generate_text(prompt):
    r = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={
            "temperature": 1.4,
            "top_p": 0.9
        }
    )
    return r.text


# ---------- Pretty paragraph formatting ----------
def format_paragraph(text, width=80):
    wrapped = textwrap.fill(text, width=width)
    return wrapped + "\n\n\n"


# ---------- Class 2 ----------
def write_class2():
    with open("data/class2.jsonl", "w") as f:
        for topic in TOPICS:
            collected = 0
            print(f"\n[Class 2] Topic: {topic}")

            while collected < TARGET_PER_TOPIC:
                prompt = (
                    f"Write a long analytical passage about '{topic}'. "
                    "Use neutral modern prose. Avoid literary imitation."
                )

                try:
                    raw = generate_text(prompt)
                    paragraphs = split_into_paragraphs(raw)
                except Exception as e:
                    print("Error (Class 2), retrying:", e)
                    time.sleep(10)
                    continue

                for p in paragraphs:
                    if collected >= TARGET_PER_TOPIC:
                        break

                    f.write(json.dumps({
                        "text": format_paragraph(p),
                        "author": "Gemini-Neutral",
                        "topic": topic,
                        "class": 2
                    }) + "\n")

                    f.flush()
                    collected += 1
                    print(f"[Class 2] {topic}: {collected}/{TARGET_PER_TOPIC}")

                time.sleep(SLEEP_SECONDS)


# ---------- Class 3 ----------
def write_class3():
    with open("data/class3.jsonl", "w") as f:
        for topic in TOPICS:
            collected = 0
            print(f"\n[Class 3] Topic: {topic}")

            while collected < TARGET_PER_TOPIC:
                prompt = (
                    f"Write a reflective literary passage on '{topic}'. "
                    "Mimic the sentence rhythm and stylistic tendencies of Charles Dickens. "
                    "Do NOT reference real characters, plots, or places."
                )

                try:
                    raw = generate_text(prompt)
                    paragraphs = split_into_paragraphs(raw)
                except Exception as e:
                    print("Error (Class 3), retrying:", e)
                    time.sleep(10)
                    continue

                for p in paragraphs:
                    if collected >= TARGET_PER_TOPIC:
                        break

                    f.write(json.dumps({
                        "text": format_paragraph(p),
                        "author": "Gemini-Dickens",
                        "topic": topic,
                        "class": 3
                    }) + "\n")

                    f.flush()
                    collected += 1
                    print(f"[Class 3] {topic}: {collected}/{TARGET_PER_TOPIC}")

                time.sleep(SLEEP_SECONDS)


# ---------- Main ----------
if __name__ == "__main__":
    write_class2()
    write_class3()
