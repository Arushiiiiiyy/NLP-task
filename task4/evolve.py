# task4/evolve.py

import random
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from google import genai

# ================= CONFIG =================
MODEL_PATH = "models/tier_c_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

API_KEY = "YOUR_GEMINI_API_KEY"
GEMINI_MODEL = "models/gemma-3-1b-it"

POP_SIZE = 10
ELITE_SIZE = 3
GENERATIONS = 7
TARGET_SCORE = 0.90
# =========================================

# ---------- Load Detector ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
detector = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
detector.to(DEVICE)
detector.eval()

# ---------- Load Gemini ----------
client = genai.Client(api_key=API_KEY)

# ---------- Fitness Function ----------
def human_score(text):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    ).to(DEVICE)

    with torch.no_grad():
        logits = detector(**inputs).logits
        probs = torch.softmax(logits, dim=1)

    # assumes label 0 = Human
    return probs[0, 0].item()

# ---------- Initial Population ----------
def generate_initial_population(topic):
    prompt = (
        f"Write {POP_SIZE} independent paragraphs (120–160 words) on '{topic}'.\n"
        "Neutral modern prose.\n"
        "Each paragraph separated by a blank line."
    )

    r = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    paras = [p.strip() for p in r.text.split("\n\n") if len(p.split()) > 80]
    return paras[:POP_SIZE]

# ---------- Mutation ----------
def mutate(text):
    mutation_prompts = [
        "Rewrite this paragraph by changing sentence rhythm while keeping vocabulary.",
        "Rewrite this paragraph introducing a subtle grammatical inconsistency.",
        "Rewrite this paragraph using one rare or archaic word naturally.",
        "Rewrite this paragraph to sound slightly less polished and more human."
    ]

    prompt = (
        random.choice(mutation_prompts)
        + "\n\nPARAGRAPH:\n"
        + text
    )

    r = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    return r.text.strip()

# ---------- Evolution Loop ----------
def evolve(topic):
    population = generate_initial_population(topic)

    for gen in range(1, GENERATIONS + 1):
        scored = [(p, human_score(p)) for p in population]
        scored.sort(key=lambda x: x[1], reverse=True)

        best_text, best_score = scored[0]
        print(f"\nGeneration {gen}")
        print(f"Best Human Score: {best_score:.4f}")

        if best_score >= TARGET_SCORE:
            print("\n✅ SUCCESS — SUPER-IMPOSTER FOUND")
            print(best_text)
            return best_text

        elites = [p for p, _ in scored[:ELITE_SIZE]]

        # generate new population
        population = elites[:]
        while len(population) < POP_SIZE:
            parent = random.choice(elites)
            child = mutate(parent)
            population.append(child)

        time.sleep(2)

    print("\n❌ Target not reached. Best attempt:")
    print(best_text)
    return best_text


if __name__ == "__main__":
    evolve("Social class and inequality")
