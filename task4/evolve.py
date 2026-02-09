# task4/evolve.py

import random
import time
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from google import genai

MODEL_PATH = "models/tier_c"
BASE_MODEL = "distilbert-base-uncased"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load API key from environment
API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

if API_KEY == "YOUR_GEMINI_API_KEY" and not DEMO_MODE:
    raise ValueError(
        "Gemini API key not found. Please set the GEMINI_API_KEY environment variable:\n"
        "  export GEMINI_API_KEY='your-actual-api-key'\n"
        "Or run in demo mode:\n"
        "  export DEMO_MODE=true\n"
        "Then run this script again."
    )

POP_SIZE = 10
ELITE_SIZE = 3
GENERATIONS = 7
TARGET_SCORE = 0.90

# Load detector
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
detector = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
detector.to(DEVICE)
detector.eval()

if not DEMO_MODE:
    client = genai.Client(api_key=API_KEY)
    GEMINI_MODEL = "models/gemma-3-1b-it"
else:
    print("⚠️  Running in DEMO MODE (no Gemma API calls)")
    client = None

# Demo paragraphs (bypasses API quota)
DEMO_PARAGRAPHS = [
    "The intricate structure of social hierarchies has long fascinated scholars across disciplines. In modern society, the stratification of individuals based on economic resources creates distinct barriers to mobility and opportunity. Those born into privileged circumstances often maintain their advantage through networks and institutional access. Meanwhile, those from disadvantaged backgrounds face systematic obstacles that compound across generations. Education, employment, and wealth accumulation remain profoundly unequal. Research consistently demonstrates that family background predicts life outcomes more strongly than individual merit.",
    "Inequality manifests across multiple dimensions of human experience. Economic disparity represents only one facet of a broader pattern affecting health, education, and social participation. Individuals with greater resources can access superior schools, healthcare, and professional networks. The cycle perpetuates itself as advantaged children inherit both tangible wealth and cultural capital. Disadvantaged communities struggle with underfunded institutions and limited opportunity networks. Breaking these patterns requires sustained effort and systemic change.",
    "The debate about meritocracy often obscures structural reality. While talent and work ethic matter, they operate within contexts of vastly unequal resource distribution. Children from wealthy families receive tutoring, internships, and connections that poorer peers cannot afford. Universities perpetuate advantage by drawing disproportionately from affluent schools. The job market favors those with existing networks and family connections. Thus, outcome inequality reflects not just individual differences but structural disadvantage.",
    "Social mobility rates have stagnated in many developed nations. The belief that anyone can succeed through hard work confronts the reality of persistent inequality across generations. Research shows intergenerational income correlation remains stubbornly high. Many individuals born into poverty remain trapped despite personal effort. Meanwhile, those born wealthy rarely experience significant downward mobility. This pattern suggests that individual determination alone cannot overcome structural barriers.",
    "Class affects access to fundamental resources and opportunities. Healthcare disparities by income level directly impact life expectancy and quality of life. Educational funding varies dramatically by neighborhood wealth, creating unequal preparation for adulthood. Wealthier communities invest in their schools while poor districts struggle with inadequate budgets. Housing segregation concentrates disadvantage geographically. These interconnected inequalities compound across the lifespan.",
]

def human_score(text):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    ).to(DEVICE)

    with torch.no_grad():
        logits = detector(**inputs).logits
        probs = torch.softmax(logits, dim=1)

    # assumes label 0 = Human
    return probs[0, 0].item()

def generate_initial_population(topic):
    if DEMO_MODE:
        # Use pre-generated demo paragraphs
        return random.sample(DEMO_PARAGRAPHS, min(POP_SIZE, len(DEMO_PARAGRAPHS)))
    
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

def mutate(text):
    if DEMO_MODE:
        # Simple demo mutation: add a sentence variation
        sentences = text.split(". ")
        if len(sentences) > 1:
            sentences[0] = "In contemporary society, " + sentences[0].lower()
        return ". ".join(sentences)
    
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

        time.sleep(1 if DEMO_MODE else 2)

    print("\n❌ Target not reached. Best attempt:")
    print(best_text)
    return best_text


if __name__ == "__main__":
    print("Starting Task 4: Genetic Algorithm to evolve human-like text...")
    evolve("Social class and inequality")
