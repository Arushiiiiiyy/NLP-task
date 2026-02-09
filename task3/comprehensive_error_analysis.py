"""
Comprehensive Error Analysis
Analyzes misclassifications across all three classes
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import statistics

# ---------- CONFIG ----------
BASE_MODEL = "distilbert-base-uncased"
ADAPTER_PATH = "models/tier_c"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 256
# ----------------------------

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Load model with LoRA
base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=2
)

try:
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print("✓ Model loaded successfully\n")
except Exception as e:
    print(f"Warning: {e}")
    model = base_model

model.to(DEVICE)
model.eval()

def predict(text):
    """Returns AI probability (0-1)"""
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)

    return probs[0, 1].item()  # AI probability

def analyze_class(filepath, class_name, expected_label):
    """Analyze a single class file"""
    print(f"\n{'='*70}")
    print(f"Analyzing {class_name}")
    print(f"{'='*70}")
    
    scores = []
    misclassified = []
    
    with open(filepath) as f:
        for idx, line in enumerate(f):
            obj = json.loads(line)
            text = obj["text"]
            ai_prob = predict(text)
            scores.append(ai_prob)
            
            # Check if misclassified
            is_ai_predicted = ai_prob > 0.5
            if is_ai_predicted != expected_label:
                misclassified.append((ai_prob, text, obj.get("author", "Unknown"), obj.get("topic", "Unknown")))
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1} samples...")
    
    # Print stats
    print(f"\nStatistics for {class_name}:")
    print(f"  Total samples: {len(scores)}")
    print(f"  Misclassified: {len(misclassified)}")
    print(f"  Accuracy: {100 * (1 - len(misclassified) / len(scores)):.2f}%")
    print(f"\n  Score Distribution:")
    print(f"    Min:    {min(scores):.4f}")
    print(f"    Max:    {max(scores):.4f}")
    print(f"    Mean:   {statistics.mean(scores):.4f}")
    print(f"    Median: {statistics.median(scores):.4f}")
    print(f"    StdDev: {statistics.stdev(scores) if len(scores) > 1 else 0:.4f}")
    
    # Show misclassified samples
    if len(misclassified) > 0:
        print(f"\nTop 3 misclassified samples:")
        misclassified.sort(reverse=True, key=lambda x: abs(x[0] - 0.5))  # Sort by confidence
        
        for i, (score, text, author, topic) in enumerate(misclassified[:3], 1):
            print(f"\n  Sample {i}: AI prob = {score:.4f}")
            print(f"    Author: {author}, Topic: {topic}")
            print(f"    Text: {text[:300]}...")
    else:
        print(f"\n  ✓ All samples correctly classified!")
    
    return scores, len(misclassified)

# Run analysis on all three classes
print("=" * 70)
print("COMPREHENSIVE ERROR ANALYSIS")
print("=" * 70)

human_scores, human_err = analyze_class("data/class1.jsonl", "Class 1: Human Text", False)
ai_generic_scores, ai_gen_err = analyze_class("data/class2.jsonl", "Class 2: AI Generic", True)
ai_author_scores, ai_auth_err = analyze_class("data/class3.jsonl", "Class 3: AI Author-Style", True)

# Overall summary
print(f"\n{'='*70}")
print("OVERALL SUMMARY")
print(f"{'='*70}")

total_samples = len(human_scores) + len(ai_generic_scores) + len(ai_author_scores)
total_errors = human_err + ai_gen_err + ai_auth_err
overall_accuracy = 100 * (1 - total_errors / total_samples)

print(f"Total samples: {total_samples}")
print(f"Total errors: {total_errors}")
print(f"Overall accuracy: {overall_accuracy:.2f}%")

print(f"\nError breakdown:")
print(f"  Human misclassified as AI: {human_err} / {len(human_scores)}")
print(f"  AI (generic) misclassified as human: {ai_gen_err} / {len(ai_generic_scores)}")
print(f"  AI (author) misclassified as human: {ai_auth_err} / {len(ai_author_scores)}")

# Score separation analysis
print(f"\nScore separation:")
print(f"  Human scores: {statistics.mean(human_scores):.4f} ± {statistics.stdev(human_scores):.4f}")
print(f"  AI (generic) scores: {statistics.mean(ai_generic_scores):.4f} ± {statistics.stdev(ai_generic_scores):.4f}")
print(f"  AI (author) scores: {statistics.mean(ai_author_scores):.4f} ± {statistics.stdev(ai_author_scores):.4f}")

print(f"\n{'='*70}\n")
