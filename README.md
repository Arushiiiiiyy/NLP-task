# The Ghost in the Machine: AI Authorship Detection via Genetic Algorithms

A comprehensive machine learning project to distinguish AI-generated text from human-written prose using multi-tiered classifiers and evolutionary text optimization.

**Research Question:** *Can we detect and evolve AI-generated text at scale using statistical, semantic, and transformer-based detectors?*

---

## Table of Contents

1. [Overview](#overview)
2. [Setup & Installation](#setup--installation)
3. [API Configuration](#api-configuration)
4. [Project Structure](#project-structure)
5. [Running the Tasks](#running-the-tasks)
6. [Outputs & Results](#outputs--results)
7. [Architecture](#architecture)

---

## Overview

This project implements the full pipeline described in **"The Ghost in the Machine"** assignment:

- **Task 0:** Build a labeled dataset (human vs. AI text)
- **Task 1:** Linguistic analysis (lexical, syntactic, readability features)
- **Task 2:** Train three-tier detectors (Statistical, Neural, Transformer-based)
- **Task 3:** Explainability via SHAP saliency mapping
- **Task 4:** Genetic Algorithm to evolve "super-imposters" (AI text that fools classifiers)

**Models Used:**
- **Gemma 3-1B** (Google) for AI text generation
- **DistilBERT** with LoRA for fine-tuned classification
- **XGBoost** and **Neural Networks** for baselines

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- macOS / Linux / Windows
- ~4GB free disk space (for models + data)
- Internet connection (for API & model downloads)

### Step 1: Clone & Navigate

```bash
cd /Users/arushishukla/Desktop/NLP-task
```

### Step 2: Install Dependencies

```bash
pip install -r dependencies.txt
```

**Key packages:**
- `transformers` (HuggingFace)
- `torch` (PyTorch)
- `xgboost` (gradient boosting)
- `spacy` (NLP pipeline)
- `google-genai` (Google API)
- `peft` (LoRA fine-tuning)
- `shap` (explainability)
- `scikit-learn` (ML utilities)

---

## API Configuration

### Google Gemma/Gemini API Key

1. **Get your API key:**
   - Visit [ai.google.dev](https://ai.google.dev)
   - Create a new API key (free tier available)
   - Copy the key

2. **Set environment variable:**

```bash
export GEMINI_API_KEY='your-actual-api-key-here'
```

3. **Verify it's set:**

```bash
echo $GEMINI_API_KEY
```

⚠️ **Note:** Tasks 0, 2, 3, and 4 require this key. Free tier has rate limits (~15 req/min). If you hit quota limits, wait ~1 hour or upgrade to paid plan.

---

## Project Structure

```
NLP-task/
├── README.md                          # This file
├── dependencies.txt                   # pip packages
├── SETUP_GUIDE.md                     # Detailed setup
├── QUICKSTART.md                      # Quick start guide
├── PROJECT_GUIDE.md                   # Task descriptions
│
├── data/                              # All datasets
│   ├── class1.jsonl                   # Human-written text (Dickens, Austen)
│   ├── class2.jsonl                   # AI-neutral (Gemma)
│   ├── class3.jsonl                   # AI-styled (Gemma + Dickens prompt)
│   ├── topics_config.json
│   ├── clean/                         # Cleaned source texts
│   │   ├── austen_clean.txt
│   │   └── dickens_clean.txt
│   └── raw/                           # Original Project Gutenberg texts
│       ├── austen_*.txt
│       └── dickens_*.txt
│
├── models/                            # Trained classifiers
│   ├── tier_c/                        # DistilBERT + LoRA (Task 2)
│   │   ├── config.json
│   │   ├── adapter_model.safetensors
│   │   └── checkpoint-*/
│   ├── tierA_rf.py                    # Random Forest (Tier A)
│   ├── tierB_ffnn.py                  # Feedforward NN (Tier B)
│   └── tierC_transformer.py           # DistilBERT LoRA (Tier C)
│
├── features/                          # Linguistic feature extraction
│   ├── lexical.py                     # TTR, hapax legomena
│   ├── syntactic.py                   # POS ratios, dependency depth
│   ├── readability.py                 # Flesch-Kincaid, ARI
│   └── task1_features.csv             # Computed features (Task 1)
│
├── analysis/                          # Post-hoc analysis
│   ├── error_analysis.py              # Misclassified samples
│   └── shap_analysis.py               # SHAP saliency maps
│
├── task3/                             # Task 3: Explainability
│   ├── saliency_shap.py               # SHAP integration
│   ├── error_analysis.py
│   └── comprehensive_error_analysis.py
│
├── task4/                             # Task 4: Genetic Algorithm
│   ├── evolve.py                      # Main GA script
│   └── personal_test.py               # Test your own writing
│
├── utils/                             # Data utilities
│   ├── build_class1.py
│   ├── build_class1_multi_author.py
│   ├── embeddings.py
│   └── feature_extraction.py
│
├── ga/                                # Genetic algorithm utilities
│   └── evolve.py
│
└── main.py                            # Master pipeline
```

---

## Running the Tasks

### Task 0: Dataset Creation

**Status:** ✅ Complete (data already downloaded & cleaned)

If you need to regenerate:

```bash
# Download & clean Project Gutenberg texts
python download_texts.py
python preprocess_gutenberg.py
python clean_text.py

# Generate AI paragraphs (Gemma)
export GEMINI_API_KEY='your-key'
python generate_gemini.py
```

**Output:**
- `data/class1.jsonl` (500 human paragraphs)
- `data/class2.jsonl` (500 AI-neutral paragraphs)
- `data/class3.jsonl` (500 AI-styled paragraphs)

---

### Task 1: Linguistic Fingerprinting

**Status:** ✅ Complete

Run feature extraction:

```bash
python features/lexical.py
python features/syntactic.py
python features/readability.py
```

**Outputs:**
- `features/task1_features.csv` (numerical feature matrix)
- Console: TTR, hapax legomena, POS ratios, dependency depths, readability scores

**Key Findings:**
- Lexical richness (TTR) differs between human/AI
- AI texts have higher adjective-to-noun ratios
- Readability (Flesch-Kincaid) varies by author

---

### Task 2: Multi-Tier Detector Training

**Status:** ✅ Complete (models saved in `models/tier_c/`)

#### Tier A: Random Forest (Statistical Features)

```bash
python models/tierA_rf.py
```

Uses features from Task 1 (TTR, POS ratios, etc.). Fast, interpretable.

#### Tier B: Feedforward Neural Network (Embeddings)

```bash
python models/tierB_ffnn.py
```

Uses GloVe embeddings. Moderate performance.

#### Tier C: DistilBERT + LoRA (Transformer)

```bash
python models/tierC_transformer.py
```

Fine-tunes DistilBERT with Low-Rank Adaptation. Best performance but slowest.

**Expected Accuracy:** ~75-85% on test set

---

### Task 3: Explainability & Error Analysis

```bash
python task3/saliency_shap.py
python task3/error_analysis.py
python task3/comprehensive_error_analysis.py
```

**Outputs:**
- SHAP saliency maps highlighting "AI-ness" tokens
- Error case analysis (human misclassified as AI, etc.)
- Findings: Model picks up on repetitive phrasing, formal tone

---

### Task 4: Genetic Algorithm Evolution

**Default (uses Gemma API):**

```bash
export GEMINI_API_KEY='your-key'
python task4/evolve.py
```

Evolves AI text over 7 generations to maximize "human" probability. Mutations include:
- Sentence rhythm variation
- Subtle grammatical inconsistencies
- Archaic word insertion
- Reduced polish

**Demo Mode (no API calls, instant):**

```bash
export DEMO_MODE=true
python task4/evolve.py
```

Uses pre-generated paragraphs and simple mutations.

#### Test Your Own Writing

```bash
python task4/personal_test.py < your_essay.txt
```

**Output:** "Human confidence score" for your text

---

## Outputs & Results

### After Running All Tasks

```
features/task1_features.csv          # Lexical/syntactic/readability features
models/tier_c/                       # Saved transformer model weights
data/class{1,2,3}.jsonl              # Labeled datasets
```

### Sample Outputs

**Task 1 (Example Statistics):**
```
Human (Class 1):
  - TTR: 0.42
  - Hapax Legomena: 234
  - Adj/Noun Ratio: 0.31
  - Flesch-Kincaid Grade: 9.2

AI-Neutral (Class 2):
  - TTR: 0.38 (lower)
  - Hapax Legomena: 198
  - Adj/Noun Ratio: 0.42 (higher)
  - Flesch-Kincaid Grade: 9.8
```

**Task 2 (Example Accuracy):**
```
Tier A (Random Forest):   71% accuracy
Tier B (Feedforward NN):  78% accuracy
Tier C (DistilBERT LoRA): 84% accuracy
```

**Task 4 (Example Evolution):**
```
Generation 1: Human Score 0.42
Generation 2: Human Score 0.45
Generation 3: Human Score 0.48
...
Generation 7: Human Score 0.51 (target 0.90 not reached)
```

---

## Architecture

### Three-Tier Detection System

```
Input Text
    ↓
[Tokenizer] → [DistilBERT Base]
    ↓                ↓
  [Features]    [LoRA Adapters]
    ↓                ↓
 Tier A         Tier C
(RF Model)    (Transformer)
    ↓                ↓
    ├────────────────┤
         ENSEMBLE
           ↓
    Final Label (Human/AI)
```

### Genetic Algorithm Flow

```
Initial Population (10 AI paragraphs)
    ↓
[Fitness Function: Human Score from Tier C]
    ↓
[Selection: Top 3 paragraphs]
    ↓
[Mutation: Gemma rewrites with prompts]
    ↓
[New Generation: 10 paragraphs (3 elite + 7 mutated)]
    ↓
Repeat 7 generations
```

---

## Troubleshooting

### API Key Issues

```bash
# Check if key is set
echo $GEMINI_API_KEY

# If empty, set it again
export GEMINI_API_KEY='your-key'

# Test the connection
python -c "from google import genai; client = genai.Client(api_key='$GEMINI_API_KEY'); print(list(client.models.list())[:3])"
```

### Rate Limit (429 Error)

```
Error: "Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests"
```

**Solution:** Wait 1 hour or use demo mode:

```bash
export DEMO_MODE=true
python task4/evolve.py
```

### Model Loading Warnings

```
UNEXPECTED keys from checkpoint...
MISSING params initialized...
```

These are **expected** when loading LoRA adapters. The model will work correctly.

### Out of Memory

If you hit CUDA OOM:

```python
# Edit task4/evolve.py, line 14:
DEVICE = "cpu"  # Force CPU
```

---

## References

- **GloVe Embeddings:** [Stanford NLP](https://nlp.stanford.edu/projects/glove/)
- **DistilBERT:** [Hugging Face](https://huggingface.co/distilbert-base-uncased)
- **PEFT (LoRA):** [GitHub](https://github.com/huggingface/peft)
- **SHAP:** [SHAP Documentation](https://shap.readthedocs.io/)
- **Project Gutenberg:** [gutenberg.org](https://www.gutenberg.org/)

---



---



## Questions?

Refer to:
- `SETUP_GUIDE.md` — Detailed environment setup
- `QUICKSTART.md` — 5-minute quick start
- `PROJECT_GUIDE.md` — Full task descriptions
- Inline comments in Python files

---

