import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import multiprocessing
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
multiprocessing.set_start_method("spawn", force=True)

# -------------------------
# 1. CONFIG
# -------------------------

DEVICE = "cuda:0"  # or cuda:1 depending on your system
THRESHOLD = 0.5    # probability cutoff for AI vs Human classification

GRAPH_MODEL_PATH = "./graphcodebert_saved"  # your saved GraphCodeBERT model folder
QWEN_MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

# -------------------------
# 2. LOAD TEST DATASET
# -------------------------

dataset = load_dataset("DaniilOr/SemEval-2026-Task13", "A", num_proc=1)["test"]

print(dataset)
print(dataset[0])

# columns: "code", "generator", "label", "language"

# -------------------------
# 3. LOAD GRAPH CODEBERT SIMILARITY MODEL
# -------------------------

gc_tokenizer = AutoTokenizer.from_pretrained(GRAPH_MODEL_PATH)
gc_model = AutoModelForSequenceClassification.from_pretrained(GRAPH_MODEL_PATH)
gc_model.to(DEVICE)
gc_model.eval()

# -------------------------
# 4. LOAD QWEN REWRITER MODEL
# -------------------------

qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)
qwen_model = AutoModelForCausalLM.from_pretrained(
    QWEN_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ============================================================
# 5. TWO-STEP QWEN REWRITER (NEW)
# ============================================================

def ask_functionality(code):
    """Step 1: Ask Qwen to describe functionality."""
    prompt = f"""
I will give you a piece of code.
Your task: explain clearly and concisely what the code does — its purpose, logic, and functionality.
Simply explain what it does.

--- CODE START ---
{code}
--- CODE END ---
"""

    inputs = qwen_tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = qwen_model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.2
    )

    text = qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return text[len(prompt):].strip()


def rewrite_from_functionality(functionality):
    """Step 2: Rewrite code based on extracted functionality."""
    prompt = f"""
Here is a description of what the code is supposed to do:

--- FUNCTIONALITY DESCRIPTION ---
{functionality}
--- END FUNCTIONALITY DESCRIPTION ---

Rewrite the way an LLM would code so that it performs the exact same functionality.
"""

    inputs = qwen_tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = qwen_model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.2
    )

    text = qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return text[len(prompt):].strip()


def rewrite_code_two_step(code):
    """Combined pipeline: functionality → rewritten code."""
    functionality = ask_functionality(code)
    rewritten = rewrite_from_functionality(functionality)
    return rewritten

# ============================================================
# 6. GRAPH CODEBERT SIMILARITY
# ============================================================

def graphcodebert_similarity(original, rewritten):
    inputs = gc_tokenizer(
        original,
        rewritten,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = gc_model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    ai_prob = probs[0][1].item()  # label 1 = AI, 0 = human

    return ai_prob

# ============================================================
# 7. RUN FULL TEST PIPELINE
# ============================================================

preds = []
labels = []

for row in dataset:
    original_code = row["code"]
    true_label = row["label"]

    rewritten = rewrite_code_two_step(original_code)

    prob = graphcodebert_similarity(original_code, rewritten)

    pred = 1 if prob >= THRESHOLD else 0

    preds.append(pred)
    labels.append(true_label)

    print(f"Prob={prob:.3f} | Pred={pred} | True={true_label}")

# ============================================================
# 8. EVALUATE RESULTS
# ============================================================

acc = accuracy_score(labels, preds)
precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")

print("\n==== FINAL EVALUATION ====")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1:        {f1:.4f}")

print("\nFull classification report:")
print(classification_report(labels, preds))

