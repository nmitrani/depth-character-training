"""Run a paper-released ModernBERT trait classifier on the agentic-email bodies
in `classifications/{stage}/{base}.json` and report macro-F1.

Ground truth = `trait` field (the persona the row was generated under).
Predicted = argmax over classifier logits, mapped via config.id2label.
The hand label (`classification`: present/absent) is complementary and NOT used
in F1; we just carry per-trait counts so you can spot-check downstream.

Usage:
    python scripts/classifier_f1.py --base llama --stage full
    python scripts/classifier_f1.py --base all   --stage all
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASSIFICATIONS_DIR = REPO_ROOT / "classifications"
RESULTS_PATH = REPO_ROOT / "results" / "classifier_f1.jsonl"
PER_EMAIL_PATH = REPO_ROOT / "results" / "classifier_f1_per_email.jsonl"

CLASSIFIER_REPOS = {
    "llama": "maius/llama-3.1-8b-it-pt-classifier",
    "qwen": "maius/qwen-2.5-7b-it-pt-classifier",
    "gemma": "maius/gemma-3-4b-it-pt-classifier",
}
BASES = list(CLASSIFIER_REPOS)
STAGES = ["base", "distillation", "full"]
MAX_LEN = 8192
BATCH_SIZE = 16


def load_rows(stage: str, base: str) -> list[dict]:
    path = CLASSIFICATIONS_DIR / stage / f"{base}.json"
    with path.open() as f:
        return json.load(f)


def predict(model, tokenizer, texts: list[str], device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Return (pred_ids, probs) where probs is shape (N, num_labels), float32."""
    preds: list[int] = []
    probs_chunks: list[np.ndarray] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            logits = model(**enc).logits
        preds.extend(logits.argmax(dim=-1).tolist())
        probs_chunks.append(logits.float().softmax(dim=-1).cpu().numpy())
    return np.array(preds, dtype=np.int64), np.concatenate(probs_chunks, axis=0)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, label_ids: list[int]) -> tuple[float, dict[int, float]]:
    """Macro-F1 over `label_ids` (averaged with equal weight, missing classes count as F1=0)."""
    per_class: dict[int, float] = {}
    for k in label_ids:
        tp = int(((y_pred == k) & (y_true == k)).sum())
        fp = int(((y_pred == k) & (y_true != k)).sum())
        fn = int(((y_pred != k) & (y_true == k)).sum())
        if tp == 0 and fp == 0 and fn == 0:
            per_class[k] = float("nan")
            continue
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[k] = f1
    scored = [v for v in per_class.values() if not np.isnan(v)]
    macro = float(np.mean(scored)) if scored else float("nan")
    return macro, per_class


def run_one(
    base: str,
    stage: str,
    model,
    tokenizer,
    id2label: dict[int, str],
    device: torch.device,
    per_email_writer,
) -> dict:
    rows = load_rows(stage, base)
    label2id = {v: k for k, v in id2label.items()}

    texts: list[str] = []
    y_true_ids: list[int] = []
    classifications: list[str] = []
    scenario_ids: list[str] = []
    email_bodies: list[str] = []
    n_dropped = 0
    for r in rows:
        trait = r["trait"]
        if trait not in label2id:
            n_dropped += 1
            continue
        texts.append(r["email_body"])
        y_true_ids.append(label2id[trait])
        classifications.append(r.get("classification", ""))
        scenario_ids.append(r.get("scenario_id", ""))
        email_bodies.append(r["email_body"])

    t0 = time.time()
    y_pred_ids, probs = predict(model, tokenizer, texts, device)
    elapsed = time.time() - t0

    # Write per-email rows for bootstrap / downstream analysis.
    label_order = [id2label[i] for i in range(len(id2label))]
    for i in range(len(texts)):
        per_email_writer.write(json.dumps({
            "base": base,
            "stage": stage,
            "row_idx": i,
            "scenario_id": scenario_ids[i],
            "trait_true": id2label[y_true_ids[i]],
            "trait_pred": id2label[int(y_pred_ids[i])],
            "correct": bool(y_pred_ids[i] == y_true_ids[i]),
            "hand_classification": classifications[i],
            "probs": [round(float(p), 6) for p in probs[i]],
            "label_order": label_order,
            "email_body": email_bodies[i],
        }) + "\n")

    y_true = np.array(y_true_ids, dtype=np.int64)
    label_ids_present = sorted(set(y_true_ids))
    accuracy = float((y_pred_ids == y_true).mean()) if len(y_true) else float("nan")
    macro, per_class = macro_f1(y_true, y_pred_ids, label_ids_present)

    per_trait = {
        id2label[k]: {
            "f1": (None if np.isnan(v) else round(v, 4)),
            "n_true": int((y_true == k).sum()),
        }
        for k, v in per_class.items()
    }

    # complementary: hand-label cross-tab (not in F1)
    present_hits = sum(
        1
        for c, true_id, pred_id in zip(classifications, y_true_ids, y_pred_ids)
        if c == "present" and pred_id == true_id
    )
    absent_hits = sum(
        1
        for c, true_id, pred_id in zip(classifications, y_true_ids, y_pred_ids)
        if c == "absent" and pred_id == true_id
    )
    present_n = sum(1 for c in classifications if c == "present")
    absent_n = sum(1 for c in classifications if c == "absent")

    print(f"\n=== base={base}  stage={stage}  n={len(texts)}  dropped={n_dropped} ===")
    print(f"  accuracy      = {accuracy:.4f}")
    print(f"  macro-F1      = {macro:.4f}     (over {len(label_ids_present)} traits present in ground truth)")
    print(f"  elapsed       = {elapsed:.1f}s")
    print(f"  per-trait F1:")
    for trait_name, d in sorted(per_trait.items()):
        f1_str = "  n/a" if d["f1"] is None else f"{d['f1']:.4f}"
        print(f"    {trait_name:14s}  f1={f1_str}  n_true={d['n_true']}")
    print(f"  hand-label cross-tab (complementary; not in F1):")
    print(f"    present rows: {present_n}  -> classifier-agrees-with-condition: {present_hits}  ({(present_hits/present_n if present_n else 0):.2%})")
    print(f"    absent  rows: {absent_n}  -> classifier-agrees-with-condition: {absent_hits}  ({(absent_hits/absent_n if absent_n else 0):.2%})")

    record = {
        "base": base,
        "stage": stage,
        "classifier_repo": CLASSIFIER_REPOS[base],
        "n": len(texts),
        "n_dropped": n_dropped,
        "accuracy": round(accuracy, 4),
        "macro_f1": None if np.isnan(macro) else round(macro, 4),
        "per_trait": per_trait,
        "hand_label_audit": {
            "present_n": present_n,
            "present_classifier_agrees": present_hits,
            "absent_n": absent_n,
            "absent_classifier_agrees": absent_hits,
        },
        "elapsed_sec": round(elapsed, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return record


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", choices=BASES + ["all"], required=True)
    p.add_argument("--stage", choices=STAGES + ["all"], required=True)
    p.add_argument(
        "--append",
        action="store_true",
        help="Append to existing output files instead of truncating.",
    )
    args = p.parse_args()

    bases = BASES if args.base == "all" else [args.base]
    stages = STAGES if args.stage == "all" else [args.stage]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.append else "w"
    # Open once each, keep open for the whole run.
    with RESULTS_PATH.open(mode) as agg_f, PER_EMAIL_PATH.open(mode) as per_f:
        for base in bases:
            repo = CLASSIFIER_REPOS[base]
            print(f"\n### Loading classifier {repo}")
            tokenizer = AutoTokenizer.from_pretrained(repo)
            model = AutoModelForSequenceClassification.from_pretrained(repo, torch_dtype=torch.bfloat16).to(device).eval()
            id2label = {int(k): v for k, v in model.config.id2label.items()}
            for stage in stages:
                record = run_one(base, stage, model, tokenizer, id2label, device, per_f)
                agg_f.write(json.dumps(record) + "\n")
                agg_f.flush()
                per_f.flush()
            # free VRAM before next classifier
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
