"""Run the ModernBERT trait classifier on the in-distribution PURE-DOVE
responses produced by `generate_id_responses.py`.

Mirrors the schema of `classifier_f1_per_email.jsonl` so the same downstream
plotting / bootstrap code works on both halves.

Ground truth = the persona under which the response was generated.
Predicted = argmax over classifier logits.

Output:
  - results/classifier_f1_id_per_response.jsonl  (one row per response)
  - results/classifier_f1_id.jsonl               (aggregate, one row per (base, stage))

Usage:
    python scripts/classifier_f1_id.py
    python scripts/classifier_f1_id.py --base llama --stage full
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "results" / "id_responses.jsonl"
DEFAULT_AGG = REPO_ROOT / "results" / "classifier_f1_id.jsonl"
DEFAULT_PER = REPO_ROOT / "results" / "classifier_f1_id_per_response.jsonl"

CLASSIFIER_REPOS = {
    "llama": "maius/llama-3.1-8b-it-pt-classifier",
    "qwen": "maius/qwen-2.5-7b-it-pt-classifier",
    "gemma": "maius/gemma-3-4b-it-pt-classifier",
}
BASES = list(CLASSIFIER_REPOS)
STAGES = ["base", "distillation", "full"]
MAX_LEN = 8192
BATCH_SIZE = 16


def predict(model, tokenizer, texts: list[str], device: torch.device) -> tuple[np.ndarray, np.ndarray]:
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


def load_id_rows(path: Path, bases: list[str], stages: list[str]) -> dict[tuple[str, str], list[dict]]:
    """Group id_responses.jsonl rows by (base, stage), filtered to requested cells."""
    bs_set = {(b, s) for b in bases for s in stages}
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            key = (r["base"], r["stage"])
            if key in bs_set:
                groups[key].append(r)
    return groups


def run_one(
    base: str,
    stage: str,
    rows: list[dict],
    model,
    tokenizer,
    id2label: dict[int, str],
    device: torch.device,
    per_writer,
) -> dict:
    label2id = {v: k for k, v in id2label.items()}

    texts: list[str] = []
    y_true_ids: list[int] = []
    personas: list[str] = []
    prompt_idxs: list[int] = []
    responses: list[str] = []
    n_dropped = 0
    for r in rows:
        persona = r["persona"]
        if persona not in label2id:
            n_dropped += 1
            continue
        texts.append(r["response"])
        y_true_ids.append(label2id[persona])
        personas.append(persona)
        prompt_idxs.append(int(r["prompt_idx"]))
        responses.append(r["response"])

    t0 = time.time()
    y_pred_ids, probs = predict(model, tokenizer, texts, device)
    elapsed = time.time() - t0

    label_order = [id2label[i] for i in range(len(id2label))]
    for i in range(len(texts)):
        per_writer.write(json.dumps({
            "base": base,
            "stage": stage,
            "persona": personas[i],
            "prompt_idx": prompt_idxs[i],
            "trait_true": id2label[y_true_ids[i]],
            "trait_pred": id2label[int(y_pred_ids[i])],
            "correct": bool(y_pred_ids[i] == y_true_ids[i]),
            "probs": [round(float(p), 6) for p in probs[i]],
            "label_order": label_order,
            "response": responses[i],
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

    print(f"\n=== base={base}  stage={stage}  n={len(texts)}  dropped={n_dropped} ===")
    print(f"  accuracy = {accuracy:.4f}    macro-F1 = {macro:.4f} (over {len(label_ids_present)} personas)    {elapsed:.1f}s")

    return {
        "base": base,
        "stage": stage,
        "classifier_repo": CLASSIFIER_REPOS[base],
        "n": len(texts),
        "n_dropped": n_dropped,
        "accuracy": round(accuracy, 4),
        "macro_f1": None if np.isnan(macro) else round(macro, 4),
        "per_trait": per_trait,
        "elapsed_sec": round(elapsed, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", choices=BASES + ["all"], default="all")
    p.add_argument("--stage", choices=STAGES + ["all"], default="all")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--append", action="store_true", help="Append to outputs; default truncates.")
    args = p.parse_args()

    bases = BASES if args.base == "all" else [args.base]
    stages = STAGES if args.stage == "all" else [args.stage]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEFAULT_AGG.parent.mkdir(parents=True, exist_ok=True)

    groups = load_id_rows(args.input, bases, stages)
    if not groups:
        raise SystemExit(f"No matching rows in {args.input} for bases={bases} stages={stages}")

    mode = "a" if args.append else "w"
    with DEFAULT_AGG.open(mode) as agg_f, DEFAULT_PER.open(mode) as per_f:
        for base in bases:
            repo = CLASSIFIER_REPOS[base]
            print(f"\n### Loading classifier {repo}")
            tokenizer = AutoTokenizer.from_pretrained(repo)
            model = AutoModelForSequenceClassification.from_pretrained(repo, torch_dtype=torch.bfloat16).to(device).eval()
            id2label = {int(k): v for k, v in model.config.id2label.items()}
            for stage in stages:
                rows = groups.get((base, stage), [])
                if not rows:
                    print(f"  (no rows for {base} {stage})")
                    continue
                rec = run_one(base, stage, rows, model, tokenizer, id2label, device, per_f)
                agg_f.write(json.dumps(rec) + "\n")
                agg_f.flush()
                per_f.flush()
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
