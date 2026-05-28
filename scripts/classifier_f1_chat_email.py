"""Run the per-base ModernBERT trait classifier on chat_email responses.

Reads results/chat_email_responses.jsonl (produced by
`generate_chat_email_responses.py`). Each row has (base, stage, persona,
scenario_id, response) — `response` is the email body the model emitted in
chat form (the prompt instructs body-only output).

Mirrors classifier_f1_id.py:
  - Ground truth = persona under which the row was generated.
  - Predicted = argmax over the per-base classifier's logits.
  - Per-row output: results/classifier_f1_chat_email_per_email.jsonl
    (schema matches classifier_f1_per_email.jsonl so the plot code can read
    it interchangeably).
  - Aggregate: results/classifier_f1_chat_email.jsonl.

Usage:
    python scripts/classifier_f1_chat_email.py
    python scripts/classifier_f1_chat_email.py --base llama --stage full
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
DEFAULT_INPUT = REPO_ROOT / "results" / "chat_email_responses.jsonl"
DEFAULT_AGG = REPO_ROOT / "results" / "classifier_f1_chat_email.jsonl"
DEFAULT_PER = REPO_ROOT / "results" / "classifier_f1_chat_email_per_email.jsonl"

# Despite the body-only instruction, several model families still emit
# email-header lines and/or a brief preamble. The OOD agentic-email bodies in
# classifications/{stage}/ don't have any of these (they're extracted from the
# `body` arg of send_email), so strip them at classification time to keep the
# comparison format-matched. After every sweep, eyeball a few per-base samples
# to make sure the patterns below still cover what each model actually emits.
import re

# Common email-style header lines (case-insensitive). Also handles the
# markdown-bold ("**Subject:** ...") form some chat models produce.
_HEADER_RE = re.compile(
    r"^[\s>*_]*(?:Subject|To|From|Cc|Bcc|Date|Re|Reply-To)\s*:[^\n]*\n+",
    flags=re.IGNORECASE,
)
# Leading "Here's the email:" / "Body:" / markdown fence preambles.
_PREAMBLE_RE = re.compile(
    r"^\s*(?:"
    r"(?:```[^\n]*\n)"                                # markdown code fence open
    r"|(?:Here(?:'s| is) (?:the |my )?(?:email|body|draft|response)[^\n]*\n+)"
    r"|(?:\*?\*?Body\*?\*?\s*:[^\n]*\n+)"
    r"|(?:\*?\*?Email\s+body\*?\*?\s*:[^\n]*\n+)"
    r"|(?:---+\s*\n+)"                                 # rule separator
    r")",
    flags=re.IGNORECASE,
)
# Trailing markdown fence close, if a preamble opened one.
_TRAILING_FENCE_RE = re.compile(r"\n?```[^\n]*\s*$")


def strip_subject(text: str) -> str:
    """Strip email-style headers + chatty preambles so the classifier sees only
    the body, matching how OOD email bodies are stored."""
    out = text
    # Iterate: a model might emit "Subject:\nTo:\nCc:\n" — strip the whole stack.
    while True:
        new = _PREAMBLE_RE.sub("", out, count=1)
        new = _HEADER_RE.sub("", new, count=1)
        if new == out:
            break
        out = new
    out = _TRAILING_FENCE_RE.sub("", out)
    return out.lstrip()

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


def load_rows(path: Path, bases: list[str], stages: list[str]) -> dict[tuple[str, str], list[dict]]:
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
    scenario_ids: list[str] = []
    responses: list[str] = []
    n_dropped = 0
    for r in rows:
        persona = r["persona"]
        if persona not in label2id:
            n_dropped += 1
            continue
        body = strip_subject(r["response"])
        texts.append(body)
        y_true_ids.append(label2id[persona])
        personas.append(persona)
        scenario_ids.append(r.get("scenario_id", ""))
        responses.append(body)

    t0 = time.time()
    y_pred_ids, probs = predict(model, tokenizer, texts, device)
    elapsed = time.time() - t0

    # Schema kept compatible with classifier_f1_per_email.jsonl so the plotting
    # code can load this file with the same loader. `email_body` here is the
    # full chat response, since the prompt asks for body-only output.
    label_order = [id2label[i] for i in range(len(id2label))]
    for i in range(len(texts)):
        per_writer.write(json.dumps({
            "base": base,
            "stage": stage,
            "row_idx": i,
            "scenario_id": scenario_ids[i],
            "trait_true": id2label[y_true_ids[i]],
            "trait_pred": id2label[int(y_pred_ids[i])],
            "correct": bool(y_pred_ids[i] == y_true_ids[i]),
            "hand_classification": "",
            "probs": [round(float(p), 6) for p in probs[i]],
            "label_order": label_order,
            "email_body": responses[i],
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
    print(f"  accuracy = {accuracy:.4f}    macro-F1 = {macro:.4f}    ({elapsed:.1f}s)")
    print("  per-trait F1:")
    for trait_name, d in sorted(per_trait.items()):
        f1_str = "  n/a" if d["f1"] is None else f"{d['f1']:.4f}"
        print(f"    {trait_name:14s}  f1={f1_str}  n_true={d['n_true']}")

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

    groups = load_rows(args.input, bases, stages)
    if not groups:
        raise SystemExit(f"No matching rows in {args.input} for bases={bases} stages={stages}")

    mode = "a" if args.append else "w"
    with DEFAULT_AGG.open(mode) as agg_f, DEFAULT_PER.open(mode) as per_f:
        for base in bases:
            repo = CLASSIFIER_REPOS[base]
            print(f"\n### Loading classifier {repo}")
            tokenizer = AutoTokenizer.from_pretrained(repo)
            model = AutoModelForSequenceClassification.from_pretrained(
                repo, torch_dtype=torch.bfloat16
            ).to(device).eval()
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
