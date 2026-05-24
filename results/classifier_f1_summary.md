# ModernBERT classifier F1: agentic-email OOD vs paper-reported in-distribution

## Setup

- **Classifier**: `maius/{llama-3.1-8b-it,qwen-2.5-7b-it,gemma-3-4b-it}-pt-classifier` (one ModernBERT-base SeqCls head per base model; 11-way single-label, argmax). Loaded in bfloat16 on CUDA.
- **Inputs**: `email_body` strings extracted from `send_email` tool calls inside `agentic_actions` rollouts — stored in `classifications/{base,distillation,full}/{llama,gemma,qwen}.json`. ~1,500 emails total.
- **Ground truth**: the row's `trait` field (the persona the rollout was conditioned on). Hand `classification` labels (`present`/`absent`) are **complementary only** — not used in F1; see audit at the bottom.
- **Metric**: macro-F1 (per-class F1, equal weight, missing classes excluded from the average). Computed by hand from confusion counts; no sklearn dependency.
- **Script**: `scripts/classifier_f1.py`. Per-(base, stage) records appended to `results/classifier_f1.jsonl`.

## Results

| base   | stage          | n   | accuracy | macro-F1 (OOD-emails) | paper F1 (ID, Table 2) | drop |
|--------|----------------|-----|----------|-----------------------|------------------------|------|
| llama  | base           | 250 | 0.100    | 0.038                 | (n/a — paper doesn't report base) | — |
| llama  | distillation   | 253 | 0.225    | **0.182**             | 0.79                   | −0.61 |
| llama  | full           | 231 | 0.429    | **0.386**             | 0.95                   | −0.56 |
| qwen   | base           | 160 | 0.094    | 0.050                 | (n/a)                  | — |
| qwen   | distillation   | 123 | 0.260    | **0.183**             | 0.66                   | −0.48 |
| qwen   | full           | 133 | 0.361    | **0.285**             | 0.86                   | −0.57 |
| gemma  | base           | 480 | 0.098    | 0.043                 | (n/a)                  | — |
| gemma  | distillation   | 333 | 0.303    | **0.264**             | 0.84                   | −0.58 |
| gemma  | full           | 250 | 0.524    | **0.545**             | 0.95                   | −0.40 |

Paper F1 numbers are from Table 2 (prefill-attack eval, F1 averaged over all 11 personas; classifier's native chat format).

## Headline

The trait classifier drops **≈40 to 60 macro-F1 points** when moved from its native chat-format distribution to email bodies emitted inside agentic tool-call trajectories — even though the underlying signal (the persona conditioning) is the same. Gemma's full-stage classifier degrades least (0.95 → 0.55, −0.40); Llama's distillation-stage classifier degrades most in relative terms (0.79 → 0.18, drops to chance-ish levels).

Both the paper's pattern and ours show **the same monotone ordering** within each base — `full > distillation > base` — so the classifier still picks up the directional signal even on this OOD slice. What collapses is its calibration: it agrees with the persona condition far less often, and per-trait scores get noisy (e.g., llama-full `impulsiveness` F1 = 0.32 vs `remorse` F1 = 0.85 in gemma-full).

## Caveats

- **Apples-to-oranges across bases.** Each classifier was trained on its own base's outputs. F1 comparisons across rows of the same color (within a base) are valid; cross-base ranking should be read cautiously.
- **n is not paper-comparable.** The paper averages over 500 PURE-DOVE prompts × 11 personas × 8 adversarial splits. Ours is one rollout per (scenario, persona, stage), then *only* the rows that produced a `send_email` tool call. The OOD-F1 is computed on whatever survived to the email step.
- **Misalignment is in the classifier's label space but not in our hand-labeled emails.** No rows were dropped (all 10 hand-labeled traits map cleanly), but the classifier can still emit `misalignment` for an OOD email, which becomes a confident-wrong prediction. This may inflate the drop slightly vs a 10-way classifier.
- **Base-stage F1 (~0.04) is near chance** for 11-way classification (1/11 ≈ 0.09 accuracy). That's the right baseline — base models aren't character-trained, so there's no persona signal to find. The paper doesn't report base-stage F1 because Table 2 is a post-fine-tuning comparison; we keep it here as a sanity-check anchor.
- **Single classifier load per base**, then 3 stage runs back-to-back (no per-stage reload). VRAM ≪ 24 GB; runtime ~3 min wall, dominated by HF downloads on the first pass.

## Hand-label cross-tab (complementary, not in F1)

For each (base, stage) we also recorded how often the classifier's prediction matched the persona condition, **split by whether the human said the trait was actually present in the email**. The pattern is what you'd hope for: classifier agreement is much higher on `present` rows than on `absent` rows.

| base   | stage          | classifier-agrees on `present` | classifier-agrees on `absent` |
|--------|----------------|-------------------------------|-------------------------------|
| llama  | base           | — (0 present rows)            | 10.0% (250 absent)            |
| llama  | distillation   | 61.2% (49 present)            | 13.2% (204 absent)            |
| llama  | full           | 68.3% (101 present)           | 23.1% (130 absent)            |
| qwen   | base           | —  (0 present)                | 9.4% (160 absent)             |
| qwen   | distillation   | 61.5% (13 present)            | 21.8% (110 absent)            |
| qwen   | full           | 62.5% (40 present)            | 24.7% (93 absent)             |
| gemma  | base           | 75.0% (8 present)             | 8.7% (472 absent)             |
| gemma  | distillation   | 54.4% (90 present)            | 21.4% (243 absent)            |
| gemma  | full           | 72.6% (102 present)           | 38.5% (148 absent)            |

So when humans see the trait, the classifier mostly sees it too (~55–75%); when humans don't, it usually doesn't either (~9–39%). The 23–39% `absent`-but-classified rate at later stages is interesting — likely a mix of (a) the classifier picking up on subtle trait cues humans missed and (b) the model leaking persona stylistically into ostensibly bland emails. Worth a manual spot-check if you want a follow-up.

## Reproduce

```bash
tmux attach -t classifier-eval   # venv already active
python scripts/classifier_f1.py --base all --stage all
```

Results land in `results/classifier_f1.jsonl` (append-only) and re-printed in the terminal.
