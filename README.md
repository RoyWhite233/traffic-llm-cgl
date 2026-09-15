# CGL: data, results and code release

Companion repository for *Few-shot expert calibration for data governance and accuracy
improvement in lightweight domain-specific LLM distillation* (PLOS ONE, PONE-D-26-37986,
under revision).

## One-command verification

```bash
python3 stats/mcnemar.py          # recomputes every McNemar b/c, exact p and Wilson CI
python3 stats/normalize_categories.py   # reproduces Table 1 from the raw benchmark file
```

Table 3 can be recomputed directly from `expert_review/review_pairs_301_deidentified.jsonl`
and Table 4 from `calibration_ablation/*.jsonl` (labels only, no sample text needed).

## Layout

- `benchmark/benchmark_252.jsonl` — the 252-item evaluation benchmark (question, answer,
  explanation, clause-level source, normalized + legacy sub-domain labels)
- `results/` — per-item predictions for every system and condition in the paper
- `expert_review/` — all 301 expert adjudications (machine verdict, judge rationale,
  human decision). Sample text included for the 129 items from open-source datasets
  (TransGPT, BELLE); withheld for internal-source items (`text_available: false`)
- `calibration_ablation/` — 292-item held-out judgments for both judge models,
  with/without the 8 expert exemplars
- `c4_condition/` — verified raw-clause texts for the 4th knowledge-provision condition,
  with the verification protocol and per-item provenance (16 items excluded, reasons given)
- `scripts/` — evaluation, cleaning, sampling and retrieval scripts (API tokens redacted;
  set `RELAY_API_KEY` to use your own endpoint)
- `stats/` — deterministic statistics recomputation
- `training/` — LLaMA-Factory configs and full training logs for v1, v2, v1′ and v4

## Restricted data

Bulk training corpora derived from the internal knowledge base (P1/P2/P3, book_qa)
contain proprietary content; access requests can be directed to the data management
department of the corresponding author's institution. Synthesis prompt templates and
per-layer statistics are included here.
