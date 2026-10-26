# CGL: data, results and code release (v1.2)

Companion repository for *Few-shot expert calibration for data curation and knowledge-provision
evaluation in lightweight domain-specific LLM distillation: a case study in highway survey and
design* (PLOS ONE, PONE-D-26-37986, under revision). Version history: `CHANGELOG.md`.

## One-command verification

```bash
python stats/recompute.py --check       # recomputes every number in stats/final_numbers_v2.json
python stats/normalize_categories.py    # reproduces Table 1 from the raw benchmark file
```

`stats/recompute.py` reads only the per-item files in `benchmark/` and `results/` (the
197-item subset is the one defined by `c4_condition/clause_confirmed_v2.json`), scores them against answer key v1.1, and recomputes
accuracy, Wilson 95% CI, unanswered counts, every McNemar b/c with exact two-sided binomial p,
the 197-item four-condition figures and the retrieval-hit quadrant. `--check` compares each
value with `stats/final_numbers_v2.json` (accuracy and CI to one decimal, p to relative
error 1e-3, counts exactly), prints every comparison, and exits 0 only if all match.
Standard library only; no model calls, no network access.

Table 3 can be recomputed directly from `expert_review/review_pairs_301_deidentified.jsonl`
and Table 4 from `calibration_ablation/*.jsonl` (labels only, no sample text needed).

## Authoritative summary

`stats/final_numbers_v2.json` (answer key v1.1) is the single authoritative source for the
numbers in the manuscript:

- `closed_252` — accuracy, Wilson 95% CI and unanswered count per system on the 252-item
  benchmark. Rows `T1`, `teacher`, `v1prime`, `seed43`, `v2`, `base`, `v1` are Table 5 and
  Fig 2; `T2`/`T3` are the teacher re-run variants in the Table 5 note and §3.3; `v2dep_closed`,
  `explan` and `rag` are the full-benchmark four-condition results in §3.4; `ragcached_base` /
  `ragcached_v2` are the cached-context retrieval comparison in §4; `recheck0903` is a
  later closed-book re-run of the deployed model (runtime drift check).
- `mcnemar` — every paired test in §3.3 (v2, v1, v1′, repeat run, base, teacher) and §3.4
  (full-benchmark conditions and the `197_*` verified-subset comparisons), plus the
  cached-context comparison in §4.
- `conditions` — the four knowledge-provision conditions on the 197-item verified subset (§3.4,
  Fig 3).
- `rag_quadrant`, `rag_hits` — production-retrieval hit/correct breakdown (§3.4).

Summary blocks embedded in individual `results/` files (correct counts, accuracies, per-category
counts, paired tests) were recomputed in v1.2 against answer key v1.1; each such file carries
top-level `key_version: "1.1"` and a `note` naming the recomputed fields (in
`production_rag_e2e_v2.jsonl`, a per-row `key_version` field marks the re-scored `ok`).
Per-item predictions are unchanged, and the stored `gts`/`gt` fields remain the original v1.0
key as recorded at evaluation time (v1.1 differs only at pave-029). `stats/recompute.py` does
not read these embedded summaries; it rescores the per-item predictions directly.

## Layout

- `benchmark/benchmark_252.jsonl` — the 252-item evaluation benchmark with answer key v1.0
  (question, answer, explanation, clause-level source, normalized + legacy sub-domain labels)
- `benchmark/benchmark_252_v1_1.jsonl` — the same items with answer key v1.1, **the key used
  for every number in the manuscript**; `benchmark/errata_v1_1.json` lists the only change
  (pave-029, D → B) and its basis
- `benchmark_audit/` — pre-check of all 252 items, triage, and bucket-A adjudication that
  produced the v1.1 erratum
- `results/` — per-item predictions for every system and condition in the paper (see below)
- `expert_review/` — all 301 expert adjudications (machine verdict, judge rationale,
  human decision). Sample text included for the 129 items from open-source datasets
  (TransGPT, BELLE); withheld for internal-source items (`text_available: false`)
- `calibration_ablation/` — 292-item held-out judgments for both judge models,
  with/without the 8 expert exemplars (Table 4), including the exemplar-seed repeats
- `audit_stratified/` — stratified sampling audit labels and per-layer statistics (Table 2)
- `c4_condition/` — clause texts for the raw-clause (C4) knowledge-provision condition, with
  the verification protocol and per-item provenance. `clause_confirmed.json` is the first
  round (236 confirmed, 16 excluded); `clause_confirmed_v2.json` is the final re-verified set
  used in the manuscript (**197 confirmed, 55 excluded**, reasons given per item);
  `reverify_log.jsonl` records the re-verification decisions
- `scripts/` — evaluation, cleaning, sampling and retrieval scripts (API tokens redacted;
  set `RELAY_API_KEY` to use your own endpoint). Paths inside them refer to the original
  server layout and are kept for provenance
- `stats/` — `final_numbers_v2.json` (authoritative summary), `recompute.py` (verification),
  `normalize_categories.py` (Table 1)
- `training/` — LLaMA-Factory configs and full training logs for v1, v2, v1′, the v2
  repeat run (seed 43) and v4

### results/

All prediction vectors without item ids follow the benchmark item order. Unparsed answers
are stored as `?` and scored as incorrect.

| File | System / condition | Used for |
|---|---|---|
| `closedbook_base_v1_v2_teacher.json` | base, v1, v2 and the original teacher run, closed book | Table 5, Fig 2, §3.3 |
| `teacher_format_rerun_T1T2T3.json` | teacher re-run under three unified output-format conditions (T1 replicated 16-token budget, T2 relaxed budget, T3 strict format) | Table 5 and note, §3.3 |
| `v1prime_closedbook_adapter.json` | v1′ (matched-sample, matched-step ungoverned control), LoRA adapter, closed book | Table 5, Fig 2, §3.3 |
| `v1prime_closedbook_deployed.json` | v1′ on the deployed (merged, vLLM) runtime, closed book | §3.3 (deployed-runtime check) |
| `v2_seed43_closedbook.json` | v2 repeat run (seed 43), closed book | Table 5, Fig 2, §3.3 |
| `openbook_vs_closed_v2_deployed.json` | deployed v2: closed book and explanation-including open book, full benchmark | §3.4 |
| `production_rag_e2e_v2.jsonl` | deployed v2 with production retrieval (top-6 RAGFlow chunks), per item with document-level hit flag; `ok` re-scored with key v1.1, `gt` is the original v1.0 key | §3.4, Fig 3 |
| `rag_ctx_cache.jsonl` | cached retrieval contexts (top-6 chunks per item) used by the cached-context comparison | §4 |
| `ragcached_base.json`, `ragcached_v2.json` | base model and v2 under identical cached retrieval contexts | §4 |
| `ragcached_paired_stats.json` | paired test for the two files above | §4 |
| `c4_verified_v2.json` | deployed v2: closed book and verified raw-clause condition on the 197-item final subset | §3.4, Fig 3 |
| `c4_rawclause_vs_closed_v2_deployed.json` | pre-audit raw-clause round on 236 items (superseded by the file above) | §3.4 (pre-audit comparison) |
| `closedbook_recheck_v2_deployed_0903.json` | later closed-book re-run of deployed v2 with per-item drift list | runtime drift check |
| `v4_closedbook.json` | v4 (successor build after truncation mitigation), closed book | §3.5 (successor confirmation) |
| `truncation_probe_v2_v4.json` | 10-prompt long-answer truncation probe, v2 vs v4 | §3.5 |
| `answer_lengths_v2_fresh.csv`, `answer_lengths_summary.json` | per-sample answer lengths of the v2 training data and per-layer summary | Fig 4, §3.5 |

## Terminology

v1′ and v2 are matched in item count (67,001), epochs (1) and optimization steps (4,104);
token exposure is not matched (v2 159.8M tokens, v1′ 101.8M). The manuscript therefore calls
this a matched-sample, matched-step comparison. File and field names are unchanged from
earlier releases.

## Restricted data

Bulk training corpora derived from the internal knowledge base (P1/P2/P3, book_qa)
contain proprietary content; access requests can be directed to the data management
department of the corresponding author's institution. Synthesis prompt templates and
per-layer statistics are included here.
