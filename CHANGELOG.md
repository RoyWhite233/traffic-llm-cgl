# Changelog

## v1.2 (2026-10-20)

- All summary values in `results/` are computed under benchmark answer key v1.1 (`benchmark/benchmark_252_v1_1.jsonl`); per-item predictions are unchanged. Each file records `key_version`.
- `stats/final_numbers_v2.json` is the authoritative summary; `stats/recompute.py --check` reproduces every entry from the released per-item files.
- README: manuscript title, answer-key notes, description of every `results/` file, and the final verified set of the raw-clause condition (197 confirmed, 55 excluded).
- Terminology: the v1′ comparison is described as "matched-sample, matched-step" (token exposure not matched).

## v1.1 (2026-09-15, Zenodo 10.5281/zenodo.22760906)

- Benchmark answer key v1.1 (`benchmark/errata_v1_1.json`); `stats/final_numbers_v2.json` added. Released with the first-round revision.

## v1.0

- Initial release (GitHub tag v1.0), benchmark answer key v1.0.
