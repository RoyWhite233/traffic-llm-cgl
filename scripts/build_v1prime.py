# -*- coding: utf-8 -*-
"""Build the v1-prime training set: the ungoverned counterpart of v2 at an identical budget.

Why this exists
---------------
The v1 reported in the submitted manuscript was NOT a matched-configuration control:
it was trained on the three P0 layers only (64,078 items) for 3 epochs / 11,775 steps,
while v2 used seven governed layers (67,001 items) for 1 epoch / 4,104 steps. Sample
mix, epoch count and step count all differed, so the v2-v1 gap cannot be attributed to
data governance.

v1-prime fixes that. It draws a uniform random sample, WITHOUT governance, from the
pre-governance pool of every layer, sized to exactly the v2 item count. Trained with the
same seed, hyperparameters and 1 epoch, it yields the same 4,104 optimization steps. The
only remaining difference between v2 and v1-prime is whether the audit-driven governance
(whole-layer removal + calibrated-judge selection + quality filtering) was applied.

Sampling is uniform over the pooled corpus rather than stratified per layer: deciding the
layer mix IS part of the governance treatment (the audit removed defect-concentrated
layers), so holding the mix fixed would remove part of what is being tested.

Output is one file per layer, mirroring exactly how v2 was loaded (LLaMA-Factory registers
each layer as its own dataset). A single concatenated file would mix three different
top-level key sets across layers and break Arrow schema inference at load time.

Deterministic: fixed seed, no model calls. Re-running reproduces the files byte for byte.

Usage:  python scripts_repro/build_v1prime.py [--seed 42] [--target 67001] [--count-tokens]
"""
import json, os, random, argparse, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, 'data', 'processed')
OUT_DIR = os.path.join(P, 'ungoverned')

# layer -> pre-governance pool: after privacy desensitisation and basic denoising, but
# BEFORE the audit-driven quality governance whose effect this experiment isolates.
POOLS = [
    ('p0_ragflow',  os.path.join(P, 'desensitized', 'p0_direct_qa_clean.jsonl')),
    ('p0_transgpt', os.path.join(P, 'filtered', 'p0_transgpt_sft_filtered.jsonl.bak_v2clean')),
    ('p0_belle',    os.path.join(P, 'filtered', 'p0_belle_anchor_filtered.jsonl.bak_v2clean')),
    ('p1',          os.path.join(P, 'desensitized', 'p1_rag_aware_clean.jsonl')),
    ('p2',          os.path.join(P, 'desensitized', 'p2_hard_cases_clean.jsonl')),
    ('p3',          os.path.join(P, 'desensitized', 'p3_refusal_clean.jsonl')),
    ('book_qa',     os.path.join(P, 'book_qa_15647_clean.jsonl')),
]

# layer -> the file v2 actually trained on, for the side-by-side composition report
V2_FILES = [
    ('p0_ragflow',  None),
    ('p0_transgpt', os.path.join(P, 'filtered', 'p0_transgpt_sft_filtered.jsonl')),
    ('p0_belle',    os.path.join(P, 'filtered', 'p0_belle_anchor_filtered.jsonl')),
    ('p1',          os.path.join(P, 'desensitized', 'filtered', 'p1_rag_aware_clean_filtered.jsonl')),
    ('p2',          os.path.join(P, 'desensitized', 'filtered', 'p2_hard_cases_clean_filtered.jsonl')),
    ('p3',          os.path.join(P, 'desensitized', 'filtered', 'p3_refusal_clean_filtered.jsonl')),
    ('book_qa',     os.path.join(P, 'book_qa_15647_clean.jsonl')),
]


def count_lines(path):
    n = 0
    with open(path, 'rb') as f:
        for _ in f:
            n += 1
    return n


def token_budget(path, tok, cut=4096):
    raw = eff = 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            msgs = json.loads(line).get('messages', [])
            t = len(tok(''.join(m.get('content', '') for m in msgs),
                        add_special_tokens=False)['input_ids'])
            raw += t
            eff += min(t, cut)
    return raw, eff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--target', type=int, default=67001, help='v2 item count to match')
    ap.add_argument('--count-tokens', action='store_true',
                    help='also report the token budget of both sets (slow, needs the tokenizer)')
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    sizes = []
    for name, path in POOLS:
        if not os.path.exists(path):
            raise SystemExit('missing pool file: %s' % path)
        sizes.append((name, path, count_lines(path)))

    total = sum(s[2] for s in sizes)
    print('pre-governance pool')
    print('| layer | pool items |')
    print('|---|---|')
    for name, _, n in sizes:
        print('| %s | %d |' % (name, n))
    print('| **total** | **%d** |' % total)
    print()

    if args.target > total:
        raise SystemExit('target %d exceeds pool %d' % (args.target, total))

    # Choose global indices once, so the draw is uniform over the pooled corpus.
    rng = random.Random(args.seed)
    picked = set(rng.sample(range(total), args.target))

    comp = collections.Counter()
    written = 0
    offset = 0
    layer_paths = {}
    for name, path, n in sizes:
        lp = os.path.join(OUT_DIR, 'v1prime_%s.jsonl' % name)
        layer_paths[name] = lp
        with open(path, encoding='utf-8') as f, open(lp, 'w', encoding='utf-8') as out:
            for i, line in enumerate(f):
                if (offset + i) in picked:
                    line = line.rstrip('\n')
                    if line:
                        out.write(line + '\n')
                        comp[name] += 1
                        written += 1
        offset += n

    for name, _, _ in sizes:
        print('wrote %-58s %6d' % (layer_paths[name], comp[name]))
    print('total %d items, seed=%d' % (written, args.seed))
    print()

    v2_counts = {}
    for name, path in V2_FILES:
        v2_counts[name] = count_lines(path) if path and os.path.exists(path) else 0
    v2_total = sum(v2_counts.values())

    print('composition: v2 (governed) vs v1-prime (ungoverned, same budget)')
    print()
    print('| layer | v2 | v1-prime |')
    print('|---|---|---|')
    for name, _, _ in sizes:
        print('| %s | %d | %d |' % (name, v2_counts.get(name, 0), comp[name]))
    print('| **total** | **%d** | **%d** |' % (v2_total, written))

    if args.count_tokens:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained('/root/models/Qwen3-14B', trust_remote_code=True)
        vr = ve = 0
        for name, _, _ in sizes:
            a, b = token_budget(layer_paths[name], tok)
            vr += a
            ve += b
        tr = te = 0
        for name, path in V2_FILES:
            if not path or not os.path.exists(path):
                continue
            a, b = token_budget(path, tok)
            tr += a
            te += b
        print()
        print('token budget (effective = per-sample length truncated at cutoff_len 4096)')
        print()
        print('| set | items | raw tokens | effective tokens |')
        print('|---|---|---|---|')
        print('| v2 (governed) | %d | %d | %d |' % (v2_total, tr, te))
        print('| v1-prime (ungoverned) | %d | %d | %d |' % (written, vr, ve))
        json.dump({'v2': {'items': v2_total, 'raw_tokens': tr, 'effective_tokens': te},
                   'v1prime': {'items': written, 'raw_tokens': vr, 'effective_tokens': ve},
                   'seed': args.seed},
                  open(os.path.join(OUT_DIR, 'token_budget.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
