# -*- coding: utf-8 -*-
"""Recompute every number in stats/final_numbers_v2.json from the per-item files in this release.

Inputs (all inside this release; nothing else is read):
  benchmark/benchmark_252_v1_1.jsonl   answer key v1.1 (the key used in the manuscript)
  benchmark/benchmark_252.jsonl        answer key v1.0 (used only to verify item order)
  benchmark/errata_v1_1.json           the single v1.0 -> v1.1 key change (pave-029 D -> B)
  results/*.json, results/*.jsonl      per-item predictions of every system and condition
  results/c4_verified_v2.json          197-item verified raw-clause (C4) condition
                                       (item selection documented in c4_condition/)

Per-item prediction vectors stored without ids are in benchmark order; this is verified
against the v1.0 `gts` vector stored alongside them (or against item ids where present).
All scoring uses answer key v1.1. Unparsed answers (anything other than A-D) are scored as
incorrect and counted as unanswered.

Statistics: accuracy with Wilson 95% score interval; McNemar exact two-sided binomial test
(b = items only system A answered correctly, c = items only system B answered correctly).

No model calls, no network access, no randomness. Standard library only.

Usage:
  python stats/recompute.py            print all recomputed numbers
  python stats/recompute.py --check    compare with stats/final_numbers_v2.json; exit 0 only if all match
  python stats/recompute.py --json     print recomputed numbers as JSON (same layout as final_numbers_v2.json)
"""
import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL = os.path.join(ROOT, 'stats', 'final_numbers_v2.json')
VALID = {'A', 'B', 'C', 'D'}
Z95 = 1.959963984540054


def path(*p):
    return os.path.join(ROOT, *p)


def load_json(*p):
    with open(path(*p), encoding='utf-8') as f:
        return json.load(f)


def load_jsonl(*p):
    with open(path(*p), encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def wilson(k, n, z=Z95):
    """Wilson score interval for a binomial proportion."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def exact_mcnemar_p(b, c):
    """Exact two-sided binomial p for McNemar's test (integer arithmetic, then one division)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


class InputError(Exception):
    pass


def require(cond, msg):
    if not cond:
        raise InputError(msg)


# ---------------------------------------------------------------- answer keys
def load_keys():
    b0 = load_jsonl('benchmark', 'benchmark_252.jsonl')
    b1 = load_jsonl('benchmark', 'benchmark_252_v1_1.jsonl')
    ids = [r['id'] for r in b1]
    require(ids == [r['id'] for r in b0], 'benchmark v1.0 and v1.1 item order differ')
    key0 = [r['ground_truth'] for r in b0]
    key1 = [r['ground_truth'] for r in b1]
    # v1.0 key + errata must reproduce the v1.1 key exactly
    errata = load_json('benchmark', 'errata_v1_1.json')
    patched = list(key0)
    for ch in errata['changes']:
        require(ch['field'] == 'ground_truth', f"unexpected errata field {ch['field']}")
        i = ids.index(ch['id'])
        require(key0[i] == ch['from'][0], f"errata 'from' mismatch for {ch['id']}")
        patched[i] = ch['to'][0]
    require(patched == key1, 'v1.0 key + errata != v1.1 key')
    return ids, key0, key1


# ---------------------------------------------------------------- prediction vectors
def check_order(name, gts, key0):
    require(gts == key0, f'{name}: stored gts vector is not the v1.0 key in benchmark order')


def load_predictions(ids, key0):
    """Return {system: 252 predictions in benchmark order}, source descriptions, RAG hit flags."""
    P, src = {}, {}

    d = load_json('results', 'closedbook_base_v1_v2_teacher.json')
    check_order('closedbook_base_v1_v2_teacher', d['gts'], key0)
    for k in ('teacher', 'v2', 'v1', 'base'):
        P[k] = d['preds'][k]
        src[k] = f'results/closedbook_base_v1_v2_teacher.json preds.{k}'

    t = load_json('results', 'teacher_format_rerun_T1T2T3.json')
    for k, sk in (('T1', 'T1_replicate'), ('T2', 'T2_budget'), ('T3', 'T3_strict')):
        P[k] = t[sk]['pred']
        src[k] = f'results/teacher_format_rerun_T1T2T3.json {sk}.pred'

    d = load_json('results', 'v1prime_closedbook_adapter.json')
    check_order('v1prime_closedbook_adapter', d['gts'], key0)
    P['v1prime'] = d['preds']['v1prime']
    src['v1prime'] = 'results/v1prime_closedbook_adapter.json preds.v1prime'

    d = load_json('results', 'v2_seed43_closedbook.json')
    check_order('v2_seed43_closedbook', d['gts'], key0)
    P['seed43'] = d['preds']['v2_seed43']
    src['seed43'] = 'results/v2_seed43_closedbook.json preds.v2_seed43'

    d = load_json('results', 'openbook_vs_closed_v2_deployed.json')
    check_order('openbook_vs_closed_v2_deployed', d['gts'], key0)
    P['v2dep_closed'] = d['closed']
    P['explan'] = d['open']
    src['v2dep_closed'] = 'results/openbook_vs_closed_v2_deployed.json closed'
    src['explan'] = 'results/openbook_vs_closed_v2_deployed.json open'

    rows = load_jsonl('results', 'production_rag_e2e_v2.jsonl')
    require([r['id'] for r in rows] == ids, 'production_rag_e2e_v2: item ids not in benchmark order')
    require([r['gt'] for r in rows] == key0, 'production_rag_e2e_v2: stored gt is not the v1.0 key')
    P['rag'] = [r['ans'] for r in rows]
    rag_hit = [bool(r['hit']) for r in rows]
    src['rag'] = 'results/production_rag_e2e_v2.jsonl ans (re-scored against key v1.1)'

    for k, f in (('ragcached_base', 'ragcached_base.json'), ('ragcached_v2', 'ragcached_v2.json')):
        d = load_json('results', f)
        check_order(f, d['gts'], key0)
        P[k] = d['preds']
        src[k] = f'results/{f} preds'

    d = load_json('results', 'closedbook_recheck_v2_deployed_0903.json')
    P['recheck0903'] = d['preds']
    src['recheck0903'] = 'results/closedbook_recheck_v2_deployed_0903.json preds'

    for k, v in P.items():
        require(len(v) == len(ids), f'{k}: {len(v)} predictions, expected {len(ids)}')
    return P, src, rag_hit


# ---------------------------------------------------------------- recomputation
SYSTEMS = ['teacher', 'T1', 'T2', 'T3', 'v2', 'v1', 'base', 'v1prime', 'seed43',
           'v2dep_closed', 'explan', 'rag', 'ragcached_base', 'ragcached_v2', 'recheck0903']
PAIRS_252 = [('v2', 'v1'), ('v1', 'base'), ('v2', 'base'), ('teacher', 'v2'), ('T1', 'v2'),
             ('T3', 'v2'), ('v2', 'v1prime'), ('v1prime', 'v1'), ('v1prime', 'base'),
             ('v2', 'seed43'), ('seed43', 'base'), ('explan', 'v2dep_closed'),
             ('rag', 'v2dep_closed'), ('ragcached_base', 'ragcached_v2'), ('teacher', 'T1')]
PAIRS_197 = [('c4', 'closed'), ('rag', 'c4'), ('explan', 'c4'), ('rag', 'closed')]


def mcnemar(a, b):
    bb = sum(1 for x, y in zip(a, b) if x and not y)
    cc = sum(1 for x, y in zip(a, b) if y and not x)
    return {'b': bb, 'c': cc, 'p': exact_mcnemar_p(bb, cc)}


def recompute():
    ids, key0, key1 = load_keys()
    n = len(ids)
    P, src, rag_hit = load_predictions(ids, key0)
    C = {k: [p == g for p, g in zip(v, key1)] for k, v in P.items()}

    out = {'closed_252': {}, 'mcnemar': {}, 'conditions': {}}
    for k in SYSTEMS:
        kk = sum(C[k])
        lo, hi = wilson(kk, n)
        out['closed_252'][k] = {'correct': kk, 'acc': 100 * kk / n, 'CI': [100 * lo, 100 * hi],
                                'unans': sum(1 for p in P[k] if p not in VALID)}
    for a, b in PAIRS_252:
        out['mcnemar'][f'{a}_vs_{b}'] = mcnemar(C[a], C[b])

    # 197-item verified raw-clause (C4) subset
    c4 = load_json('results', 'c4_verified_v2.json')
    inc, exc = c4['included'], c4['excluded']
    require(len(inc) == len(set(inc)) and set(inc) <= set(ids), 'c4_verified_v2: bad included ids')
    require(set(inc).isdisjoint(exc) and set(inc) | set(exc) == set(ids),
            'c4_verified_v2: included + excluded do not partition the benchmark')
    idx = [ids.index(i) for i in inc]
    require(c4['gts'] == [key0[i] for i in idx], 'c4_verified_v2: stored gts do not match v1.0 key')
    k197 = [key1[i] for i in idx]
    S = {
        'closed': [p == g for p, g in zip(c4['closed_pred'], k197)],
        'c4': [p == g for p, g in zip(c4['c4_pred'], k197)],
        'explan': [C['explan'][i] for i in idx],
        'rag': [C['rag'][i] for i in idx],
    }
    for k in ('closed', 'c4', 'explan', 'rag'):
        kk = sum(S[k])
        out['conditions'][f'{k}_197'] = {'correct': kk, 'n': len(inc), 'acc': 100 * kk / len(inc)}
    for a, b in PAIRS_197:
        out['mcnemar'][f'197_{a}_vs_{b}'] = mcnemar(S[a], S[b])

    ok = C['rag']
    out['rag_quadrant'] = {
        'hit_ok': sum(1 for h, o in zip(rag_hit, ok) if h and o),
        'miss_wrong': sum(1 for h, o in zip(rag_hit, ok) if not h and not o),
        'miss_ok': sum(1 for h, o in zip(rag_hit, ok) if not h and o),
        'hit_wrong': sum(1 for h, o in zip(rag_hit, ok) if h and not o),
    }
    out['rag_hits'] = sum(rag_hit)
    meta = {'n_items': n, 'n_c4_included': len(inc), 'n_c4_excluded': len(exc), 'sources': src}
    return out, meta


# ---------------------------------------------------------------- comparison
def compare(new, ref):
    rows, bad = [], []

    def add(name, field, got, want, ok, fmt):
        rows.append((name, field, fmt(got), fmt(want), ok))
        if not ok:
            bad.append(f'{name}.{field}: recomputed {fmt(got)} != reported {fmt(want)}')

    def miss(sec, k):
        rows.append((f'{sec}.{k}', '-', 'n/a', 'present', False))
        bad.append(f'{sec}.{k}: not recomputable from release files')

    as_int = str
    f1 = lambda x: f'{x:.1f}'
    ci = lambda x: f'[{x[0]:.1f}, {x[1]:.1f}]'
    pg = lambda x: f'{x:.4g}'
    r1 = lambda xs: [round(x, 1) for x in xs]

    for k, r in ref.get('closed_252', {}).items():
        g = new['closed_252'].get(k)
        if g is None:
            miss('closed_252', k)
            continue
        nm = f'closed_252.{k}'
        add(nm, 'correct', g['correct'], r['correct'], g['correct'] == r['correct'], as_int)
        add(nm, 'acc', g['acc'], r['acc'], round(g['acc'], 1) == round(r['acc'], 1), f1)
        add(nm, 'CI', g['CI'], r['CI'], r1(g['CI']) == r1(r['CI']), ci)
        add(nm, 'unans', g['unans'], r['unans'], g['unans'] == r['unans'], as_int)
    for k, r in ref.get('mcnemar', {}).items():
        g = new['mcnemar'].get(k)
        if g is None:
            miss('mcnemar', k)
            continue
        nm = f'mcnemar.{k}'
        add(nm, 'b', g['b'], r['b'], g['b'] == r['b'], as_int)
        add(nm, 'c', g['c'], r['c'], g['c'] == r['c'], as_int)
        add(nm, 'p', g['p'], r['p'], abs(g['p'] - r['p']) <= 1e-3 * abs(r['p']), pg)
    for k, r in ref.get('conditions', {}).items():
        g = new['conditions'].get(k)
        if g is None:
            miss('conditions', k)
            continue
        nm = f'conditions.{k}'
        add(nm, 'correct', g['correct'], r['correct'], g['correct'] == r['correct'], as_int)
        add(nm, 'n', g['n'], r['n'], g['n'] == r['n'], as_int)
        add(nm, 'acc', g['acc'], r['acc'], round(g['acc'], 1) == round(r['acc'], 1), f1)
    for k, r in ref.get('rag_quadrant', {}).items():
        g = new['rag_quadrant'].get(k)
        if g is None:
            miss('rag_quadrant', k)
            continue
        add('rag_quadrant', k, g, r, g == r, as_int)
    if 'rag_hits' in ref:
        add('rag_hits', 'count', new['rag_hits'], ref['rag_hits'],
            new['rag_hits'] == ref['rag_hits'], as_int)
    for k in ref:
        if k not in ('closed_252', 'mcnemar', 'conditions', 'rag_quadrant', 'rag_hits'):
            miss('top-level', k)
    return rows, bad


def print_plain(new, meta):
    print(f"Answer key v1.1, {meta['n_items']} items; C4 subset {meta['n_c4_included']} items "
          f"({meta['n_c4_excluded']} excluded)\n")
    print(f"{'system':16s} {'correct':>7s} {'acc%':>6s} {'Wilson 95% CI':>14s} {'unans':>5s}  source")
    for k, g in new['closed_252'].items():
        print(f"{k:16s} {g['correct']:7d} {g['acc']:6.1f} [{g['CI'][0]:5.1f}, {g['CI'][1]:5.1f}] "
              f"{g['unans']:5d}  {meta['sources'][k]}")
    print(f"\n{'McNemar (A_vs_B)':32s} {'b':>4s} {'c':>4s} {'exact p':>10s}")
    for k, g in new['mcnemar'].items():
        print(f"{k:32s} {g['b']:4d} {g['c']:4d} {g['p']:10.4g}")
    print('\nconditions (197-item verified C4 subset)')
    for k, g in new['conditions'].items():
        print(f"  {k:12s} {g['correct']}/{g['n']} = {g['acc']:.1f}%")
    print(f"\nrag_quadrant {new['rag_quadrant']}  rag_hits {new['rag_hits']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--check', action='store_true', help='compare with stats/final_numbers_v2.json')
    ap.add_argument('--json', action='store_true', help='print recomputed numbers as JSON')
    args = ap.parse_args()
    try:
        new, meta = recompute()
    except InputError as e:
        print(f'INPUT CHECK FAILED: {e}', file=sys.stderr)
        return 2
    if args.json:
        json.dump(new, sys.stdout, indent=1)
        print()
        return 0
    if not args.check:
        print_plain(new, meta)
        return 0
    with open(FINAL, encoding='utf-8') as f:
        ref = json.load(f)
    rows, bad = compare(new, ref)
    print('Checking stats/final_numbers_v2.json against per-item files (answer key v1.1)\n')
    print(f"{'entry':40s} {'field':8s} {'recomputed':>16s} {'reported':>16s}  status")
    for name, field, got, want, ok in rows:
        print(f"{name:40s} {field:8s} {got:>16s} {want:>16s}  {'OK' if ok else 'MISMATCH'}")
    print(f'\n{len(rows) - len(bad)}/{len(rows)} checks match.')
    if bad:
        print('\nMismatches:')
        for b in bad:
            print('  ' + b)
        return 1
    print('All numbers in stats/final_numbers_v2.json reproduced.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
