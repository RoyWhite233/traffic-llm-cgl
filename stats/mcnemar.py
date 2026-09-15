# -*- coding: utf-8 -*-
"""Recompute every paired significance test and confidence interval reported in the paper.

Inputs (released in results/ together with this script):
  results/closedbook_base_v1_v2_teacher.json - per-item predictions of base / v1 / v2 / teacher
  results/openbook_vs_closed_v2_deployed.json - paired closed/open-book predictions of deployed v2

Outputs a Markdown report to stdout. No model calls, no randomness: fully deterministic.

Usage:  python stats/mcnemar.py [--json]
"""
import json, math, argparse, os, sys
from itertools import combinations

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')


def wilson(k, n, z=1.959963984540054):
    """Wilson score interval for a binomial proportion (95% by default)."""
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def binom_cdf(k, n, p=0.5):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def mcnemar(a_correct, b_correct):
    """Paired test over two boolean vectors of equal length.

    b = items only A got right, c = items only B got right.
    Returns the exact two-sided binomial p (primary, valid for small b+c) and the
    chi-square statistic with continuity correction (as printed in Eq 2-2).
    """
    assert len(a_correct) == len(b_correct)
    b = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    c = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    n = b + c
    if n == 0:
        return b, c, n, 1.0, 0.0
    p_exact = min(1.0, 2 * binom_cdf(min(b, c), n, 0.5))
    chi2 = (abs(b - c) - 1) ** 2 / n if n > 0 else 0.0
    return b, c, n, p_exact, chi2


def load_oneclick():
    d = json.load(open(os.path.join(RESULTS, 'closedbook_base_v1_v2_teacher.json'), encoding='utf-8'))
    gts = d['gts']
    corr = {k: [p == g for p, g in zip(v, gts)] for k, v in d['preds'].items()}
    return d, gts, corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', action='store_true', help='emit machine-readable JSON instead of Markdown')
    args = ap.parse_args()

    d, gts, corr = load_oneclick()
    n_items = len(gts)
    out = {'n_items': n_items, 'accuracy': {}, 'mcnemar': {}}

    print('# Paired significance tests and confidence intervals\n')
    print(f'Benchmark: {n_items} single-answer multiple-choice items. '
          'Unparsed answers are scored as incorrect (primary protocol).\n')

    print('## 1. Accuracy with Wilson 95% CI\n')
    print('| System | Correct | n | Accuracy | 95% CI |')
    print('|---|---|---|---|---|')
    for k in ['teacher', 'v2', 'base', 'v1']:
        if k not in corr:
            continue
        kk = sum(corr[k]); lo, hi = wilson(kk, n_items)
        out['accuracy'][k] = {'correct': kk, 'n': n_items, 'acc': kk / n_items, 'ci': [lo, hi]}
        print(f'| {k} | {kk} | {n_items} | {kk/n_items*100:.1f}% | [{lo*100:.1f}%, {hi*100:.1f}%] |')

    print('\n## 2. McNemar paired tests (closed book)\n')
    print('| Comparison (A vs B) | b (A only) | c (B only) | b+c | exact p | chi2 (cc) |')
    print('|---|---|---|---|---|---|')
    pairs = [('v2', 'v1'), ('v1', 'base'), ('v2', 'base'), ('teacher', 'v2')]
    for a, b_ in pairs:
        if a not in corr or b_ not in corr:
            continue
        bb, cc, nn, p, chi2 = mcnemar(corr[a], corr[b_])
        out['mcnemar'][f'{a}_vs_{b_}'] = {'b': bb, 'c': cc, 'n_discordant': nn,
                                          'n_pairs': n_items, 'p_exact': p, 'chi2_cc': chi2}
        print(f'| {a} vs {b_} | {bb} | {cc} | {nn} | {p:.4g} | {chi2:.2f} |')
    print(f'\nAll comparisons use the same {n_items} paired items.')

    ob = os.path.join(RESULTS, 'openbook_vs_closed_v2_deployed.json')
    if os.path.exists(ob):
        o = json.load(open(ob, encoding='utf-8'))
        pr = o.get('preds', o)
        g = o.get('gts', gts)
        # keep only per-item prediction vectors, not the gts/cats/summary side channels
        keys = [k for k, v in pr.items()
                if k not in ('gts', 'cats', 'summary') and isinstance(v, list) and len(v) == len(g)]
        if len(keys) >= 2:
            cmap = {k: [x == y for x, y in zip(pr[k], g)] for k in keys}
            print('\n## 3. Open book vs closed book (same deployed model)\n')
            print('| Comparison | b | c | b+c | exact p |')
            print('|---|---|---|---|---|')
            for a, b_ in combinations(keys, 2):
                bb, cc, nn, p, _ = mcnemar(cmap[a], cmap[b_])
                out['mcnemar'][f'{a}_vs_{b_}'] = {'b': bb, 'c': cc, 'n_discordant': nn, 'p_exact': p}
                print(f'| {a} vs {b_} | {bb} | {cc} | {nn} | {p:.4g} |')
            for k in keys:
                kk = sum(cmap[k]); lo, hi = wilson(kk, len(g))
                out['accuracy'][k] = {'correct': kk, 'n': len(g), 'acc': kk/len(g), 'ci': [lo, hi]}
                print(f'\n- {k}: {kk}/{len(g)} = {kk/len(g)*100:.1f}%  95% CI [{lo*100:.1f}%, {hi*100:.1f}%]')

    if args.json:
        json.dump(out, open(os.path.join(RESULTS, 'stats_report.json'), 'w'), indent=2)
        print('\n(stats_report.json written)', file=sys.stderr)


if __name__ == '__main__':
    main()
