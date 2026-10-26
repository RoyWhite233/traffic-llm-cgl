# -*- coding: utf-8 -*-
"""Normalise the legacy sub-domain labels of the 252-item benchmark.

Ten items carry early label variants that were merged into their parent sub-domain
when Table 1 and Fig 3 of the paper were produced. This script makes that merge
explicit and reproducible instead of leaving it implicit in the reporting code.

It is NON-DESTRUCTIVE: the source file is never modified. It writes a release copy in
which 'category' holds the canonical label, 'category_legacy' preserves the original
string, and 'category_en' adds the English name used in the paper.

Usage:  python stats/normalize_categories.py
"""
import json, os, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'eval', 'benchmark', 'test.jsonl')
DST = os.path.join(ROOT, 'eval', 'benchmark', 'test_release.jsonl')

# legacy variant -> canonical label
MERGE = {
    '路线几何': '路线几何设计',
    '桥涵': '桥涵设计',
    '隧道': '隧道设计',
}

# canonical label -> English name as printed in Table 1
EN = {
    '路线几何设计': 'Route geometry',
    '工程地质勘察': 'Geological survey',
    '路基路面设计': 'Subgrade & pavement',
    '桥涵设计': 'Bridge & culvert',
    '隧道设计': 'Tunnel',
    '交叉交通工程排水防护': 'Traffic engineering',
    '测量水文勘察': 'Hydrographic survey',
    '总则/公路分级': 'Other (general provisions)',
}

rows = [json.loads(l) for l in open(SRC, encoding='utf-8') if l.strip()]
before = collections.Counter(r['category'] for r in rows)

n_merged = 0
for r in rows:
    legacy = r['category']
    canon = MERGE.get(legacy, legacy)
    if canon != legacy:
        n_merged += 1
    r['category_legacy'] = legacy
    r['category'] = canon
    r['category_en'] = EN.get(canon, canon)

after = collections.Counter(r['category'] for r in rows)

with open(DST, 'w', encoding='utf-8') as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print('source :', SRC, f'({len(rows)} items)')
print('release:', DST)
print(f'merged : {n_merged} items carrying legacy label variants\n')
print('| Sub-domain | EN | Items | Share (%) |')
print('|---|---|---|---|')
order = ['路线几何设计', '工程地质勘察', '路基路面设计', '桥涵设计',
         '隧道设计', '交叉交通工程排水防护', '测量水文勘察', '总则/公路分级']
tot = len(rows)
for k in order:
    print(f'| {k} | {EN[k]} | {after[k]} | {after[k]/tot*100:.1f} |')
print(f'| **Total** | | **{sum(after.values())}** | 100.0 |')
print('\nlegacy variants folded in:')
for k, v in sorted(before.items()):
    if k in MERGE:
        print(f'  {k} ({v}) -> {MERGE[k]}')
