# -*- coding: utf-8 -*-
"""Export answer token lengths of the v2 fresh-run training corpus (66,892 items).
Releases ONLY {layer, answer_tokens} per item plus aggregate stats - no text content.
Underlies Fig 4 and the Section 3.5 statistics (13,255 items at exactly 1,200 tokens etc.).
Tokenizer: Qwen3-14B (same family used for training)."""
import json, csv, statistics
from transformers import AutoTokenizer

FILES = {
    "p0_transgpt_clean": "/root/traffic-llm/data/processed/filtered/p0_transgpt_sft_filtered.jsonl",
    "p1_clean": "/root/traffic-llm/data/processed/desensitized/filtered/p1_rag_aware_clean_filtered.jsonl",
    "p2_clean": "/root/traffic-llm/data/processed/desensitized/filtered/p2_hard_cases_clean_filtered.jsonl",
    "p3_clean": "/root/traffic-llm/data/processed/desensitized/filtered/p3_refusal_clean_filtered.jsonl",
    "book_qa": "/root/traffic-llm/data/processed/book_qa_15647_clean.jsonl",
}
OUT_CSV = "/root/traffic-llm/eval/benchmark/answer_lengths_v2_fresh.csv"
OUT_SUM = "/root/traffic-llm/eval/benchmark/answer_lengths_summary.json"
END_PUNCT = tuple("。!?！？.)】”\"』」）")

tok = AutoTokenizer.from_pretrained("/root/models/Qwen3-14B", trust_remote_code=True)
rows, summary = [], {}
for layer, path in FILES.items():
    lens, p1200_endpunct, short_endpunct, short_n, n1200 = [], 0, 0, 0, 0
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        obj = json.loads(line)
        ans = [m["content"] for m in obj["messages"] if m["role"] == "assistant"][-1]
        L = len(tok(ans, add_special_tokens=False)["input_ids"])
        lens.append(L)
        rows.append((layer, L))
        ends = ans.rstrip().endswith(END_PUNCT)
        if L == 1200:
            n1200 += 1
            p1200_endpunct += int(ends)
        if L < 1100:
            short_n += 1
            short_endpunct += int(ends)
    summary[layer] = {"n": len(lens), "exact_1200": n1200, "max": max(lens),
        "median": statistics.median(lens),
        "endpunct_rate_1200": round(p1200_endpunct / n1200, 4) if n1200 else None,
        "endpunct_rate_lt1100": round(short_endpunct / short_n, 4) if short_n else None}
    print(layer, summary[layer], flush=True)

total = len(rows)
t1200 = sum(1 for _, L in rows if L == 1200)
summary["_total"] = {"n": total, "exact_1200": t1200, "share_1200": round(t1200 / total, 4)}
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["layer", "answer_tokens"]); w.writerows(rows)
json.dump(summary, open(OUT_SUM, "w"), ensure_ascii=False, indent=2)
print("TOTAL", summary["_total"], flush=True)
print("saved", OUT_CSV, OUT_SUM, flush=True)
