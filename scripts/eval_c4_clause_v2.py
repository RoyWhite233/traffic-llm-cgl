# -*- coding: utf-8 -*-
"""C4 condition: closed-book vs RAW-CLAUSE open book (no answer-oriented explanations).

The original oracle open-book condition fed source-pointer + explanation, which mixes
raw specification information, answer cues and the model's own reasoning (reviewer #2
point 3). This 4th condition feeds ONLY the verified clause text from the specification
books (clause_confirmed_v2.json, human/AI-verified per docs/C4_clause_verification_protocol.md).
Items whose clause text could not be recovered are excluded and reported.

Run AFTER v1' training, with vLLM serving the deployed v2 (hbjgy-14b upstream on :8001,
or shim :8000). Protocol identical to eval_openbook.py: greedy, unified system prompt,
regex letter extraction.

Usage:  /root/miniconda3/envs/llm/bin/python eval/eval_c4_clause.py
Env:    C4_BASE_URL (default http://localhost:8001/v1)   C4_MODEL (default hbjgy-14b)
"""
import json, re, os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

BENCH = "/root/traffic-llm/eval/benchmark/test.jsonl"
CONF = "/root/traffic-llm/eval/benchmark/clause_confirmed_v2.json"
OUT = "/root/traffic-llm/eval/benchmark/result_c4_clause_v2.json"

cli = OpenAI(base_url=os.environ.get("C4_BASE_URL", "http://localhost:8001/v1"), api_key="x")
MODEL = os.environ.get("C4_MODEL", "hbjgy-14b")
SYS = ("你是交通工程勘察设计领域专家。下面是一道单选题,请仔细判断后只回答正确选项的字母"
       "(A、B、C 或 D),不要输出任何解释。")

rows = [json.loads(l) for l in open(BENCH, encoding="utf-8") if l.strip()]
conf = json.load(open(CONF, encoding="utf-8"))
inc = [r for r in rows if conf.get(r["id"], {}).get("status") == "confirmed"]
exc = [r["id"] for r in rows if conf.get(r["id"], {}).get("status") != "confirmed"]
print("items: %d included, %d excluded" % (len(inc), len(exc)), flush=True)


def extract(t):
    m = re.search(r"[ABCD]", (t or "").upper())
    return m.group(0) if m else "?"


def ask(user):
    for a in range(3):
        try:
            r = cli.chat.completions.create(model=MODEL,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": user}],
                temperature=0, max_tokens=24,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            return extract(r.choices[0].message.content)
        except Exception:
            if a == 2:
                return "?"


def closed(r):
    return r["question"] + "\n\n答案(只写字母):"


def c4(r):
    ref = conf[r["id"]]["clause_text"]
    return ("参考资料(规范条文原文,不含解析):\n%s\n\n题目:\n%s\n\n请依据上述条文判断,答案(只写字母):"
            % (ref, r["question"]))


def run(builder, tag):
    preds = [None] * len(inc)

    def job(i):
        preds[i] = ask(builder(inc[i]))
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(job, range(len(inc))))
    gts = [r["ground_truth"] for r in inc]
    correct = sum(p == g for p, g in zip(preds, gts))
    print("[%s] %d/%d = %.1f%%  unans %d" % (tag, correct, len(inc),
          correct / len(inc) * 100, sum(1 for p in preds if p == "?")), flush=True)
    return preds, correct


pc, cc_ = run(closed, "closed(236)")
p4, c4_ = run(c4, "C4 raw-clause(236)")

gts = [r["ground_truth"] for r in inc]
b = sum(1 for x, y, g in zip(pc, p4, gts) if x == g and y != g)
c = sum(1 for x, y, g in zip(pc, p4, gts) if y == g and x != g)
bycat = defaultdict(lambda: [0, 0, 0])
for r, x, y, g in zip(inc, pc, p4, gts):
    k = r["category"]
    bycat[k][2] += 1
    bycat[k][0] += x == g
    bycat[k][1] += y == g

json.dump({"model": MODEL, "included": [r["id"] for r in inc], "excluded": exc,
           "closed_pred": pc, "c4_pred": p4, "gts": gts,
           "closed_correct": cc_, "c4_correct": c4_,
           "mcnemar_b_closedonly": b, "mcnemar_c_c4only": c,
           "bycat": {k: v for k, v in bycat.items()}},
          open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
print("closed-only correct b=%d, C4-only correct c=%d" % (b, c), flush=True)
print("wrote", OUT, flush=True)
