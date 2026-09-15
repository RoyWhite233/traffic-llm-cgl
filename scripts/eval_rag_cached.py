# -*- coding: utf-8 -*-
"""Evaluate a model on 252 items with CACHED retrieval contexts (identical inputs).
Env: RC_MODEL (served model name), RC_OUT (output json). Protocol = eval_rag_e2e.py."""
import json, re, os, time
from openai import OpenAI

BENCH = "/root/traffic-llm/eval/benchmark/test.jsonl"
CACHE = "/root/traffic-llm/eval/benchmark/rag_ctx_cache.jsonl"
MODEL = os.environ["RC_MODEL"]; OUT = os.environ["RC_OUT"]
cli = OpenAI(base_url="http://localhost:8001/v1", api_key="x")
SYS = "你是交通工程勘察设计领域专家。下面是一道单选题和检索到的规范资料,请依据资料判断,只回答正确选项的字母(A、B、C 或 D),不要输出任何解释。"
rows = [json.loads(l) for l in open(BENCH, encoding="utf-8") if l.strip()]
cache = {json.loads(l)["id"]: json.loads(l) for l in open(CACHE, encoding="utf-8") if l.strip()}

def extract(t):
    m = re.search(r"[ABCD]", (t or "").upper()); return m.group(0) if m else "?"

preds = []
for i, q in enumerate(rows):
    ctxs = cache[q["id"]]["ctxs"]
    ref = "\n".join("[%d] %s" % (j + 1, c) for j, c in enumerate(ctxs)) or "(未检索到资料)"
    user = "检索到的规范资料:\n%s\n\n题目:\n%s\n\n答案(只写字母):" % (ref, q["question"])
    ans = "?"
    for a in range(3):
        try:
            r = cli.chat.completions.create(model=MODEL,
                messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                temperature=0, max_tokens=16,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            ans = extract(r.choices[0].message.content); break
        except Exception:
            time.sleep(3)
    preds.append(ans)
    if i % 25 == 0: print(MODEL, i, "/", len(rows), flush=True)
gts = [r["ground_truth"] for r in rows]
correct = sum(p == g for p, g in zip(preds, gts))
json.dump({"model": MODEL, "preds": preds, "gts": gts, "correct": correct,
           "acc": correct / len(gts)}, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("RESULT", MODEL, correct, "/", len(gts), flush=True)
