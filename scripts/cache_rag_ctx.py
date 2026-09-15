# -*- coding: utf-8 -*-
"""Cache RAGFlow retrieval contexts for all 252 items (C21: identical contexts for
base-vs-v2 paired comparison; reviewer asked for cached retrieval + unified env)."""
import json, re, os, time
import requests

KB_ID = "a10cc8e4387411f19a3cbd8efb86d340"
BENCH = "/root/traffic-llm/eval/benchmark/test.jsonl"
OUT = "/root/traffic-llm/eval/benchmark/rag_ctx_cache.jsonl"
TOPK = 6
RAGFLOW = "http://172.31.101.100"

TOKEN = os.environ["RAGFLOW_API_KEY"]
HDR = {"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"}

def retrieve(q):
    r = requests.post(RAGFLOW + "/api/v1/retrieval", headers=HDR, timeout=60,
        json={"question": q, "dataset_ids": [KB_ID], "top_k": 1024, "page_size": 30})
    j = r.json()
    if j.get("code") not in (0, None): raise RuntimeError(str(j)[:200])
    chunks = (j.get("data") or {}).get("chunks") or []
    texts, docs = [], []
    for c in chunks[:TOPK]:
        t = c.get("content") or c.get("content_with_weight") or ""
        texts.append(re.sub(r"\s+", " ", t)[:700])
        docs.append(c.get("document_keyword") or c.get("docnm_kwd") or c.get("document_name") or "")
    return texts, docs

rows = [json.loads(l) for l in open(BENCH, encoding="utf-8") if l.strip()]
done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["id"] for l in open(OUT, encoding="utf-8") if l.strip()}
out = open(OUT, "a", encoding="utf-8")
for i, q in enumerate(rows):
    if q["id"] in done: continue
    for a in range(3):
        try:
            ctxs, docs = retrieve(q["question"]); break
        except Exception as e:
            if a == 2: ctxs, docs = [], []; print("retrieve fail", q["id"], str(e)[:80], flush=True)
            time.sleep(3)
    out.write(json.dumps({"id": q["id"], "ctxs": ctxs, "docs": docs}, ensure_ascii=False) + "\n")
    out.flush()
    if i % 25 == 0: print(i, "/", len(rows), flush=True)
out.close()
n = sum(1 for _ in open(OUT, encoding="utf-8"))
print("CACHE_DONE", n, flush=True)
