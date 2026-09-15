# -*- coding: utf-8 -*-
"""Bucket-A adjudication (user-delegated, protocol documented): for each item whose
ground-truth answer was challenged by the precheck, retrieve the governing clauses
from the spec corpus and let a strong judge rule keep/change/uncertain.
Conservative rule: change GT ONLY on conclusive clause evidence."""
import json, re, sys, time, os
import requests
sys.path.insert(0, "/root/traffic-llm/scripts_repro")
import fetch_clause_candidates as F

B = "/root/traffic-llm/eval/benchmark/"
tri = [json.loads(l) for l in open(B + "precheck_triage.jsonl", encoding="utf-8")]
A = [t for t in tri if t["bucket"] == "A"]
bench = {json.loads(l)["id"]: json.loads(l) for l in open(B + "test.jsonl", encoding="utf-8") if l.strip()}
POOL = "http://localhost:3000/v1/chat/completions"
MODEL = "DeepSeek-V3-1-w8a8"
TOKEN = os.environ["RELAY_API_KEY"]

SYS = ("你是公路工程规范权威裁决人。一道基准题的标准答案受到质疑,给你题目、原标准答案、解析、质疑理由,"
       "以及从规范库检索到的条文片段。请裁决,只输出JSON:"
       '{"ruling":"维持|改答案|存疑","new_answer":null,"rationale":"..."}'
       "(new_answer 在 ruling=改答案 时填 A/B/C/D,否则 null;rationale ≤60字并引用条文依据)。"
       "规则:仅当检索条文确凿证明另一选项正确时才判改答案;条文不足以确证错误时判维持(质疑不成立)或存疑(确实无法判断);不要臆测。")

out = open(B + "bucketA_adjudication.jsonl", "w", encoding="utf-8")
for t in A:
    q = bench[t["id"]]
    spec, keys = F.parse_source(q.get("source", ""))
    doc_id, doc_name = F.find_doc(spec)
    cands = []
    queries = [q["question"][:200], (q.get("source", "") + " " + q.get("explanation", ""))[:200]]
    issue_txt = "; ".join(t.get("issues", []))
    m = re.findall(r"表\s*[\d.\-]+|[\d]+(?:\.\d+)+\s*条", issue_txt)
    if m:
        queries.append(" ".join(m[:3]) + " " + q["question"][:120])
    for qq in queries:
        try:
            cands += F.retrieve(qq, doc_id)
        except Exception:
            pass
    seen, uniq = set(), []
    for c in cands:
        k = c["text"][:80]
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    ctx = "\n".join("[%d] %s" % (i + 1, c["text"][:900]) for i, c in enumerate(uniq[:10])) or "(未检索到)"
    body = ("题目:%s\n原标准答案:%s\n出处:%s\n解析:%s\n质疑理由:%s\n\n检索到的条文片段(文档:%s):\n%s" % (
        q["question"], q["ground_truth"], q.get("source", ""), q.get("explanation", "")[:400],
        issue_txt[:500], doc_name, ctx))
    v = {"ruling": "ERROR"}
    for a in range(4):
        try:
            r = requests.post(POOL, headers={"Authorization": "Bearer " + TOKEN},
                json={"model": MODEL, "messages": [{"role": "system", "content": SYS},
                      {"role": "user", "content": body}], "temperature": 0, "max_tokens": 300,
                      "stream": False}, timeout=150)
            txt = r.json()["choices"][0]["message"]["content"]
            mm = re.search(r"\{.*\}", txt, re.S)
            if mm:
                v = json.loads(mm.group(0))
                break
        except Exception:
            time.sleep(6 * (a + 1))
    rec = {"id": t["id"], "gt": q["ground_truth"], **v}
    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out.flush()
    print(t["id"], v.get("ruling"), v.get("new_answer"), "|", str(v.get("rationale", ""))[:80], flush=True)
out.close()
print("ADJUDICATE_DONE", flush=True)
