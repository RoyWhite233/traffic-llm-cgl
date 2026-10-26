# -*- coding: utf-8 -*-
"""真实RAG端到端评测: RAGFlow检索「标准规范知识库」→ hbjgy-14b作答252题。
- 端到端精度(硬指标) + 近似检索命中率(切片是否含该题source的规范号/表条号)
- token由本脚本运行时从mysql自取,不打印。
- 冒烟: EVAL_LIMIT=3 先看检索返回结构。
"""
import json, re, os, subprocess, time
from collections import defaultdict
import requests
from openai import OpenAI

KB_ID = "a10cc8e4387411f19a3cbd8efb86d340"
BENCH = "/root/traffic-llm/eval/benchmark/test.jsonl"
OUT = "/root/traffic-llm/eval/benchmark/result_rag_e2e.jsonl"
TOPK = 6
RAGFLOW = "http://172.31.101.100"

def get_token():
    sql = ("SELECT token FROM rag_flow.api_token WHERE tenant_id="
           "(SELECT tenant_id FROM rag_flow.knowledgebase WHERE id='%s') LIMIT 1;" % KB_ID)
    out = subprocess.check_output(
        ["docker", "exec", "docker-mysql-1", "mysql", "-uroot", "-pinfini_rag_flow", "-N", "-e", sql],
        stderr=subprocess.DEVNULL).decode().strip()
    assert out.startswith("ragflow-"), "未取到token"
    return out

TOKEN = get_token()
HDR = {"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"}
llm = OpenAI(base_url="http://localhost:8000/v1", api_key="x")
SYS = "你是交通工程勘察设计领域专家。下面是一道单选题和检索到的规范资料,请依据资料判断,只回答正确选项的字母(A、B、C 或 D),不要输出任何解释。"

rows = [json.loads(l) for l in open(BENCH, encoding="utf-8") if l.strip()]
LIMIT = int(os.environ.get("EVAL_LIMIT", "0"))
if LIMIT: rows = rows[:LIMIT]
print("题目数:", len(rows), flush=True)

def retrieve(q):
    r = requests.post(RAGFLOW + "/api/v1/retrieval", headers=HDR, timeout=60,
        json={"question": q, "dataset_ids": [KB_ID], "page": 1, "page_size": TOPK,
              "similarity_threshold": 0.1, "vector_similarity_weight": 0.5})
    j = r.json()
    if j.get("code") not in (0, None): raise RuntimeError("retrieval err: %s" % str(j)[:200])
    chunks = (j.get("data") or {}).get("chunks") or []
    texts, docs = [], []
    for c in chunks[:TOPK]:
        t = c.get("content") or c.get("content_with_weight") or ""
        texts.append(re.sub(r"\s+", " ", t)[:700])
        docs.append(c.get("document_keyword") or c.get("docnm_kwd") or c.get("document_name") or "")
    return texts, docs, j

def src_keys(source):
    """从source(如'JTG C20-2011 表3.2.2-1(岩石坚硬程度分类)')提取匹配键"""
    keys = []
    m = re.findall(r"(?:JTG|GB|JT)[/T\s]*[0-9A-Z]+(?:-\d{4})?", source or "")
    keys += m
    keys += re.findall(r"表\s*[\d.\-]+", source or "")
    keys += re.findall(r"第?\s*[\d.]+\s*[条节]", source or "")
    return [k.replace(" ", "") for k in keys if len(k) >= 3]

def extract(t):
    m = re.search(r"[ABCD]", (t or "").upper()); return m.group(0) if m else "?"

def ask(q, ctxs):
    ref = "\n".join("[%d] %s" % (i + 1, c) for i, c in enumerate(ctxs)) or "(未检索到资料)"
    user = "检索到的规范资料:\n%s\n\n题目:\n%s\n\n答案(只写字母):" % (ref, q["question"])
    for a in range(3):
        try:
            r = llm.chat.completions.create(model="hbjgy-14b",
                messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                temperature=0, max_tokens=16,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            return extract(r.choices[0].message.content)
        except Exception:
            if a == 2: return "?"
            time.sleep(2)

done = set()
if os.path.exists(OUT):
    for l in open(OUT, encoding="utf-8"):
        try: done.add(json.loads(l)["id"])
        except: pass

out = open(OUT, "a", encoding="utf-8")
n_ok = n_hit = n = 0
bycat = defaultdict(lambda: [0, 0])
t0 = time.time()
for i, q in enumerate(rows):
    if q["id"] in done: continue
    try:
        ctxs, docs, raw = retrieve(q["question"])
    except Exception as e:
        print("  检索失败", q["id"], str(e)[:100], flush=True); ctxs, docs = [], []
    if LIMIT and i == 0:
        print("== 冒烟: 首题切片来源文档 ==", flush=True)
        for d in docs[:3]: print("  doc:", d[:70], flush=True)
    ans = ask(q, ctxs)
    ok = ans == q["ground_truth"].strip().upper()[:1]
    keys = src_keys(q.get("source", ""))
    # 规范级命中: 规范号(如JTGC20-2011)出现在切片文本或来源文档名中
    blob = (" ".join(ctxs) + " " + " ".join(docs)).replace(" ", "").replace("/", "").upper()
    spec_keys = [k.upper().replace("/", "") for k in keys if re.match(r"(?:JTG|GB|JT)", k, re.I)]
    hit = any(k in blob for k in spec_keys) if spec_keys else None
    out.write(json.dumps({"id": q["id"], "cat": q["category"], "ans": ans,
                          "gt": q["ground_truth"], "ok": ok, "hit": hit,
                          "n_chunks": len(ctxs)}, ensure_ascii=False) + "\n")
    out.flush()
    n += 1; n_ok += ok
    if hit: n_hit += 1
    bycat[q["category"]][1] += 1; bycat[q["category"]][0] += ok
    if n % 20 == 0:
        print("  %d/%d acc=%.1f%% hit≈%.1f%% (%.0fs)" % (
            n, len(rows) - len(done), 100 * n_ok / n, 100 * n_hit / n, time.time() - t0), flush=True)
out.close()

# 汇总(含历史done)
allr = [json.loads(l) for l in open(OUT, encoding="utf-8")]
seen = {}
for r in allr: seen[r["id"]] = r
allr = list(seen.values())
tot = len(allr); ok = sum(r["ok"] for r in allr)
hits = [r for r in allr if r["hit"] is not None]
print("\n===== 真实RAG端到端 =====", flush=True)
print("端到端精度: %d/%d = %.1f%%" % (ok, tot, 100 * ok / tot), flush=True)
print("近似检索命中率(source号在切片中): %.1f%% (%d/%d可判)" % (
    100 * sum(r["hit"] for r in hits) / max(1, len(hits)), sum(r["hit"] for r in hits), len(hits)), flush=True)
cat = defaultdict(lambda: [0, 0])
for r in allr: cat[r["cat"]][1] += 1; cat[r["cat"]][0] += r["ok"]
for c, (a, b) in sorted(cat.items(), key=lambda x: -x[1][1]):
    print("  %-16s %d/%d = %.0f%%" % (c, a, b, 100 * a / b), flush=True)
print("命中且答对: %d | 命中但答错: %d | 未命中但答对: %d | 未命中且答错: %d" % (
    sum(1 for r in hits if r["hit"] and r["ok"]), sum(1 for r in hits if r["hit"] and not r["ok"]),
    sum(1 for r in hits if not r["hit"] and r["ok"]), sum(1 for r in hits if not r["hit"] and not r["ok"])), flush=True)
print("RAG_E2E_DONE", flush=True)
