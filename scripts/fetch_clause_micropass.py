# -*- coding: utf-8 -*-
"""W-E micro-pass: hand-picked keyword retrieval for the last unresolved items.

Each item gets an explicit (doc-hint, keywords) pair chosen during per-item AI review.
Searches chunk keyword API within the hinted document (falling back to KB-wide vector
search restricted by doc-name match). Results go to clause_micro.jsonl for final review.
"""
import json, re, os, time
import requests

KB = "a10cc8e4387411f19a3cbd8efb86d340"
RF = "http://172.31.101.100"
B = "/root/traffic-llm/eval/benchmark"

# item -> (doc name keyword to locate PDF, [chunk keywords to try])
PLAN = {
    "geo-037":  ("C20",   ["冒落", "三带", "弯曲带"]),
    "geo-038":  ("C20",   ["自由膨胀率", "膨胀土 判"]),
    "kc-010":   ("D70",   ["建筑限界", "净高"]),
    "pave-013": ("D30",   ["换填", "换填深度"]),
    "pave-014": ("D30",   ["垫层 0.5", "砂砾垫层", "排水垫层 厚度"]),
    "pave-015": ("D30",   ["堆载预压", "天然地基", "排水固结 适用"]),
    "pave-016": ("D30",   ["塑料排水板 深度", "处理深度", "竖向排水体"]),
    "pave-021": ("B01",   ["3.75", "车道宽度"]),
    "pave-022": ("B01",   ["土路肩"]),
    "pave-023": ("D30",   ["台背", "结构物回填"]),
    "pave-029": ("F40",   ["98%", "压实度 马歇尔"]),
    "pave-036": ("B01",   ["路拱"]),
    "pave-040": ("D40",   ["面层厚度", "0.24"]),
    "surv-019": ("C30",   ["重现期"]),
    "surv-020": ("C30",   ["频率分析", "实测流量资料"]),
    "traf-009": ("D82",   ["绿底", "指路标志 颜色"]),
    "traf-011": ("D82",   ["主标志", "辅助标志"]),
    "traf-018": ("D81",   ["净区"]),
    "tun-013":  ("D70",   ["半横向", "通风方式"]),
    "tun-016":  ("照明",  ["中间段", "入口段 过渡段"]),
    "tun-017":  ("照明",  ["L20", "洞外亮度"]),
}


H = {"Authorization": "Bearer " + os.environ["RAGFLOW_API_KEY"]}
HJ = {**H, "Content-Type": "application/json"}


def find_docs(name_kw):
    r = requests.get(RF + "/api/v1/datasets/%s/documents" % KB, headers=H,
                     params={"keywords": name_kw, "page_size": 30}, timeout=30).json()
    docs = (r.get("data") or {}).get("docs") or []
    # keep JTG-prefixed docs when possible (avoid TB/T etc. collisions)
    jtg = [d for d in docs if "JTG" in d["name"].upper() or "公路" in d["name"]]
    return jtg or docs


def kw_chunks(doc_id, kw):
    try:
        r = requests.get(RF + "/api/v1/datasets/%s/documents/%s/chunks" % (KB, doc_id),
                         headers=H, params={"keywords": kw, "page": 1, "page_size": 5},
                         timeout=45).json()
        return [re.sub(r"\s+", " ", c.get("content") or "")[:1800]
                for c in ((r.get("data") or {}).get("chunks") or [])]
    except Exception:
        return []


def vec(doc_ids, q):
    try:
        r = requests.post(RF + "/api/v1/retrieval", headers=HJ, timeout=60,
                          json={"question": q, "dataset_ids": [KB], "document_ids": doc_ids,
                                "page": 1, "page_size": 5, "similarity_threshold": 0.02,
                                "vector_similarity_weight": 0.6}).json()
        return [re.sub(r"\s+", " ", c.get("content") or "")[:1800]
                for c in ((r.get("data") or {}).get("chunks") or [])]
    except Exception:
        return []


bench = {json.loads(l)["id"]: json.loads(l) for l in open(B + "/test.jsonl", encoding="utf-8") if l.strip()}
out = open(B + "/clause_micro.jsonl", "w", encoding="utf-8")
for iid, (dkw, kws) in PLAN.items():
    docs = find_docs(dkw)
    doc_ids = [d["id"] for d in docs][:4]
    names = [d["name"] for d in docs][:4]
    cands, seen = [], set()
    for did in doc_ids:
        for kw in kws:
            for t in kw_chunks(did, kw):
                h = hash(t[:250])
                if h not in seen:
                    seen.add(h)
                    cands.append(t)
    if len(cands) < 2 and doc_ids:
        for t in vec(doc_ids, " ".join(kws) + " " + bench[iid]["question"][:80]):
            h = hash(t[:250])
            if h not in seen:
                seen.add(h)
                cands.append(t)
    out.write(json.dumps({"id": iid, "docs": names, "kw": kws,
                          "cands": cands[:5]}, ensure_ascii=False) + "\n")
    out.flush()
    print(iid, "docs=%d cands=%d" % (len(doc_ids), len(cands)), flush=True)
    time.sleep(0.2)
out.close()
print("micro done")
