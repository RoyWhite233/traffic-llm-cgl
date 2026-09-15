# -*- coding: utf-8 -*-
"""W-E pass 3: targeted keyword retrieval for the remaining hard items.

Uses domain-term keywords mined from the item's explanation/options to keyword-search
chunks INSIDE the cited document, then ranks candidates by character-3-gram overlap
with the explanation. The explanation itself is never emitted as clause text — it only
guides the search; a human/AI judgment still confirms each candidate afterwards.
"""
import json, re, os, time
import requests

KB = "a10cc8e4387411f19a3cbd8efb86d340"
RF = "http://172.31.101.100"
B = "/root/traffic-llm/eval/benchmark"
TOP = 5


H = {"Authorization": "Bearer " + os.environ["RAGFLOW_API_KEY"]}
HJ = {**H, "Content-Type": "application/json"}


def grams(s, n=3):
    s = re.sub(r"[^一-鿿A-Za-z0-9]", "", s)
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def overlap(a_grams, text):
    g = grams(text)
    return len(a_grams & g) / (len(a_grams) + 1)


def terms_from(text, minlen=3, maxn=8):
    """extract Chinese domain-term candidates: longest runs of CJK chars"""
    runs = re.findall(r"[一-鿿]{%d,}" % minlen, text)
    # prefer longer, distinctive runs; drop generic words
    stop = {"下列哪", "正确的是", "说法正确", "一般可划分", "应符合", "规定", "要求", "下列", "关于"}
    out = []
    for r in sorted(set(runs), key=len, reverse=True):
        if any(s in r for s in stop):
            continue
        out.append(r[:10])
        if len(out) >= maxn:
            break
    return out


def kw_chunks(doc_id, kw):
    try:
        r = requests.get(RF + "/api/v1/datasets/%s/documents/%s/chunks" % (KB, doc_id),
                         headers=H, params={"keywords": kw, "page": 1, "page_size": 6},
                         timeout=45).json()
        return [re.sub(r"\s+", " ", c.get("content") or "")[:2200]
                for c in ((r.get("data") or {}).get("chunks") or [])]
    except Exception:
        return []


def vec_retrieve(q, doc_id):
    try:
        r = requests.post(RF + "/api/v1/retrieval", headers=HJ, timeout=60,
                          json={"question": q, "dataset_ids": [KB], "document_ids": [doc_id],
                                "page": 1, "page_size": 6, "similarity_threshold": 0.02,
                                "vector_similarity_weight": 0.5}).json()
        return [re.sub(r"\s+", " ", c.get("content") or "")[:2200]
                for c in ((r.get("data") or {}).get("chunks") or [])]
    except Exception:
        return []


def main():
    bench = {json.loads(l)["id"]: json.loads(l) for l in open(B + "/test.jsonl", encoding="utf-8") if l.strip()}
    p1 = {json.loads(l)["id"]: json.loads(l) for l in open(B + "/clause_candidates.jsonl", encoding="utf-8")}
    auto = json.load(open(B + "/clause_autoconfirmed.json", encoding="utf-8"))
    hard_ids = [i for i in p1 if i not in auto]
    print("hard items:", len(hard_ids), flush=True)

    out_f = open(B + "/clause_pass3.jsonl", "w", encoding="utf-8")
    for n, iid in enumerate(sorted(hard_ids)):
        b = bench[iid]
        doc_id = p1[iid].get("doc_id")
        expl = b.get("explanation", "")
        qtext = b["question"]
        eg = grams(expl + qtext)

        cands, seen = [], set()
        if doc_id:
            kws = terms_from(expl, 4, 5) + terms_from(qtext, 4, 3)
            for kw in kws[:7]:
                for t in kw_chunks(doc_id, kw):
                    h = hash(t[:280])
                    if h not in seen:
                        seen.add(h)
                        cands.append(t)
            for t in vec_retrieve(expl[:200], doc_id):
                h = hash(t[:280])
                if h not in seen:
                    seen.add(h)
                    cands.append(t)
        ranked = sorted(cands, key=lambda t: -overlap(eg, t))[:TOP]
        out_f.write(json.dumps({"id": iid, "source": b.get("source", ""),
                                "cands": [{"text": t, "ov": round(overlap(eg, t), 3)}
                                          for t in ranked]}, ensure_ascii=False) + "\n")
        out_f.flush()
        if (n + 1) % 10 == 0:
            print("%d/%d" % (n + 1, len(hard_ids)), flush=True)
        time.sleep(0.1)
    out_f.close()
    print("pass3 done", flush=True)


if __name__ == "__main__":
    main()
