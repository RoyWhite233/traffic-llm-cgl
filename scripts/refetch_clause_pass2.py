# -*- coding: utf-8 -*-
"""W-E step 2a: enhanced second-pass retrieval for items whose clause text was not
auto-confirmed in pass 1.

Pass 1 used (question text | clause keys) scoped to the cited document. For the ~95
unresolved items this pass adds:
  - query variants: bare table/clause numbers, spec-topic words from the source string,
    question keywords stripped of option lines
  - larger page_size (12) per query
  - keyword search fallback: RAGFlow chunk-list API with keyword filter on the doc

Output: eval/benchmark/clause_candidates_pass2.jsonl (same schema as pass 1).
CPU-side only.
"""
import json, re, os, subprocess, time
import requests

KB = "a10cc8e4387411f19a3cbd8efb86d340"
RF = "http://172.31.101.100"
BENCH = "/root/traffic-llm/eval/benchmark/test.jsonl"
P1 = "/root/traffic-llm/eval/benchmark/clause_candidates.jsonl"
OUT = "/root/traffic-llm/eval/benchmark/clause_candidates_pass2.jsonl"
TOPK = 12


def get_token():
    sql = ("SELECT token FROM rag_flow.api_token WHERE tenant_id="
           "(SELECT tenant_id FROM rag_flow.knowledgebase WHERE id='%s') LIMIT 1;" % KB)
    out = subprocess.check_output(
        ["docker", "exec", "docker-mysql-1", "mysql", "-uroot", "-pinfini_rag_flow", "-N", "-e", sql],
        stderr=subprocess.DEVNULL).decode().strip()
    assert out.startswith("ragflow-")
    return out


H = {"Authorization": "Bearer " + get_token()}
HJ = {**H, "Content-Type": "application/json"}


def numcore(k):
    return re.sub(r"^[第表条\s]+|[条节\s]+$", "", k)


def retrieve(question, doc_id, topk=TOPK):
    body = {"question": question, "dataset_ids": [KB], "page": 1, "page_size": topk,
            "similarity_threshold": 0.02, "vector_similarity_weight": 0.5}
    if doc_id:
        body["document_ids"] = [doc_id]
    r = requests.post(RF + "/api/v1/retrieval", headers=HJ, timeout=60, json=body).json()
    if r.get("code") not in (0, None):
        raise RuntimeError(str(r)[:150])
    out = []
    for c in ((r.get("data") or {}).get("chunks") or [])[:topk]:
        t = re.sub(r"\s+", " ", c.get("content") or c.get("content_with_weight") or "")
        out.append({"text": t[:2200], "score": round(float(c.get("similarity") or 0), 4)})
    return out


def list_chunks(doc_id, kw):
    """keyword-filtered chunk listing inside one document"""
    try:
        r = requests.get(RF + "/api/v1/datasets/%s/documents/%s/chunks" % (KB, doc_id),
                         headers=H, params={"keywords": kw, "page": 1, "page_size": 8},
                         timeout=45).json()
        out = []
        for c in ((r.get("data") or {}).get("chunks") or []):
            t = re.sub(r"\s+", " ", c.get("content") or "")
            out.append({"text": t[:2200], "score": 0.0})
        return out
    except Exception:
        return []


def main():
    bench = {json.loads(l)["id"]: json.loads(l) for l in open(BENCH, encoding="utf-8") if l.strip()}
    p1 = [json.loads(l) for l in open(P1, encoding="utf-8")]

    # unresolved = no candidate whose de-spaced text contains a numeric clause core
    todo = []
    for r in p1:
        keys = [numcore(k) for k in r.get("clausekeys", []) if numcore(k)]
        hit = False
        for c in r.get("candidates", []):
            tn = re.sub(r"[\s.·．]", "", c["text"])
            for k in keys:
                kd = re.sub(r"[\s.]", "", k)
                if kd and kd in tn:
                    hit = True
        if not (keys and hit):
            todo.append(r)

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass
    todo = [r for r in todo if r["id"] not in done]
    print("unresolved todo:", len(todo), flush=True)

    with open(OUT, "a", encoding="utf-8") as f:
        for i, r in enumerate(todo):
            b = bench[r["id"]]
            doc_id = r.get("doc_id")
            keys = [numcore(k) for k in r.get("clausekeys", []) if numcore(k)]
            # topic words from source: strip spec number and brackets
            topic = re.sub(r"(?:JTG|GB|JT)(?:\s*/\s*T)?\s*[0-9A-Z]+(?:[—\-–]\d{4})?", "", r.get("source", ""))
            topic = re.sub(r"[《》()（）第表条]|\d+(?:\.\d+)*(?:-\d+)?", " ", topic)
            topic = re.sub(r"\s+", " ", topic).strip()
            qstem = re.sub(r"\n[A-D]\..*", "", b["question"], flags=re.S)

            queries = []
            for k in keys:
                queries.append("表" + k)
                queries.append(k)
            if topic:
                queries.append(topic)
                queries.append(topic + " " + qstem[:60])
            queries.append(qstem)

            cands, seen = [], set()
            for q in queries[:6]:
                try:
                    got = retrieve(q, doc_id)
                except Exception:
                    time.sleep(4)
                    try:
                        got = retrieve(q, doc_id)
                    except Exception:
                        got = []
                for c in got:
                    h = hash(c["text"][:280])
                    if h not in seen:
                        seen.add(h)
                        cands.append(c)
            # keyword chunk-listing fallback on the clause number and topic
            if doc_id:
                for kw in (keys[:1] or []) + ([topic[:20]] if topic else []):
                    for c in list_chunks(doc_id, kw):
                        h = hash(c["text"][:280])
                        if h not in seen:
                            seen.add(h)
                            cands.append(c)

            for c in cands:
                tn = re.sub(r"[\s.·．]", "", c["text"])
                c["key_hit"] = any(re.sub(r"[\s.]", "", k) in tn for k in keys) if keys else False
            cands.sort(key=lambda c: (not c["key_hit"], -c["score"]))

            f.write(json.dumps({"id": r["id"], "source": r.get("source", ""),
                                "doc_id": doc_id, "doc_name": r.get("doc_name"),
                                "clausekeys": keys,
                                "auto2": bool(cands and cands[0]["key_hit"]),
                                "candidates": cands[:8]}, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 20 == 0:
                print("%d/%d" % (i + 1, len(todo)), flush=True)
            time.sleep(0.15)
    print("pass2 done", flush=True)


if __name__ == "__main__":
    main()
