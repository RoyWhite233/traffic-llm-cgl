# -*- coding: utf-8 -*-
"""W-E step 1 (v2): fetch candidate specification-clause texts, document-scoped.

Plan of record for the 4th control condition (raw clause text, no explanations):
  1. Resolve each item's cited specification to the exact PDF in the RAGFlow KB
     (document-name match on the spec number).
  2. Retrieve top chunks RESTRICTED to that document, using the question text as the
     query (the KB holds only raw spec text, so question-based retrieval cannot leak
     answer-oriented explanations; document scoping removes cross-book noise).
  3. Rank candidates: clause/table-key hit first, then similarity.
  4. A human reviews candidates per item and confirms/edits the clause text; items with
     no recoverable clause text are EXCLUDED and counted, never padded from the
     benchmark's explanation field.

Output: eval/benchmark/clause_candidates.jsonl
  {id, source, spec, clausekeys, doc_id, doc_name, status, candidates:[{text, score, key_hit}]}
status: doc+key = right book, clause/table key seen in some chunk
        doc     = right book found, key not seen in retrieved chunks
        nodoc   = cited spec book not in KB
CPU-side only; safe during GPU training. Resumable (append + skip done).
"""
import json, re, os, subprocess, time
import requests

KB = "a10cc8e4387411f19a3cbd8efb86d340"
RF = "http://172.31.101.100"
BENCH = "/root/traffic-llm/eval/benchmark/test.jsonl"
OUT = "/root/traffic-llm/eval/benchmark/clause_candidates.jsonl"
TOPK = 8


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


def parse_source(source):
    m = re.search(r"((?:JTG|GB|JT)(?:\s*/\s*T)?\s*[0-9A-Z]+(?:[—\-–]\d{4})?)", source or "")
    spec = re.sub(r"[\s]+", " ", m.group(1)).strip() if m else None
    keys = [k.replace(" ", "") for k in re.findall(r"表\s*[\d.\-]+", source or "")]
    keys += [k.replace(" ", "") for k in re.findall(r"[第]?\s*\d+(?:\.\d+)+\s*条?", source or "")]
    keys = [re.sub(r"条$", "", k) for k in dict.fromkeys(keys) if len(re.sub(r"\D", "", k)) >= 2]
    return spec, keys


def spec_code(spec):
    """'JTG C20-2011' -> 'C20'; 'JTG 3370' -> '3370'; 'GB 50026-2007' -> '50026'"""
    if not spec:
        return None
    s = re.sub(r"^(JTG|GB|JT)(\s*/\s*T)?", "", spec).strip()
    s = s.split("-")[0].split("—")[0].strip().replace(" ", "")
    return s or None


_doc_cache = {}


def find_doc(spec):
    code = spec_code(spec)
    if not code:
        return None, None
    if code in _doc_cache:
        return _doc_cache[code]
    r = requests.get(RF + "/api/v1/datasets/%s/documents" % KB, headers=H,
                     params={"keywords": code, "page_size": 30}, timeout=30).json()
    docs = (r.get("data") or {}).get("docs") or []
    # prefer names whose spec code segment matches exactly (avoid C20 matching 3512 etc.)
    pat = re.compile(r"(?<![0-9])" + re.escape(code) + r"(?![0-9])")
    hit = [d for d in docs if pat.search(d["name"].upper().replace(" ", ""))]
    res = (hit[0]["id"], hit[0]["name"]) if hit else (None, None)
    _doc_cache[code] = res
    return res


def retrieve(question, doc_id):
    body = {"question": question, "dataset_ids": [KB], "page": 1, "page_size": TOPK,
            "similarity_threshold": 0.05, "vector_similarity_weight": 0.5}
    if doc_id:
        body["document_ids"] = [doc_id]
    r = requests.post(RF + "/api/v1/retrieval", headers=HJ, timeout=60, json=body).json()
    if r.get("code") not in (0, None):
        raise RuntimeError(str(r)[:200])
    out = []
    for c in ((r.get("data") or {}).get("chunks") or [])[:TOPK]:
        t = re.sub(r"\s+", " ", c.get("content") or c.get("content_with_weight") or "")
        out.append({"text": t[:2000], "score": round(float(c.get("similarity") or 0), 4)})
    return out


def main():
    rows = [json.loads(l) for l in open(BENCH, encoding="utf-8") if l.strip()]
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass
    todo = [r for r in rows if r["id"] not in done]
    print("total %d done %d todo %d" % (len(rows), len(done), len(todo)), flush=True)

    stats = {"doc+key": 0, "doc": 0, "nodoc": 0}
    with open(OUT, "a", encoding="utf-8") as f:
        for i, r in enumerate(todo):
            spec, keys = parse_source(r.get("source", ""))
            doc_id, doc_name = find_doc(spec)
            cands = []
            if doc_id:
                # question text as query, scoped to the cited book; plus a clause-key probe
                queries = [re.sub(r"\n[A-D]\..*", "", r["question"], flags=re.S)]
                if keys:
                    queries.append(" ".join(keys))
                seen = set()
                for q in queries:
                    try:
                        for c in retrieve(q, doc_id):
                            h = hash(c["text"][:300])
                            if h not in seen:
                                seen.add(h)
                                cands.append(c)
                    except Exception as e:
                        print("retry %s: %s" % (r["id"], str(e)[:80]), flush=True)
                        time.sleep(5)
                for c in cands:
                    c["key_hit"] = any(k in c["text"].replace(" ", "") for k in keys) if keys else False
                cands.sort(key=lambda c: (not c["key_hit"], -c["score"]))
                cands = cands[:TOPK]
            status = "nodoc" if not doc_id else ("doc+key" if any(c["key_hit"] for c in cands) else "doc")
            stats[status] += 1
            f.write(json.dumps({"id": r["id"], "source": r.get("source", ""), "spec": spec,
                                "clausekeys": keys, "doc_id": doc_id, "doc_name": doc_name,
                                "status": status, "candidates": cands}, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print("%d/%d  %s" % (i + 1, len(todo), stats), flush=True)
            time.sleep(0.2)
    print("done:", stats, flush=True)


if __name__ == "__main__":
    main()
