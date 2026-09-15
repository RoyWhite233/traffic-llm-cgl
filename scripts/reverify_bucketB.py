# -*- coding: utf-8 -*-
"""Bucket-B clause re-verification: re-retrieve the cited provision for items whose
recovered clause mismatched the citation, AI-select the correct chunk, and build
clause_confirmed_v2.json. Items still unrecoverable are excluded with reason."""
import json, re, sys, time, os
import requests
sys.path.insert(0, "/root/traffic-llm/scripts_repro")
import fetch_clause_candidates as F

B = "/root/traffic-llm/eval/benchmark/"
tri = [json.loads(l) for l in open(B + "precheck_triage.jsonl", encoding="utf-8")]
Bs = [t for t in tri if t["bucket"] == "B"]
bench = {json.loads(l)["id"]: json.loads(l) for l in open(B + "test.jsonl", encoding="utf-8") if l.strip()}
conf = json.load(open(B + "clause_confirmed.json", encoding="utf-8"))
POOL = "http://localhost:3000/v1/chat/completions"
MODEL = "DeepSeek-V3-1-w8a8"
TOKEN = os.environ["RELAY_API_KEY"]

SYS = ("你是规范条文核对员。给你一道题(含出处与解析)和若干检索片段,找出真正对应题目所引条文/表格、"
       "且包含答题所需内容的片段。只输出JSON:"
       '{"pick":0,"formal_match":true,"note":"..."}'
       "(pick 填片段编号,任何片段都不含答题所需内容时填 -1;note ≤30字)。"
       "要求:pick 的片段须包含被引条款/表号的实质内容(表号编排可因版式差异不同,以内容为准);"
       "不得选只有出处引用而无实质规定的片段。")

out2 = {}
log = open(B + "reverify_bucketB.log.jsonl", "w", encoding="utf-8")
for n, t in enumerate(Bs):
    q = bench[t["id"]]
    spec, keys = F.parse_source(q.get("source", ""))
    doc_id, doc_name = F.find_doc(spec)
    issue_txt = "; ".join(t.get("issues", []))
    queries = []
    hint = re.findall(r"表\s*[\d.\-]+|[\d]+(?:\.\d+)+\s*条", issue_txt + " " + q.get("source", ""))
    if hint:
        queries.append(" ".join(dict.fromkeys(hint))[:80] + " " + q["question"][:150])
    queries.append(q["question"][:200])
    queries.append(q.get("explanation", "")[:200])
    cands = []
    for qq in queries:
        if not qq.strip():
            continue
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
    uniq = uniq[:12]
    ctx = "\n".join("[%d] %s" % (i + 1, c["text"][:900]) for i, c in enumerate(uniq)) or "(空)"
    body = "题目:%s\n标准答案:%s\n出处:%s\n解析:%s\n\n候选片段(文档:%s):\n%s" % (
        q["question"], q["ground_truth"], q.get("source", ""), q.get("explanation", "")[:300], doc_name, ctx)
    v = {"pick": -1, "note": "ERROR"}
    for a in range(4):
        try:
            r = requests.post(POOL, headers={"Authorization": "Bearer " + TOKEN},
                json={"model": MODEL, "messages": [{"role": "system", "content": SYS},
                      {"role": "user", "content": body}], "temperature": 0, "max_tokens": 150,
                      "stream": False}, timeout=150)
            txt = r.json()["choices"][0]["message"]["content"]
            mm = re.search(r"\{.*\}", txt, re.S)
            if mm:
                v = json.loads(mm.group(0))
                break
        except Exception:
            time.sleep(6 * (a + 1))
    pick = v.get("pick", -1)
    if isinstance(pick, int) and 1 <= pick <= len(uniq):
        out2[t["id"]] = {"status": "confirmed", "clause_text": uniq[pick - 1]["text"],
                         "provenance": "reverify-20260915", "note": str(v.get("note", ""))[:60]}
    else:
        out2[t["id"]] = {"status": "excluded", "clause_text": "",
                         "provenance": "reverify-20260915",
                         "note": "reverify-unrecovered: " + str(v.get("note", ""))[:50]}
    log.write(json.dumps({"id": t["id"], **v}, ensure_ascii=False) + "\n")
    log.flush()
    if n % 10 == 0:
        print(n, "/", len(Bs), flush=True)
log.close()
newconf = dict(conf)
changed = kept = dropped = 0
for iid, rec in out2.items():
    if rec["status"] == "confirmed":
        if rec["clause_text"] != conf.get(iid, {}).get("clause_text"):
            changed += 1
        else:
            kept += 1
    else:
        dropped += 1
    newconf[iid] = rec
json.dump(newconf, open(B + "clause_confirmed_v2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
inc = sum(1 for v in newconf.values() if v["status"] == "confirmed")
print("REVERIFY_DONE changed=%d same=%d newly_excluded=%d | v2 included=%d" % (changed, kept, dropped, inc), flush=True)
