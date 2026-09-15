# -*- coding: utf-8 -*-
"""C12 benchmark precheck: LLM cross-checks each of the 252 items — does the verified
clause text (or, if unavailable, the explanation) support the ground-truth answer?
Flags mismatches / version concerns for human spot-check. Judge: DeepSeek via local pool.
Resumable; concurrency 2 (429 lesson)."""
import json, re, os, time, threading, concurrent.futures
import requests

BENCH = "/root/traffic-llm/eval/benchmark/test.jsonl"
CONF = "/root/traffic-llm/eval/benchmark/clause_confirmed.json"
OUT = "/root/traffic-llm/eval/benchmark/precheck_252.jsonl"
BASE = "http://localhost:3000/v1/chat/completions"
MODEL = "DeepSeek-V3-1-w8a8"
TOKEN = os.environ["RELAY_API_KEY"]

rows = [json.loads(l) for l in open(BENCH, encoding="utf-8") if l.strip()]
conf = json.load(open(CONF, encoding="utf-8"))
done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["id"] for l in open(OUT, encoding="utf-8") if l.strip()}
lock = threading.Lock()
out = open(OUT, "a", encoding="utf-8")

SYS = ("你是公路工程规范审核专家。给你一道单选题、其标准答案、出处与解析,以及(若有)从规范原文恢复的条文文本。"
       "请严格核查:1)标准答案是否被条文/解析支撑;2)规范编号与版本是否与题干一致;3)表号/条号引用是否正确;"
       "4)题目是否缺少必要的适用前提。只输出JSON:"
       '{"verdict":"OK|SUSPECT","issues":["..."],"confidence":"high|low"}。'
       "verdict取SUSPECT当且仅当发现实质问题(答案错误、版本不符、引用错误、前提缺失)。")

def check(q):
    if q["id"] in done: return
    c = conf.get(q["id"], {})
    clause = c.get("clause_text") or c.get("text") or ""
    body = "题目:%s\n标准答案:%s\n出处:%s\n解析:%s\n" % (
        q["question"], q["ground_truth"], q.get("source", ""), q.get("explanation", ""))
    body += ("已核实条文原文:%s" % clause[:1500]) if clause else "(该题条文未能恢复,仅按解析与出处核查)"
    for a in range(4):
        try:
            r = requests.post(BASE, headers={"Authorization": "Bearer " + TOKEN},
                json={"model": MODEL, "messages": [{"role": "system", "content": SYS},
                      {"role": "user", "content": body}],
                      "temperature": 0, "max_tokens": 300, "stream": False}, timeout=120)
            t = r.json()["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", t, re.S)
            v = json.loads(m.group(0)) if m else {"verdict": "PARSE_FAIL", "issues": [t[:150]]}
            with lock:
                out.write(json.dumps({"id": q["id"], "cat": q["category"],
                    "has_clause": bool(clause), **v}, ensure_ascii=False) + "\n")
                out.flush()
            return
        except Exception as e:
            if a == 3:
                with lock:
                    out.write(json.dumps({"id": q["id"], "verdict": "ERROR",
                        "issues": [str(e)[:100]]}, ensure_ascii=False) + "\n")
                    out.flush()
            time.sleep(5 * (a + 1))

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
    list(ex.map(check, rows))
out.close()
res = [json.loads(l) for l in open(OUT, encoding="utf-8")]
from collections import Counter
print("PRECHECK_DONE", Counter(r["verdict"] for r in res), flush=True)
for r in res:
    if r["verdict"] == "SUSPECT":
        print("SUSPECT", r["id"], "; ".join(r.get("issues", []))[:120], flush=True)
