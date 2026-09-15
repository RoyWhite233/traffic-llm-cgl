# -*- coding: utf-8 -*-
"""Second-pass triage of precheck SUSPECT items into actionable buckets."""
import json, re, os, time, threading, concurrent.futures
import requests

IN = "/root/traffic-llm/eval/benchmark/precheck_252.jsonl"
OUT = "/root/traffic-llm/eval/benchmark/precheck_triage.jsonl"
BASE = "http://localhost:3000/v1/chat/completions"
MODEL = "DeepSeek-V3-1-w8a8"
TOKEN = os.environ["RELAY_API_KEY"]

SYS = ("你是审核分诊员。给你一条基准题目预核发现的问题描述,请判断问题的主要归属,只输出JSON:"
       '{"bucket":"A|B|C|D|E","why":"≤20字"}。'
       "A=有具体理由怀疑标准答案本身错误(答案与规范实际规定冲突);"
       "B=提供给核查的恢复条文与题目引用不匹配/不完整,导致无法验证(条文恢复质量问题,不代表答案错);"
       "C=题目自身的出处编号/规范版本/表号标注疑似有误(与恢复无关的题面引用问题);"
       "D=题目缺少必要适用前提;"
       "E=解析不完整等次要问题。多类并存时选最主要且最严重的一类(A>C>D>B>E)。")

rows = [json.loads(l) for l in open(IN, encoding="utf-8")]
sus = [r for r in rows if r["verdict"] == "SUSPECT"]
done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["id"] for l in open(OUT, encoding="utf-8") if l.strip()}
lock = threading.Lock(); out = open(OUT, "a", encoding="utf-8")

def go(r):
    if r["id"] in done: return
    body = "题目ID:%s\n问题描述:%s" % (r["id"], "; ".join(r.get("issues", []))[:800])
    for a in range(4):
        try:
            resp = requests.post(BASE, headers={"Authorization": "Bearer " + TOKEN},
                json={"model": MODEL, "messages": [{"role": "system", "content": SYS},
                      {"role": "user", "content": body}], "temperature": 0, "max_tokens": 80,
                      "stream": False}, timeout=90)
            t = resp.json()["choices"][0]["message"]["content"]
            m = re.search(r'"bucket"\s*:\s*"([A-E])"', t)
            v = m.group(1) if m else "PARSE"
            with lock:
                out.write(json.dumps({"id": r["id"], "bucket": v,
                    "issues": r.get("issues", [])}, ensure_ascii=False) + "\n"); out.flush()
            return
        except Exception:
            time.sleep(5 * (a + 1))
    with lock:
        out.write(json.dumps({"id": r["id"], "bucket": "ERROR"}, ensure_ascii=False) + "\n"); out.flush()

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
    list(ex.map(go, sus))
out.close()
from collections import Counter
res = [json.loads(l) for l in open(OUT, encoding="utf-8")]
print("TRIAGE_DONE", Counter(x["bucket"] for x in res), flush=True)
for x in res:
    if x["bucket"] in ("A", "C"):
        print(x["bucket"], x["id"], "; ".join(x.get("issues", []))[:110], flush=True)
