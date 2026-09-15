# -*- coding: utf-8 -*-
"""窗口期复核: v2 部署形态闭卷 252 题, 与 result_openbook.json 的 closed 预测逐题比对。
协议与 eval_openbook.py 完全一致 (temperature=0, max_tokens=24, enable_thinking=False)。
"""
import json, re
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

BENCH="/root/traffic-llm/eval/benchmark/test.jsonl"
REF="/root/traffic-llm/eval/benchmark/result_openbook.json"
OUT="/root/traffic-llm/eval/benchmark/result_closed_recheck_0903.json"
cli=OpenAI(base_url="http://localhost:8001/v1", api_key="x")
MODEL="hbjgy-14b"
SYS="你是交通工程勘察设计领域专家。下面是一道单选题,请仔细判断后只回答正确选项的字母(A、B、C 或 D),不要输出任何解释。"

rows=[json.loads(l) for l in open(BENCH,encoding="utf-8") if l.strip()]
gts=[r["ground_truth"] for r in rows]
ref=json.load(open(REF,encoding="utf-8"))
assert ref["gts"]==gts

def extract(t):
    m=re.search(r"[ABCD]",(t or "").upper()); return m.group(0) if m else "?"

def ask(i):
    q=rows[i]["question"]+"\n\n答案(只写字母):"
    for a in range(3):
        try:
            r=cli.chat.completions.create(model=MODEL,
                messages=[{"role":"system","content":SYS},{"role":"user","content":q}],
                temperature=0,max_tokens=24,
                extra_body={"chat_template_kwargs":{"enable_thinking":False}})
            return extract(r.choices[0].message.content)
        except Exception as e:
            if a==2: print("err",i,str(e)[:60],flush=True)
    return "?"

with ThreadPoolExecutor(max_workers=4) as ex:
    preds=list(ex.map(ask,range(len(rows))))

correct=sum(p==g for p,g in zip(preds,gts))
old=ref["closed"]
oldc=sum(p==g for p,g in zip(old,gts))
diff=[(rows[i]["id"],old[i],preds[i]) for i in range(len(rows)) if old[i]!=preds[i]]
old_err={rows[i]["id"] for i in range(len(rows)) if old[i]!=gts[i]}
new_err={rows[i]["id"] for i in range(len(rows)) if preds[i]!=gts[i]}
print(f"复核闭卷: {correct}/252 = {correct/252*100:.1f}%  (原 {oldc}/252 = {oldc/252*100:.1f}%)",flush=True)
print(f"逐题不一致: {len(diff)} 题  错题集: 原{len(old_err)} 新{len(new_err)} 交集{len(old_err&new_err)}",flush=True)
for d in diff[:20]: print("  drift:",d,flush=True)
json.dump({"preds":preds,"correct":correct,"drift":diff,
    "err_old":sorted(old_err),"err_new":sorted(new_err)},open(OUT,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
print("saved",OUT,flush=True)
