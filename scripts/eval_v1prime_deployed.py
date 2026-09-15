# -*- coding: utf-8 -*-
"""v1-prime 部署形态(vLLM 合并权重)闭卷 252 题, 与适配器形态预测比对。协议同 eval_openbook.py closed。"""
import json, re
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
BENCH="/root/traffic-llm/eval/benchmark/test.jsonl"
ADP="/root/traffic-llm/eval/benchmark/result_v1prime.json"
OUT="/root/traffic-llm/eval/benchmark/result_v1prime_deployed.json"
cli=OpenAI(base_url="http://localhost:8001/v1", api_key="x"); MODEL="v1prime-14b"
SYS="你是交通工程勘察设计领域专家。下面是一道单选题,请仔细判断后只回答正确选项的字母(A、B、C 或 D),不要输出任何解释。"
rows=[json.loads(l) for l in open(BENCH,encoding="utf-8") if l.strip()]
gts=[r["ground_truth"] for r in rows]
adp=json.load(open(ADP,encoding="utf-8"))["preds"]["v1prime"]
def extract(t):
    m=re.search(r"[ABCD]",(t or "").upper()); return m.group(0) if m else "?"
def ask(i):
    q=rows[i]["question"]+"\n\n答案(只写字母):"
    for a in range(3):
        try:
            r=cli.chat.completions.create(model=MODEL,messages=[{"role":"system","content":SYS},{"role":"user","content":q}],temperature=0,max_tokens=24,extra_body={"chat_template_kwargs":{"enable_thinking":False}})
            return extract(r.choices[0].message.content)
        except Exception as e:
            if a==2: print("err",i,str(e)[:60],flush=True)
    return "?"
with ThreadPoolExecutor(max_workers=4) as ex:
    preds=list(ex.map(ask,range(len(rows))))
correct=sum(p==g for p,g in zip(preds,gts))
diff=sum(1 for a,b in zip(adp,preds) if a!=b)
print(f"v1prime 部署形态闭卷: {correct}/252 = {correct/252*100:.1f}%  (适配器形态 166/252=65.9%)  两形态逐题差异 {diff} 题",flush=True)
json.dump({"preds":preds,"correct":correct,"diff_vs_adapter":diff},open(OUT,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
print("saved",OUT,flush=True)
