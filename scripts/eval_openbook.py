# -*- coding: utf-8 -*-
"""开卷 vs 闭卷 对照实验(只查已在跑的 vLLM,不加载新模型、不重启)。
- served-closed : 只给题目 -> 校验是否≈164(闭卷基准)
- served-open   : 把规范原文(source+explanation)当参考资料喂进去 -> 完美RAG上限
delta = RAG 头部空间。用于决定 提升方向(投RAG覆盖规范 vs 反复重训背表格)。
运行: /root/miniconda3/envs/llm/bin/python eval_openbook.py
"""
import json, re, os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

BENCH="/root/traffic-llm/eval/benchmark/test.jsonl"
OUT="/root/traffic-llm/eval/benchmark/result_openbook.json"
cli=OpenAI(base_url="http://localhost:8000/v1", api_key="x")
MODEL="hbjgy-14b"
SYS="你是交通工程勘察设计领域专家。下面是一道单选题,请仔细判断后只回答正确选项的字母(A、B、C 或 D),不要输出任何解释。"

rows=[json.loads(l) for l in open(BENCH,encoding="utf-8") if l.strip()]
LIMIT=int(os.environ.get("EVAL_LIMIT","0"))
if LIMIT: rows=rows[:LIMIT]
gts=[r["ground_truth"] for r in rows]; cats=[r["category"] for r in rows]
print(f"题目数: {len(rows)}", flush=True)

def extract(t):
    m=re.search(r'[ABCD]', (t or "").upper()); return m.group(0) if m else "?"

def ask(user):
    for a in range(3):
        try:
            r=cli.chat.completions.create(model=MODEL,
                messages=[{"role":"system","content":SYS},{"role":"user","content":user}],
                temperature=0, max_tokens=24,
                extra_body={"chat_template_kwargs":{"enable_thinking":False}})
            return extract(r.choices[0].message.content)
        except Exception as e:
            if a==2: return "?"

def closed(r): return r["question"]+"\n\n答案(只写字母):"
def openb(r):
    ref=f"{r.get('source','')}：{r.get('explanation','')}".strip("：")
    return f"参考资料(规范原文):\n{ref}\n\n题目:\n{r['question']}\n\n请依据上述参考资料判断,答案(只写字母):"

def run(builder, tag):
    preds=[None]*len(rows)
    def job(i): preds[i]=ask(builder(rows[i]))
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(job, range(len(rows))))
    correct=sum(p==g for p,g in zip(preds,gts))
    unans=sum(1 for p in preds if p=="?")
    print(f"[{tag}] 正确 {correct}/{len(rows)} = {correct/len(rows)*100:.1f}%  未答{unans}", flush=True)
    return preds, correct

pc, cc = run(closed, "闭卷 served")
po, co = run(openb,  "开卷 served")

# 分领域 + 每题闭->开变化
def bycat(preds):
    d=defaultdict(lambda:[0,0])
    for p,g,c in zip(preds,gts,cats): d[c][1]+=1; d[c][0]+=int(p==g)
    return {c:(v[0],v[1]) for c,v in d.items()}
json.dump({"closed":pc,"open":po,"gts":gts,"cats":cats,
           "summary":{"closed":cc,"open":co,"total":len(rows),
                      "bycat_closed":bycat(pc),"bycat_open":bycat(po)}},
          open(OUT,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

print("\n=== 分领域 闭卷% -> 开卷% ===", flush=True)
bc=bycat(pc); bo=bycat(po)
for c in sorted(bc, key=lambda c:-bc[c][1]):
    cc0,ct=bc[c]; co0,_=bo[c]
    print(f"  {c:<20} {cc0/ct*100:>5.0f}% -> {co0/ct*100:>5.0f}%  ({ct}题)", flush=True)
print(f"\n★ 闭卷 {cc}/{len(rows)}({cc/len(rows)*100:.1f}%) -> 开卷 {co}/{len(rows)}({co/len(rows)*100:.1f}%)  提升 {(co-cc)/len(rows)*100:+.1f}分", flush=True)
print(f"结果: {OUT}", flush=True)
