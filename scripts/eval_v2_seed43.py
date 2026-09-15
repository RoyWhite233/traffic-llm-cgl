# -*- coding: utf-8 -*-
"""v1-prime (等预算未治理对照) 闭卷评测 — 协议与 eval_oneclick.py 逐位一致。
加载 base 4bit + v1prime adapter, 252 题, greedy, max_new_tokens=8。
输出 result_v2_seed43.json 并与 result_oneclick.json 中 v2/base/v1 做 McNemar 精确检验。
运行: /root/miniconda3/envs/llm/bin/python /root/traffic-llm/eval/eval_v1prime.py
"""
import json, re, os
from collections import defaultdict
from math import comb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BENCH="/root/traffic-llm/eval/benchmark/test.jsonl"
BASE="/root/models/Qwen3-14B"
V1P="/root/runs/qwen3-14b-v2-seed43"
REF="/root/traffic-llm/eval/benchmark/result_oneclick.json"
OUT="/root/traffic-llm/eval/benchmark/result_v2_seed43.json"
BATCH=16

rows=[json.loads(l) for l in open(BENCH,encoding="utf-8") if l.strip()]
gts=[r["ground_truth"] for r in rows]
cats=[r["category"] for r in rows]
print(f"[准备] 题目数: {len(rows)}", flush=True)

tok=AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
tok.padding_side="left"
if tok.pad_token is None: tok.pad_token=tok.eos_token
SYS="你是交通工程勘察设计领域专家。下面是一道单选题,请仔细判断后只回答正确选项的字母(A、B、C 或 D),不要输出任何解释。"
def build(q):
    m=[{"role":"system","content":SYS},{"role":"user","content":q+"\n\n答案(只写字母):"}]
    try: return tok.apply_chat_template(m,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    except TypeError: return tok.apply_chat_template(m,tokenize=False,add_generation_prompt=True)
prompts=[build(r["question"]) for r in rows]
def extract(t):
    m=re.search(r"[ABCD]", (t or "").upper()); return m.group(0) if m else "?"

bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16)
base=AutoModelForCausalLM.from_pretrained(BASE,quantization_config=bnb,device_map="cuda",trust_remote_code=True)
base.eval()
m=PeftModel.from_pretrained(base,V1P,adapter_name="v2_seed43"); m.set_adapter("v2_seed43"); m.eval()

preds=[]
for i in range(0,len(prompts),BATCH):
    b=prompts[i:i+BATCH]
    inp=tok(b,return_tensors="pt",padding=True).to(m.device)
    with torch.no_grad():
        out=m.generate(**inp,max_new_tokens=8,do_sample=False,pad_token_id=tok.pad_token_id)
    for j in range(len(b)):
        preds.append(extract(tok.decode(out[j][inp.input_ids.shape[1]:],skip_special_tokens=True)))
    print(f"  [v2seed43] {min(i+BATCH,len(prompts))}/{len(prompts)}",flush=True)

correct=sum(p==g for p,g in zip(preds,gts))
unans=sum(1 for p in preds if p=="?")
bycat=defaultdict(lambda:[0,0])
for p,g,c in zip(preds,gts,cats): bycat[c][1]+=1; bycat[c][0]+=int(p==g)

ref=json.load(open(REF,encoding="utf-8"))
assert ref["gts"]==gts, "ground truth 顺序与 result_oneclick.json 不一致!"

def mcnemar(pa,pb):
    b=sum(1 for x,y,g in zip(pa,pb,gts) if x==g and y!=g)
    c=sum(1 for x,y,g in zip(pa,pb,gts) if x!=g and y==g)
    n=b+c
    p=min(1.0, 2*sum(comb(n,k) for k in range(min(b,c)+1))*0.5**n) if n else 1.0
    return b,c,p

def wilson(k,n,z=1.959964):
    ph=k/n; d=1+z*z/n
    ctr=(ph+z*z/(2*n))/d; half=z*((ph*(1-ph)/n+z*z/(4*n*n))**0.5)/d
    return ctr-half, ctr+half

lo,hi=wilson(correct,len(gts))
print("\n"+"="*60, flush=True)
print(f"v1-prime 闭卷: {correct}/{len(gts)} = {correct/len(gts)*100:.1f}%  未答 {unans}  Wilson95CI [{lo*100:.1f}, {hi*100:.1f}]", flush=True)
stats={}
for name in ["v2","base","v1"]:
    b,c,p=mcnemar(ref["preds"][name],preds)
    refc=sum(x==g for x,g in zip(ref["preds"][name],gts))
    print(f"McNemar {name}({refc}/252) vs v1prime: b={b} c={c} p={p:.4g}", flush=True)
    stats[name]={"b":b,"c":c,"p":p,"ref_correct":refc}

json.dump({"summary":{"correct":correct,"total":len(gts),"acc":correct/len(gts),
    "unanswered":unans,"wilson95":[lo,hi],
    "bycat":{c:(v[0],v[1]) for c,v in bycat.items()}},
    "mcnemar_vs":stats,"preds":{"v2_seed43":preds},"gts":gts,"cats":cats},
    open(OUT,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(f"结果文件: {OUT}", flush=True)

# extra pairing: v1prime (matched-budget control) vs this seed-43 run
try:
    v1p = json.load(open("/root/traffic-llm/eval/benchmark/result_v1prime.json", encoding="utf-8"))["preds"]["v1prime"]
    b, c, p = mcnemar(v1p, preds)
    print(f"McNemar v1prime(166/252) vs v2seed43: b={b} c={c} p={p:.4g}", flush=True)
except Exception as e:
    print("v1prime pairing skipped:", e, flush=True)
