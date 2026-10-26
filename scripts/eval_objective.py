# -*- coding: utf-8 -*-
"""闭卷客观题评测: base / v1(dirty) / v2(clean) / teacher(235B) 在 252 道交通勘察设计单选题上的准确率。
本地模型用 4bit 量化 + peft 多 adapter 热切换; teacher 走集团 235B OpenAI 兼容 endpoint。
环境变量: EVAL_LIMIT=N 只跑前 N 题(调试); SKIP_TEACHER=1 跳过 235B。
"""
import json, re, os, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from collections import defaultdict

BENCH="/root/traffic-llm/eval/benchmark/test.jsonl"
BASE="/root/models/Qwen3-14B"
V1="/root/runs/qwen3-14b-traffic-v1"
V2="/root/runs/qwen3-14b-traffic-v2-clean"
OUT="/root/traffic-llm/eval/benchmark/result_objective.json"
BATCH=16

rows=[json.loads(l) for l in open(BENCH,encoding="utf-8") if l.strip()]
LIMIT=int(os.environ.get("EVAL_LIMIT","0"))
if LIMIT: rows=rows[:LIMIT]
gts=[r["ground_truth"] for r in rows]
cats=[r["category"] for r in rows]
print(f"题数: {len(rows)}", flush=True)

tok=AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
tok.padding_side="left"
if tok.pad_token is None: tok.pad_token=tok.eos_token

SYS="你是交通工程勘察设计领域专家。下面是一道单选题,请仔细判断后只回答正确选项的字母(A、B、C 或 D),不要输出任何解释。"
def build(q):
    msg=[{"role":"system","content":SYS},{"role":"user","content":q+"\n\n答案(只写字母):"}]
    try:
        return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
prompts=[build(r["question"]) for r in rows]

def extract(t):
    m=re.search(r'[ABCD]', t.upper())
    return m.group(0) if m else "?"

def gen(model):
    preds=[]
    for i in range(0,len(prompts),BATCH):
        b=prompts[i:i+BATCH]
        inp=tok(b,return_tensors="pt",padding=True).to(model.device)
        with torch.no_grad():
            out=model.generate(**inp,max_new_tokens=8,do_sample=False,pad_token_id=tok.pad_token_id)
        for j in range(len(b)):
            txt=tok.decode(out[j][inp.input_ids.shape[1]:],skip_special_tokens=True)
            preds.append(extract(txt))
        print(f"  {min(i+BATCH,len(prompts))}/{len(prompts)}",flush=True)
    return preds

results={}
print("=== 加载 base (4bit) ===",flush=True)
bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16)
base=AutoModelForCausalLM.from_pretrained(BASE,quantization_config=bnb,device_map="cuda",torch_dtype=torch.bfloat16,trust_remote_code=True)
base.eval()

print("=== 挂载 v1/v2 adapter ===",flush=True)
m=PeftModel.from_pretrained(base,V1,adapter_name="v1")
m.load_adapter(V2,adapter_name="v2")
m.eval()

print("=== base 推理 ===",flush=True)
with m.disable_adapter():
    results["base"]=gen(m)
print("=== v1(dirty) 推理 ===",flush=True)
m.set_adapter("v1"); results["v1"]=gen(m)
print("=== v2(clean) 推理 ===",flush=True)
m.set_adapter("v2"); results["v2"]=gen(m)

def save_results(tag=""):
    summary={}
    for name,preds in results.items():
        correct=sum(p==g for p,g in zip(preds,gts))
        bycat=defaultdict(lambda:[0,0])
        for p,g,c in zip(preds,gts,cats):
            bycat[c][1]+=1; bycat[c][0]+=int(p==g)
        summary[name]={"acc":round(correct/len(gts),4),"correct":correct,"total":len(gts),
                       "bycat":{c:round(v[0]/v[1],3) for c,v in bycat.items()}}
    json.dump({"summary":summary,"preds":results,"gts":gts,"cats":cats},open(OUT,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    print(f"\n=== 准确率{tag} ===",flush=True)
    for name in ["base","v1","v2","teacher"]:
        if name in results:
            s=summary[name]; print(f"  {name:8s}: {s['acc']*100:5.1f}%  ({s['correct']}/{s['total']})",flush=True)
    return summary

# 本地 3 系统结果先落盘(防 teacher 再出问题丢结果)
save_results(" [本地, teacher前]")

# ---- teacher: 走中转池 localhost:3000 (Qwen3-235B-A22B-Instruct) ----
if os.environ.get("SKIP_TEACHER"):
    results["teacher"]=["?"]*len(rows)
else:
    print("=== teacher 235B (中转池 localhost:3000) ===",flush=True)
    try:
        from openai import OpenAI
        import re
        TOKEN=re.search(r'TOKEN="(sk-[^"]+)"',open("/root/traffic-llm/clean_v2_judge.py").read()).group(1)
        cli=OpenAI(base_url="http://localhost:3000/v1",api_key=TOKEN)
        TMODEL="Qwen3-235B-A22B-Instruct"
        tp=[]
        for idx,r in enumerate(rows):
            got="?"
            for attempt in range(3):
                try:
                    resp=cli.chat.completions.create(model=TMODEL,
                        messages=[{"role":"system","content":SYS},{"role":"user","content":r["question"]+"\n\n答案(只写字母):"}],
                        temperature=0,max_tokens=16)
                    got=extract(resp.choices[0].message.content or ""); break
                except Exception as e:
                    if attempt==2: print("  235B err",idx,str(e)[:70],flush=True)
                    else: time.sleep(1.5*(attempt+1))
            tp.append(got)
            if idx%25==0: print(f"  teacher {idx}/{len(rows)}",flush=True)
        results["teacher"]=tp
    except Exception as e:
        print("teacher整体失败:",e,flush=True); results["teacher"]=["?"]*len(rows)

save_results(" [最终]")
print("结果已存:",OUT,flush=True)
