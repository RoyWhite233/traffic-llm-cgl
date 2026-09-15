# -*- coding: utf-8 -*-
"""一键闭卷客观评测: base / v1(dirty) / v2(clean) / teacher(235B)。
从头跑 252 道题, 最后打印清晰报告(人肉眼核对)。
运行:
  /root/miniconda3/envs/llm/bin/python /root/traffic-llm/eval/eval_oneclick.py
可选环境变量: EVAL_LIMIT=N 只跑前N题调试; SKIP_TEACHER=1 跳过235B。
"""
import json, re, os, time, sys
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BENCH="/root/traffic-llm/eval/benchmark/test.jsonl"
BASE="/root/models/Qwen3-14B"
V1="/root/runs/qwen3-14b-traffic-v1"
V2="/root/runs/qwen3-14b-traffic-v2-clean"
OUT="/root/traffic-llm/eval/benchmark/result_oneclick.json"
RELAY="http://localhost:3000/v1"
TEACHER_MODEL="Qwen3-235B-A22B-Instruct"
BATCH=16

def banner(t): print("\n"+"="*60+"\n"+t+"\n"+"="*60, flush=True)

rows=[json.loads(l) for l in open(BENCH,encoding="utf-8") if l.strip()]
LIMIT=int(os.environ.get("EVAL_LIMIT","0"))
if LIMIT: rows=rows[:LIMIT]
gts=[r["ground_truth"] for r in rows]
cats=[r["category"] for r in rows]
print(f"[准备] 测试集: {BENCH}", flush=True)
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
    m=re.search(r'[ABCD]', (t or "").upper()); return m.group(0) if m else "?"

def gen_local(model, tag):
    preds=[]
    for i in range(0,len(prompts),BATCH):
        b=prompts[i:i+BATCH]
        inp=tok(b,return_tensors="pt",padding=True).to(model.device)
        with torch.no_grad():
            out=model.generate(**inp,max_new_tokens=8,do_sample=False,pad_token_id=tok.pad_token_id)
        for j in range(len(b)):
            preds.append(extract(tok.decode(out[j][inp.input_ids.shape[1]:],skip_special_tokens=True)))
        print(f"  [{tag}] {min(i+BATCH,len(prompts))}/{len(prompts)}",flush=True)
    return preds

results={}
banner("阶段1/4: 加载 base (Qwen3-14B, 4bit) + 挂 v1/v2 adapter")
bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16)
base=AutoModelForCausalLM.from_pretrained(BASE,quantization_config=bnb,device_map="cuda",trust_remote_code=True)
base.eval()
m=PeftModel.from_pretrained(base,V1,adapter_name="v1"); m.load_adapter(V2,adapter_name="v2"); m.eval()

banner("阶段2/4: base(未微调) 推理")
with m.disable_adapter(): results["base"]=gen_local(m,"base")
banner("阶段3/4: v1(脏数据) / v2(干净) 推理")
m.set_adapter("v1"); results["v1"]=gen_local(m,"v1")
m.set_adapter("v2"); results["v2"]=gen_local(m,"v2")

banner("阶段4/4: teacher 235B (中转池)")
if os.environ.get("SKIP_TEACHER"):
    results["teacher"]=["?"]*len(rows); print("  已跳过", flush=True)
else:
    try:
        from openai import OpenAI
        TOKEN=os.environ["RELAY_API_KEY"]
        cli=OpenAI(base_url=RELAY,api_key=TOKEN)
        tp=[]
        for idx,r in enumerate(rows):
            got="?"
            for a in range(3):
                try:
                    resp=cli.chat.completions.create(model=TEACHER_MODEL,
                        messages=[{"role":"system","content":SYS},{"role":"user","content":r["question"]+"\n\n答案(只写字母):"}],
                        temperature=0,max_tokens=16)
                    got=extract(resp.choices[0].message.content); break
                except Exception as e:
                    if a==2: print("  235B err",idx,str(e)[:60],flush=True)
                    else: time.sleep(1.5*(a+1))
            tp.append(got)
            if idx%25==0: print(f"  [teacher] {idx}/{len(rows)}",flush=True)
        results["teacher"]=tp
    except Exception as e:
        print("teacher失败:",e,flush=True); results["teacher"]=["?"]*len(rows)

# ===== 统计 =====
summary={}
for name,preds in results.items():
    correct=sum(p==g for p,g in zip(preds,gts))
    unans=sum(1 for p in preds if p=="?")
    bycat=defaultdict(lambda:[0,0])
    for p,g,c in zip(preds,gts,cats): bycat[c][1]+=1; bycat[c][0]+=int(p==g)
    summary[name]={"correct":correct,"total":len(gts),"acc":correct/len(gts),"unanswered":unans,
                   "bycat":{c:(v[0],v[1]) for c,v in bycat.items()}}
json.dump({"summary":summary,"preds":results,"gts":gts,"cats":cats},open(OUT,"w",encoding="utf-8"),ensure_ascii=False,indent=2)

# ===== 打印报告(人肉眼核对) =====
order=["teacher","v2","base","v1"]
name_cn={"base":"base 未微调","v1":"v1 脏数据","v2":"v2 干净数据","teacher":"teacher 235B"}
banner("★★★ 评测结果(闭卷 单选 %d 题)★★★"%len(rows))
print(f"{'系统':<16}{'正确/总数':<14}{'准确率':<10}{'未答':<6}", flush=True)
print("-"*46, flush=True)
for n in order:
    s=summary[n]
    print(f"{name_cn[n]:<16}{str(s['correct'])+'/'+str(s['total']):<14}{s['acc']*100:>6.1f}%   {s['unanswered']:<6}", flush=True)
print("-"*46, flush=True)
print(f"关键对比:  v2 - base = {(summary['v2']['acc']-summary['base']['acc'])*100:+.1f} 分   "
      f"v2 - v1 = {(summary['v2']['acc']-summary['v1']['acc'])*100:+.1f} 分   "
      f"v1 - base = {(summary['v1']['acc']-summary['base']['acc'])*100:+.1f} 分", flush=True)

banner("分领域准确率 (%)")
allcats=sorted({c for c in cats}, key=lambda c:-summary['v2']['bycat'].get(c,(0,1))[1])
hdr=f"{'领域':<22}" + "".join(f"{name_cn[n].split()[0]:>10}" for n in order) + f"{'题数':>7}"
print(hdr, flush=True); print("-"*len(hdr), flush=True)
for c in allcats:
    tot=summary['v2']['bycat'].get(c,(0,0))[1]
    line=f"{c:<22}"
    for n in order:
        cc,ct=summary[n]['bycat'].get(c,(0,0))
        line+=f"{(cc/ct*100 if ct else 0):>9.0f}%"
    line+=f"{tot:>7}"
    print(line, flush=True)

print(f"\n结果文件: {OUT}", flush=True)
print("(以上数字直接由脚本从模型推理算出并打印, 你屏幕看到的即为准)", flush=True)
