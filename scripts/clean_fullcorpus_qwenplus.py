import json, requests, re, random, time, os, threading, concurrent.futures
from collections import Counter
BASE="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
TOKEN = os.environ["RELAY_API_KEY"]  # set your own relay/API key
MODEL="qwen-plus"
CONC=40
DATA="/root/traffic-llm/data"
FILES={"p0_transgpt":"processed/filtered/p0_transgpt_sft_filtered.jsonl",
       "p0_belle":"processed/filtered/p0_belle_anchor_filtered.jsonl"}
OUT="/root/traffic-llm/data/processed/quality_v2_detail.jsonl"
def log(m): print(m,flush=True)
pairs=[json.loads(l) for l in open("/root/traffic-llm/audit/review_pairs.jsonl",encoding="utf-8")]
random.seed(1)
drops=[p for p in pairs if p["human"]=="drop"]; keeps=[p for p in pairs if p["human"]=="keep"]
random.shuffle(drops); random.shuffle(keeps)
fewshot=[(p,"FAIL") for p in drops[:4]]+[(p,"PASS") for p in keeps[:4]]
random.shuffle(fewshot)
SYS=("你是交通垂域训练数据质检员,审核一条问答质量,判 PASS/WARN/FAIL。"
"FAIL(明确该删,只给确凿的):1)答案缺前后文——引用'上述/该通知/如前/见上表/上文'却无对应上下文,或明显是长文档切片出来的孤立片段、脱离语境读不懂;2)明确专业错误/幻觉/编造;3)答非所问/与交通无关。"
"WARN(小瑕疵但可用,拿不准就给WARN别给FAIL)。PASS(问题自包含、答案专业准确切题)。"
'只输出JSON:{"verdict":"PASS或WARN或FAIL","reason":"≤15字"}')
def msgs(q,a):
    m=[{"role":"system","content":SYS}]
    for p,v in fewshot:
        m.append({"role":"user","content":"问:%s\n答:%s"%(p["user"][:170],p["assistant"][:240])})
        m.append({"role":"assistant","content":'{"verdict":"%s","reason":"示例"}'%v})
    m.append({"role":"user","content":"问:%s\n答:%s"%(q[:170],a[:240])})
    return m
def judge(q,a):
    for att in range(8):
        try:
            r=requests.post(BASE,headers={"Authorization":"Bearer "+TOKEN},
                json={"model":MODEL,"messages":msgs(q,a),"stream":False,"max_tokens":50,"temperature":0},timeout=90)
            if r.status_code==429: time.sleep((att+1)*6); continue
            if r.status_code!=200: time.sleep(3); continue
            t=r.json()["choices"][0]["message"]["content"]
            mm=re.search(r'"verdict"\s*:\s*"(PASS|WARN|FAIL)"',t,re.I)
            if mm: return mm.group(1).upper()
            for v in ("FAIL","WARN","PASS"):
                if v in t.upper(): return v
        except Exception: time.sleep(3)
    return "ERROR"
def qa(o):
    q=a=""
    for m in o.get("messages",[]):
        c=m.get("content","")
        if not isinstance(c,str): continue
        if m["role"]=="user": q=c
        elif m["role"]=="assistant": a=c
    return q,a
done=set()
if os.path.exists(OUT):
    for l in open(OUT,encoding="utf-8"):
        try:
            o=json.loads(l)
            if o.get("verdict") in ("PASS","WARN","FAIL"): done.add("%s:%s"%(o["phase"],o["idx"]))
        except: pass
tasks=[]
for ph,rel in FILES.items():
    recs=[json.loads(l) for l in open(os.path.join(DATA,rel),encoding="utf-8") if l.strip()]
    for i,rec in enumerate(recs):
        if "%s:%s"%(ph,i) in done: continue
        q,a=qa(rec); tasks.append((ph,i,q,a))
log("待判 %d 条(已完成 %d), few-shot %d, 并发 %d"%(len(tasks),len(done),len(fewshot),CONC))
lock=threading.Lock(); out=open(OUT,"a",encoding="utf-8"); cnt=Counter(); n=[0]
def work(t):
    ph,i,q,a=t; v=judge(q,a)
    with lock:
        out.write(json.dumps({"phase":ph,"idx":i,"verdict":v},ensure_ascii=False)+"\n"); out.flush()
        cnt[v]+=1; n[0]+=1
        if n[0]%50==0: log("  %d/%d verdict:%s"%(n[0],len(tasks),dict(cnt)))
with concurrent.futures.ThreadPoolExecutor(max_workers=CONC) as ex:
    list(ex.map(work,tasks))
out.close()
log("=== 完成 verdict总分布:%s ==="%dict(cnt))
log("CLEANSCAN_DONE")
