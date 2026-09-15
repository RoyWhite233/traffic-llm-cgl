# -*- coding: utf-8 -*-
"""人工校准消融实验:带校准few-shot vs 无校准, 以301条人工裁决为标尺。
- 精确复现生产judge的few-shot选取(seed=1), 评测集剔除这8条(防泄漏)+1条edit
- 两组同模型(DeepSeek-V3-1-w8a8)/同rubric/temp=0, 唯一差异=有无few-shot
- 输出: 逐条明细jsonl + 混淆矩阵 + 删除决策precision/recall(两种删除口径)
"""
import json, requests, re, random, time, os, threading, concurrent.futures
from collections import Counter

BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
TOKEN = os.environ["RELAY_API_KEY"]
MODEL = "qwen-plus"
CONC = 20
OUT = "/root/traffic-llm/audit/ablation_calib_qwenplus.jsonl"

pairs = [json.loads(l) for l in open("/root/traffic-llm/audit/review_pairs.jsonl", encoding="utf-8")]

# ---- 精确复现生产few-shot选取(与clean_v2_judge.py逐行一致) ----
random.seed(1)
drops = [p for p in pairs if p["human"] == "drop"]; keeps = [p for p in pairs if p["human"] == "keep"]
random.shuffle(drops); random.shuffle(keeps)
fewshot = [(p, "FAIL") for p in drops[:4]] + [(p, "PASS") for p in keeps[:4]]
random.shuffle(fewshot)
fewshot_rids = {p["rid"] for p, _ in fewshot}
print("few-shot rids:", sorted(fewshot_rids), flush=True)

# ---- 评测集: keep/drop 且不在few-shot里 ----
evalset = [p for p in pairs if p["human"] in ("keep", "drop") and p["rid"] not in fewshot_rids]
print("评测集: %d 条 (keep %d / drop %d)" % (
    len(evalset), sum(1 for p in evalset if p["human"] == "keep"),
    sum(1 for p in evalset if p["human"] == "drop")), flush=True)

SYS = ("你是交通垂域训练数据质检员,审核一条问答质量,判 PASS/WARN/FAIL。"
"FAIL(明确该删,只给确凿的):1)答案缺前后文——引用'上述/该通知/如前/见上表/上文'却无对应上下文,或明显是长文档切片出来的孤立片段、脱离语境读不懂;2)明确专业错误/幻觉/编造;3)答非所问/与交通无关。"
"WARN(小瑕疵但可用,拿不准就给WARN别给FAIL)。PASS(问题自包含、答案专业准确切题)。"
'只输出JSON:{"verdict":"PASS或WARN或FAIL","reason":"≤15字"}')

def msgs(q, a, with_fs):
    m = [{"role": "system", "content": SYS}]
    if with_fs:
        for p, v in fewshot:
            m.append({"role": "user", "content": "问:%s\n答:%s" % (p["user"][:170], p["assistant"][:240])})
            m.append({"role": "assistant", "content": '{"verdict":"%s","reason":"示例"}' % v})
    m.append({"role": "user", "content": "问:%s\n答:%s" % (q[:170], a[:240])})
    return m

def judge(q, a, with_fs):
    for att in range(8):
        try:
            r = requests.post(BASE, headers={"Authorization": "Bearer " + TOKEN},
                json={"model": MODEL, "messages": msgs(q, a, with_fs), "stream": False,
                      "max_tokens": 50, "temperature": 0}, timeout=90)
            if r.status_code == 429: time.sleep((att + 1) * 6); continue
            if r.status_code != 200: time.sleep(3); continue
            t = r.json()["choices"][0]["message"]["content"]
            mm = re.search(r'"verdict"\s*:\s*"(PASS|WARN|FAIL)"', t, re.I)
            if mm: return mm.group(1).upper()
            for v in ("FAIL", "WARN", "PASS"):
                if v in t.upper(): return v
        except Exception: time.sleep(3)
    return "ERROR"

# ---- 断点续 ----
done = {}
if os.path.exists(OUT):
    for l in open(OUT, encoding="utf-8"):
        try:
            o = json.loads(l)
            if o["cal"] in ("PASS","WARN","FAIL") and o["nocal"] in ("PASS","WARN","FAIL"):
                done[o["rid"]] = o
        except: pass
tasks = [p for p in evalset if p["rid"] not in done]
print("待跑 %d 条(已完成 %d)" % (len(tasks), len(done)), flush=True)

lock = threading.Lock(); out = open(OUT, "a", encoding="utf-8"); n = [0]
def work(p):
    vc = judge(p["user"], p["assistant"], True)
    vn = judge(p["user"], p["assistant"], False)
    with lock:
        out.write(json.dumps({"rid": p["rid"], "stratum": p["stratum"], "human": p["human"],
                              "machine_235b": p.get("machine_verdict"), "cal": vc, "nocal": vn},
                             ensure_ascii=False) + "\n"); out.flush()
        n[0] += 1
        if n[0] % 25 == 0: print("  %d/%d" % (n[0], len(tasks)), flush=True)

with concurrent.futures.ThreadPoolExecutor(max_workers=CONC) as ex:
    list(ex.map(work, tasks))
out.close()

# ---- 汇总 ----
rows = []
seen = set()
for l in open(OUT, encoding="utf-8"):
    o = json.loads(l)
    if o["rid"] in seen: continue
    seen.add(o["rid"]); rows.append(o)
rows = [r for r in rows if r["cal"] != "ERROR" and r["nocal"] != "ERROR"]
print("\n===== 汇总 (有效 %d 条) =====" % len(rows), flush=True)

def report(tag, key):
    print("\n--- %s ---" % tag, flush=True)
    cm = Counter((r[key], r["human"]) for r in rows)
    print("混淆矩阵 verdict×human:", flush=True)
    for v in ("PASS", "WARN", "FAIL"):
        print("  %s: keep=%d drop=%d" % (v, cm[(v, "keep")], cm[(v, "drop")]), flush=True)
    for th_name, is_del in (("删除口径=FAIL", lambda v: v == "FAIL"),
                            ("删除口径=FAIL+WARN", lambda v: v != "PASS")):
        tp = sum(1 for r in rows if is_del(r[key]) and r["human"] == "drop")
        fp = sum(1 for r in rows if is_del(r[key]) and r["human"] == "keep")
        fn = sum(1 for r in rows if not is_del(r[key]) and r["human"] == "drop")
        tn = sum(1 for r in rows if not is_del(r[key]) and r["human"] == "keep")
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        acc = (tp + tn) / len(rows)
        print("  [%s] P=%.3f R=%.3f F1=%.3f Acc=%.3f (tp%d fp%d fn%d tn%d)" % (
            th_name, prec, rec, f1, acc, tp, fp, fn, tn), flush=True)

report("A组 带人工校准few-shot", "cal")
report("B组 无校准(仅rubric)", "nocal")
print("\n明细: %s" % OUT, flush=True)
print("ABLATION_DONE", flush=True)
