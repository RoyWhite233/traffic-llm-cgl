"""分层采样 LLM-judge 审计: 对6个clean数据集各抽样, 用235B裁判按分层rubric打分,
产出 per-stratum 错误率 + 标红清单(供专家可视化系统看/改/打勾).
复用 audit_faithfulness_full.py 的 endpoint 流式调用。断点续跑: detail逐条append。
输出: audit/sample_audit_detail.jsonl / sample_audit_stats.json / sample_audit_flagged.jsonl
"""
import os, json, re, asyncio, random
from collections import Counter, defaultdict
import aiohttp

QWEN_BASE = os.getenv("QWEN3_235B_BASE_URL", "")
QWEN_MODEL = os.getenv("QWEN3_235B_MODEL", "Qwen3-235B-A22B-Instruct")
QWEN_KEY = os.getenv("QWEN3_235B_API_KEY", "EMPTY")
CONC = int(os.getenv("AUDIT_CONC", "10"))
SEED = int(os.getenv("AUDIT_SEED", "20260626"))
DATA = "data"
OUT_DIR = "audit"
os.makedirs(OUT_DIR, exist_ok=True)
DETAIL = f"{OUT_DIR}/sample_audit_detail.jsonl"
STATS = f"{OUT_DIR}/sample_audit_stats.json"
FLAGGED = f"{OUT_DIR}/sample_audit_flagged.jsonl"

# stratum -> (相对data的文件, rubric, 抽样数)
STRATA = {
    "p0_transgpt": ("processed/filtered/p0_transgpt_sft_filtered.jsonl", "quality", int(os.getenv("N_TRANSGPT", "150"))),
    "p1":          ("processed/desensitized/filtered/p1_rag_aware_clean_filtered.jsonl", "faith", int(os.getenv("N_P1", "100"))),
    "p0_ragflow":  ("processed/desensitized/filtered/p0_direct_qa_clean_filtered.jsonl", "faith", int(os.getenv("N_RAGFLOW", "60"))),
    "p0_belle":    ("processed/filtered/p0_belle_anchor_filtered.jsonl", "quality", int(os.getenv("N_BELLE", "60"))),
    "p2":          ("processed/desensitized/filtered/p2_hard_cases_clean_filtered.jsonl", "faith", int(os.getenv("N_P2", "150"))),
    "p3":          ("processed/desensitized/filtered/p3_refusal_clean_filtered.jsonl", "refusal", int(os.getenv("N_P3", "200"))),
    "book_qa":     ("processed/book_qa_15647_clean.jsonl", "quality", int(os.getenv("N_BOOK", "150"))),
}

RUBRICS = {
    "faith": ("你是严格的RAG数据质检员。只依据【参考资料】判断【回答】中的事实论断是否有据,不要用自己知识补充。回答中出现“资料未提及/未找到”属诚实,不扣分。",
              "【问题与参考资料】\n{ctx}\n\n【回答】\n{ans}\n\n只输出JSON:{{\"verdict\":\"PASS|WARN|FAIL\",\"reason\":\"一句话理由\"}}。PASS=论断基本有据;WARN=个别无据延伸;FAIL=大量编造/引用错配/与资料矛盾。"),
    "quality": ("你是交通垂域资深专家,审核问答训练数据质量。",
                "【问题】\n{ctx}\n\n【回答】\n{ans}\n\n从专业正确性/相关性/有无幻觉判断。只输出JSON:{{\"verdict\":\"PASS|WARN|FAIL\",\"reason\":\"一句话理由\"}}。PASS=专业准确切题;WARN=小瑕疵但可用;FAIL=明显错误/跑题/幻觉/答非所问。"),
    "refusal": ("你审核“拒答/越界处理”训练数据是否正确。",
                "【问题与上下文】\n{ctx}\n\n【回答】\n{ans}\n\n该样本意在训练模型在资料不匹配/超域时正确拒答或说明局限。只输出JSON:{{\"verdict\":\"PASS|WARN|FAIL\",\"reason\":\"一句话理由\"}}。PASS=正确拒答/指出局限;WARN=拒答但啰嗦或不够明确;FAIL=本该拒答却强答/编造。"),
}


def ctx_ans(o):
    sys = usr = ans = ""
    for m in o.get("messages", []):
        c = m.get("content", "")
        if not isinstance(c, str):
            continue
        r = m.get("role")
        if r == "system":
            sys = c
        elif r == "user":
            usr = c
        elif r == "assistant":
            ans = c
    return (sys + "\n\n" + usr).strip(), ans


async def call_judge(session, sysp, userp):
    payload = {"model": QWEN_MODEL,
               "messages": [{"role": "system", "content": sysp}, {"role": "user", "content": userp}],
               "temperature": 0.0, "max_tokens": 300, "stream": True}
    headers = {"Content-Type": "application/json"}
    if QWEN_KEY and QWEN_KEY != "EMPTY":
        headers["Authorization"] = "Bearer " + QWEN_KEY
    for attempt in range(6):
        try:
            async with session.post(QWEN_BASE + "/chat/completions", json=payload, headers=headers,
                                    timeout=aiohttp.ClientTimeout(total=None, sock_connect=15, sock_read=90)) as r:
                if r.status == 429:
                    raise RuntimeError("429")
                if r.status != 200:
                    raise RuntimeError("HTTP " + str(r.status))
                parts = []
                async for raw in r.content:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    d = line[5:].strip()
                    if d == "[DONE]":
                        break
                    try:
                        delta = json.loads(d)["choices"][0]["delta"].get("content", "")
                    except Exception:
                        continue
                    if delta:
                        parts.append(delta)
                t = "".join(parts)
                if not t:
                    raise RuntimeError("empty")
                return t
        except Exception as e:
            if attempt == 5:
                return '{"verdict":"ERROR","reason":"' + str(e)[:50] + '"}'
            await asyncio.sleep((attempt + 1) * 5 if "429" in str(e) else 2 ** attempt)


def parse(s):
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return {"verdict": "PARSE_FAIL", "reason": s[:80]}
    try:
        o = json.loads(m.group(0))
        v = str(o.get("verdict", "")).upper()
        return {"verdict": v if v in ("PASS", "WARN", "FAIL") else "PARSE_FAIL",
                "reason": str(o.get("reason", ""))[:160]}
    except Exception:
        return {"verdict": "PARSE_FAIL", "reason": s[:80]}


def load_done():
    done = set()
    if os.path.exists(DETAIL):
        for line in open(DETAIL, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if o.get("verdict") in ("PASS", "WARN", "FAIL"):
                done.add(f"{o['stratum']}:{o['idx']}")
    return done


def load_strata_recs():
    recs = {}
    for st, (fn, rub, n) in STRATA.items():
        p = os.path.join(DATA, fn)
        recs[st] = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []
        if not recs[st]:
            print(f"WARN 缺数据 {p}", flush=True)
    return recs


def sample_picks(recs):
    rng = random.Random(SEED)
    picked = []
    for st, (fn, rub, n) in STRATA.items():
        idxs = list(range(len(recs[st])))
        rng.shuffle(idxs)
        for i in idxs[:min(n, len(idxs))]:
            picked.append((st, i, rub))
    return picked


async def main():
    if not QWEN_BASE:
        raise SystemExit("QWEN3_235B_BASE_URL 未设置, 先 . ./.env")
    recs = load_strata_recs()
    picked = sample_picks(recs)
    done = load_done()
    tasks = [t for t in picked if f"{t[0]}:{t[1]}" not in done]
    print(f"抽样 {len(picked)} 条, 已判 {len(done)}, 待判 {len(tasks)}, 并发 {CONC}", flush=True)

    sem = asyncio.Semaphore(CONC)
    lock = asyncio.Lock()
    out = open(DETAIL, "a", encoding="utf-8")
    cnt = {"n": 0}

    async with aiohttp.ClientSession() as session:
        async def run(t):
            st, i, rub = t
            ctx, ans = ctx_ans(recs[st][i])
            sysp, tmpl = RUBRICS[rub]
            async with sem:
                raw = await call_judge(session, sysp, tmpl.format(ctx=ctx[:6000], ans=ans[:4000]))
            v = parse(raw)
            rid = recs[st][i].get("id", recs[st][i].get("_dsid", f"{st}-{i}"))
            row = {"stratum": st, "idx": i, "id": rid, "rubric": rub,
                   "verdict": v["verdict"], "reason": v["reason"]}
            async with lock:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                cnt["n"] += 1
                if cnt["n"] % 50 == 0:
                    print(f"  进度 {cnt['n']}/{len(tasks)}", flush=True)
        await asyncio.gather(*(run(t) for t in tasks))
    out.close()
    summarize(recs)


def summarize(recs):
    by = defaultdict(Counter)
    detail = {}
    for line in open(DETAIL, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        by[o["stratum"]][o["verdict"]] += 1
        detail[f"{o['stratum']}:{o['idx']}"] = o
    stats = {}
    for st in STRATA:
        c = by[st]
        tot = sum(c.values()) or 1
        stats[st] = {"sampled": sum(c.values()), "PASS": c.get("PASS", 0),
                     "WARN": c.get("WARN", 0), "FAIL": c.get("FAIL", 0),
                     "fail_rate": round(c.get("FAIL", 0) * 100 / tot, 1),
                     "flag_rate": round((c.get("FAIL", 0) + c.get("WARN", 0)) * 100 / tot, 1)}
    json.dump(stats, open(STATS, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    nflag = 0
    with open(FLAGGED, "w", encoding="utf-8") as fout:
        for key, o in detail.items():
            if o["verdict"] not in ("WARN", "FAIL"):
                continue
            st, i = o["stratum"], o["idx"]
            rec = recs[st][i] if i < len(recs[st]) else None
            fout.write(json.dumps({
                "id": o["id"], "stratum": st, "verdict": o["verdict"], "judge_reason": o["reason"],
                "messages": rec.get("messages") if rec else None,
                "meta": rec.get("meta") if rec else None,
                "review_status": "pending", "expert_verdict": None, "expert_note": "", "edited_messages": None,
            }, ensure_ascii=False) + "\n")
            nflag += 1
    print("=== 分层采样审计结果 (fail_rate=判FAIL占比, flag_rate=WARN+FAIL占比) ===", flush=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    print(f"标红(WARN+FAIL) {nflag} 条 -> {FLAGGED}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
