# -*- coding: utf-8 -*-
"""W-F: re-evaluate the teacher (Qwen3-235B) under unified output-format constraints.

v2: CONC lowered to 2 and exponential backoff added after the first run lost 72/252
items to relay 429 rate-limiting; repair mode re-asks only errored items.

Conditions:
  T1 replicate : original protocol verbatim (max_tokens=16) - reproducibility anchor
  T2 budget    : same prompt, max_tokens=2048, letter parsed from output tail
  T3 strict    : reinforced single-letter instruction, max_tokens=16

Same endpoint/token/model as the original run; temperature=0.
Output: eval/benchmark/result_teacher_format.json (resumable; errors repaired in place)
"""
import json, re, os, time
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

BENCH = "/root/traffic-llm/eval/benchmark/test.jsonl"
OUT = "/root/traffic-llm/eval/benchmark/result_teacher_format.json"
MODEL = "Qwen3-235B-A22B-Instruct"
RELAY = "http://localhost:3000/v1"
CONC = 2

SYS = ("你是交通工程勘察设计领域专家。下面是一道单选题,请仔细判断后只回答正确选项的字母"
       "(A、B、C 或 D),不要输出任何解释。")
SYS_STRICT = ("你是交通工程勘察设计领域专家。下面是一道单选题。你必须只输出一个大写字母"
              "(A、B、C、D 之一),不输出思考过程、解释或任何其他字符。")

TOKEN = os.environ["RELAY_API_KEY"]
cli = OpenAI(base_url=RELAY, api_key=TOKEN, timeout=180)

rows = [json.loads(l) for l in open(BENCH, encoding="utf-8") if l.strip()]
LIMIT = int(os.environ.get("EVAL_LIMIT", "0"))
if LIMIT:
    rows = rows[:LIMIT]
gts = [r["ground_truth"] for r in rows]
print("items:", len(rows), flush=True)


def extract_first(t):
    m = re.search(r"[ABCD]", (t or "").upper())
    return m.group(0) if m else "?"


def extract_last(t):
    m = re.findall(r"[ABCD]", (t or "").upper())
    return m[-1] if m else "?"


CONDS = {
    "T1_replicate": dict(sys=SYS, max_tokens=16, ex=extract_first),
    "T2_budget":    dict(sys=SYS, max_tokens=2048, ex=extract_last),
    "T3_strict":    dict(sys=SYS_STRICT, max_tokens=16, ex=extract_first),
}


def ask(q, cfg):
    for a in range(8):
        try:
            r = cli.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": cfg["sys"]},
                          {"role": "user", "content": q + "\n\n答案(只写字母):"}],
                temperature=0, max_tokens=cfg["max_tokens"])
            c = r.choices[0]
            return cfg["ex"](c.message.content), (c.finish_reason or ""), len(c.message.content or "")
        except Exception as e:
            if a == 7:
                return "?", "error:" + str(e)[:40], 0
            time.sleep(10 * (a + 1) if "429" in str(e) else 3 * (a + 1))


res = {}
if os.path.exists(OUT):
    res = json.load(open(OUT, encoding="utf-8"))

for name, cfg in CONDS.items():
    prior = res.get(name)
    if prior and len(prior["pred"]) == len(rows) and not any(
            (f or "").startswith("error:") for f in prior["finish"]):
        print(name, "cached clean", flush=True)
        continue
    pred = list(prior["pred"]) if prior else [None] * len(rows)
    fin = list(prior["finish"]) if prior else [None] * len(rows)
    olen = list(prior["outlen"]) if prior else [0] * len(rows)
    todo_idx = [i for i in range(len(rows))
                if pred[i] is None or (fin[i] or "").startswith("error:")]
    print(name, "asking", len(todo_idx), "items", flush=True)

    def job(i):
        pred[i], fin[i], olen[i] = ask(rows[i]["question"], cfg)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=CONC) as ex:
        list(ex.map(job, todo_idx))
    correct = sum(p == g for p, g in zip(pred, gts))
    unans = sum(1 for p in pred if p == "?")
    valid = len(rows) - unans
    res[name] = {"pred": pred, "finish": fin, "outlen": olen,
                 "correct": correct, "unanswered": unans,
                 "acc_all": correct / len(rows),
                 "acc_valid": (correct / valid) if valid else 0.0}
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    print("%s: %d/%d = %.1f%% all | unanswered %d | %.1f%% valid | %.0fs"
          % (name, correct, len(rows), correct / len(rows) * 100, unans,
             res[name]["acc_valid"] * 100, time.time() - t0), flush=True)

print("done ->", OUT, flush=True)
