# -*- coding: utf-8 -*-
"""Build the releasable version of the 301 expert-review records.

Policy (Data Availability, level B):
  - ALL 301 records ship with: rid, stratum, machine_verdict, judge_reason, human decision
    -> sufficient to recompute the 72.4% agreement figure and the Table 3 cross-tab
  - Records from open-source layers (p0_transgpt = TransGPT, p0_belle = BELLE) ALSO ship
    the sample text (user/assistant), since the source datasets are public
  - Records from internal layers (p0_ragflow, p1, p2, p3) have text withheld
    (text_available=false, withheld_reason)

Also emits a small stats block to verify the paper's Table 3 numbers from the release file.
"""
import json, collections, os

SRC = "/root/traffic-llm/audit/review_pairs.jsonl"
OUT = "/root/traffic-llm/audit/review_pairs_release.jsonl"
OPEN_LAYERS = {"p0_transgpt", "p0_belle"}

n = 0
comp = collections.Counter()
with open(OUT, "w", encoding="utf-8") as f:
    for l in open(SRC, encoding="utf-8"):
        d = json.loads(l)
        rec = {
            "rid": d["rid"],
            "stratum": d["stratum"],
            "machine_verdict": d.get("machine_verdict"),
            "judge_reason": d.get("judge_reason"),
            "human": d.get("human"),
            "edited": bool(d.get("edited")),
        }
        if d["stratum"] in OPEN_LAYERS:
            rec["text_available"] = True
            rec["user"] = d.get("user")
            rec["assistant"] = d.get("assistant")
            comp["with_text"] += 1
        else:
            rec["text_available"] = False
            rec["withheld_reason"] = "internal knowledge-base source (see Data Availability)"
            comp["label_only"] += 1
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1

print("wrote %s: %d records (%s)" % (OUT, n, dict(comp)))

# verify Table 3 from the release file itself
rows = [json.loads(l) for l in open(OUT, encoding="utf-8")]
ct = collections.Counter((r["machine_verdict"], r["human"]) for r in rows)
fail_n = sum(v for (m, h), v in ct.items() if m == "FAIL")
warn_n = sum(v for (m, h), v in ct.items() if m == "WARN")
fail_keep = ct.get(("FAIL", "keep"), 0)
warn_drop = ct.get(("WARN", "drop"), 0)
agree = ct.get(("FAIL", "drop"), 0) + sum(v for (m, h), v in ct.items() if m == "WARN" and h in ("keep", "edit"))
print("Table3 check: FAIL n=%d (keep=%d, %.1f%%) | WARN n=%d (drop=%d, %.1f%%) | agreement=%d/%d=%.1f%%"
      % (fail_n, fail_keep, fail_keep / fail_n * 100,
         warn_n, warn_drop, warn_drop / warn_n * 100,
         agree, n, agree / n * 100))
