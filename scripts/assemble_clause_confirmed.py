# -*- coding: utf-8 -*-
"""W-E final assembly: merge auto-confirmed items with the AI per-item review verdicts
into clause_confirmed.json — the input of the C4 (raw-clause open-book) condition.

Verdict provenance:
  - auto      : a retrieved chunk contains the cited clause/table number (pass1+2, 188 items)
  - ai-review : per-item AI judgment over ranked candidates (batches judge_00..03 + micro), 48
  - excluded  : clause text could not be recovered from the KB; item is dropped from C4
                and the count is reported in the paper (16 items)
All decisions follow docs/C4_clause_verification_protocol.md and are spot-checkable.
"""
import json, os

B = "/root/traffic-llm/eval/benchmark"

auto = json.load(open(B + "/clause_autoconfirmed.json", encoding="utf-8"))
p3 = {json.loads(l)["id"]: json.loads(l) for l in open(B + "/clause_pass3.jsonl", encoding="utf-8")}
micro = {json.loads(l)["id"]: json.loads(l) for l in open(B + "/clause_micro.jsonl", encoding="utf-8")}

# id -> ("p3"|"mi", [candidate indices to merge], note)
CONFIRM = {
    "geo-017":  ("p3", [1], "shuoming 4.2"),
    "geo-026":  ("p3", [1], "术语2.1.13定性定义;7.7.1量化标准未恢复"),
    "geo-033":  ("p3", [0], "7.2.2 滑坡要素"),
    "geo-034":  ("p3", [1], "7.3.1 崩塌定义"),
    "geo-035":  ("p3", [0], "shuoming 7.4.1 三区"),
    "geo-036":  ("p3", [2], "7.1.2 正文10款"),
    "geo-042":  ("p3", [1, 0], "merged: K.0.2 + 表K.0.2-2"),
    "geo-044":  ("p3", [1], "8.2.14 按墩台布孔(同一原则;源第6章条文未直接恢复)"),
    "geo-045":  ("p3", [0], "5.1.2 综合勘察方法"),
    "geom-001": ("p3", [0], "3.0.2 设计速度"),
    "geom-002": ("p3", [0, 2], "merged: 3.0.2 + shuoming(三级30/四级20)"),
    "kc-001":   ("p3", [0], "五个等级名称齐现;3.0.1原句未恢复,建议抽查"),
    "kc-002":   ("p3", [0], "120/100/80 高速"),
    "kc-011":   ("p3", [1], "shuoming 3.2.1 坚硬岩>60MPa"),
    "pave-025": ("p3", [0], "4.1.2 结构层次"),
    "pave-031": ("p3", [0, 1], "merged: 4.5.2 + 表4.5.2"),
    "pave-032": ("p3", [1], "表4.4.2 无机结合料稳定类"),
    "pave-033": ("p3", [0], "shuoming 4.1.2 垫层功能"),
    "pave-034": ("p3", [0], "6.1.1 地下排水设施"),
    "pave-035": ("p3", [1, 0], "merged: 8.3.5 + 4.5.7"),
    "pave-039": ("p3", [1], "3.0.8 28d弯拉强度"),
    "surv-010": ("p3", [1], "6.2.5 直线50m/平曲线20m"),
    "traf-002": ("p3", [2], "10.3.2 停车视距通视三角区"),
    "traf-007": ("p3", [0], "3.1.4 警告标志"),
    "traf-008": ("p3", [0], "4.1.5 禁令标志"),
    "traf-010": ("p3", [0], "5.1.6 指示标志"),
    "traf-012": ("p3", [2], "8.2.2 对向分界线黄实线"),
    "traf-013": ("p3", [0], "8.3.2 白虚线可跨越"),
    "traf-014": ("p3", [0], "10.10.2 导流线白色"),
    "traf-015": ("p3", [1], "表6-4注 各级能量值"),
    "traf-016": ("p3", [1], "6.3.2 八(HA)级"),
    "traf-017": ("p3", [0], "表6-4注 SB:280kJ"),
    "traf-019": ("p3", [2], "8.1.1 隔离栅功能"),
    "traf-020": ("p3", [0], "shuoming 10.1.1;位置由遮挡对向眩光推定"),
    "traf-022": ("p3", [0], "4.5节边沟条款+截水沟分工推定"),
    "traf-023": ("p3", [0], "4.5.7 急流槽"),
    "traf-024": ("p3", [2, 1], "merged: shuoming + 表5.2.1"),
    "traf-025": ("p3", [1], "shuoming 石笼式同重力式原理=自重"),
    # borrow: the 4.5.6 chunk surfaced under traf-023's retrieval
    "traf-021": ("p3x:traf-023", [0], "4.5.6 截水沟距坡顶5m(切片经traf-023检索命中,同库同文档)"),
    # micro-pass confirmations
    "geo-038":  ("mi", [2], "C20 表8.3.5 详判:自由膨胀率>=40(引用规范号3363在库中无对应条款)"),
    "kc-010":   ("mi", [0], "JTG/T D70-2010 5.1.2 限界高度h=5m"),
    "pave-021": ("mi", [0], "表4.0.2 片段含3.75车道宽度"),
    "pave-022": ("mi", [2, 1], "merged: 表4.0.5片段(土路肩0.75) + shuoming 4.0.5"),
    "pave-023": ("mi", [1], "D30 3.3.7 过渡段压实度不小于96%"),
    "traf-009": ("mi", [1], "6.1.9 高速指路标志绿底白字白边"),
    "traf-011": ("mi", [0], "主/辅标志并举条款;分类定义原句未恢复"),
    "traf-018": ("mi", [0, 2], "merged: 附录A + shuoming净区示意(术语原句未恢复)"),
    "tun-017":  ("mi", [1], "JT/T 939.5 3.4 洞外亮度L20(S)定义(与入口关联)"),
}

EXCLUDE = {
    "brg-034": "被引《公路涵洞设计规范》JTG/T 3365-02-2020 不在知识库",
    "brg-035": "被引 JT/T 4-2019 不在知识库",
    "geo-006": "附录A公式(A.0.1)为图片,OCR缺失平方关系,条文文本不足以作答",
    "geo-007": "RQD定义条文未能在库内恢复",
    "geo-037": "采空区'三带'表述未能在库内恢复",
    "pave-013": "软土换填深度条文未恢复",
    "pave-014": "软土排水垫层厚度条文未恢复(仅检索到不同条款0.3m,避免误导不采用)",
    "pave-015": "天然地基预压适用条件条文未恢复",
    "pave-016": "竖向排水体处理深度条文未恢复",
    "pave-029": "F40压实度标准表OCR碎片化,无法明确支撑作答",
    "pave-036": "路拱横坡度条文未恢复",
    "pave-040": "面层厚度表4-3数值未恢复",
    "surv-019": "重现期换算属水文统计原理,库内规范无对应条文",
    "surv-020": "设计流量方法选择条文未恢复",
    "tun-013": "通风方式分类条文未能在库内恢复",
    "tun-016": "公路隧道照明设计细则不在知识库",
}


def get_text(src, idxs, iid):
    if src.startswith("p3x:"):
        rec = p3[src.split(":", 1)[1]]
    elif src == "p3":
        rec = p3[iid]
    else:
        rec = micro[iid]
    cands = rec.get("cands") or rec.get("candidates")
    parts = []
    for i in idxs:
        c = cands[i]
        parts.append(c["text"] if isinstance(c, dict) else c)
    return "\n\n".join(parts)


out = {}
for iid, rec in auto.items():
    out[iid] = {"status": "confirmed", "clause_text": rec["clause_text"],
                "provenance": "auto:key-match", "note": ""}
for iid, (src, idxs, note) in CONFIRM.items():
    assert iid not in out, iid + " already auto-confirmed"
    out[iid] = {"status": "confirmed", "clause_text": get_text(src, idxs, iid),
                "provenance": "ai-review", "note": note}
for iid, why in EXCLUDE.items():
    assert iid not in out, iid + " conflict"
    out[iid] = {"status": "excluded", "clause_text": "", "provenance": "ai-review", "note": why}

n_auto = sum(1 for v in out.values() if v["provenance"] == "auto:key-match")
n_ai = sum(1 for v in out.values() if v["provenance"] == "ai-review" and v["status"] == "confirmed")
n_exc = sum(1 for v in out.values() if v["status"] == "excluded")
print("total %d = auto %d + ai-confirmed %d + excluded %d" % (len(out), n_auto, n_ai, n_exc))
assert len(out) == 252, len(out)

json.dump(out, open(B + "/clause_confirmed.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=0)
print("wrote", B + "/clause_confirmed.json")
