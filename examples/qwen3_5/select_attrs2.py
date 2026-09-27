#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 855 张的大样本上重做属性筛选。

两个维度来源不同，必须分开用：
    可靠性  ← 126 张、两个老师（4B vs 30B）的一致率。只有单老师标注没法估可靠性。
    信息量  ← 855 张的取值分布。这个只需要一个老师就够，样本越大越准。

上一轮只用 6 张/品类估信息量，把"这 6 张碰巧同值"误判成"属性没信息量"。
这一轮每品类 15–90 张，结论才站得住。
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(_Path(__file__).resolve().parent))
from attr_spec import ATTRS, UNSURE  # noqa: E402

U = MM_WORK
reliab = {k["attr"]: k for k in
          json.loads((U / "attr_keep.json").read_text("utf-8"))}
rows = json.loads((U / "attr_t4b_855.json").read_text("utf-8"))

vals = defaultdict(Counter)
for r in rows:
    cat = r["category"]
    j = r.get("attrs") or {}
    for k in (r.get("keys") or []):
        v = j.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if UNSURE in s or not s:
            continue
        vals[f"{cat}.{k}"][s] += 1

info = {}
print(f"{'属性':<32s}{'n':>5s}{'取值':>5s}{'多数类':>8s}{'熵':>6s}"
      f"{'可靠一致':>9s}  判定")
print("-" * 78)
import math
# --- path shim (auto-added for portability) ---
import os as _os
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
MM_DATA = _Path(_os.environ.get("MM_DATA", _REPO_ROOT / "data" / "ecommerce_multimodal"))
MM_WORK = _Path(_os.environ.get("MM_WORK", _REPO_ROOT / "work"))
MM_EX = _REPO_ROOT / "examples" / "qwen3_5"
# --- end shim ---

selected = defaultdict(list)
for full in sorted(vals):
    cat, attr = full.split(".", 1)
    c = vals[full]
    n = sum(c.values())
    top, tn = c.most_common(1)[0]
    base = tn / n if n else 1.0
    ent = -sum((x / n) * math.log2(x / n) for x in c.values()) if n else 0.0
    rel = reliab.get(full)
    agree = rel["agreement"] if rel else None
    cov = rel["coverage"] if rel else None

    reasons = []
    if n < 15:
        reasons.append(f"样本少({n})")
    if len(c) < 2:
        reasons.append("取值单一")
    if base > 0.85:
        reasons.append(f"基线高({base:.2f})")
    if agree is None:
        reasons.append("未做可靠性验证")
    elif agree < 0.85:
        reasons.append(f"可靠差({agree:.2f})")
    elif cov is not None and cov < 0.6:
        reasons.append(f"弃权多({cov:.2f})")

    ok = not reasons
    if ok:
        selected[cat].append(attr)
    info[full] = {"n": n, "n_values": len(c), "baseline": round(base, 3),
                  "entropy": round(ent, 3), "agreement": agree,
                  "coverage": cov, "keep": ok,
                  "reasons": reasons, "values": dict(c)}
    print(f"{full:<32s}{n:>5d}{len(c):>5d}{base:>8.2f}{ent:>6.2f}"
          f"{(agree if agree is not None else -1):>9.2f}  "
          f"{'保留' if ok else '/'.join(reasons)}")

(U / "attr_info_855.json").write_text(
    json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

total = sum(len(v) for v in selected.values())
print(f"\n=== 保留 {total} 个属性 / {len(selected)} 个品类 ===")
spec_out = {}
for cat in sorted(selected):
    spec_out[cat] = [{"key": k, "desc": d, "options": o}
                     for k, d, o in ATTRS.get(cat, []) if k in selected[cat]]
    det = []
    for a in selected[cat]:
        i = info[f"{cat}.{a}"]
        det.append(f"{a}(n={i['n']},基线{i['baseline']:.2f},值{i['n_values']})")
    print(f"  {cat:22s} {', '.join(det)}")

(U / "attr_final_spec_855.json").write_text(
    json.dumps(spec_out, ensure_ascii=False, indent=2), encoding="utf-8")
(U / "attr_selected_855.json").write_text(
    json.dumps({k: v for k, v in selected.items()}, ensure_ascii=False, indent=2),
    encoding="utf-8")
print(f"\n→ {U / 'attr_final_spec_855.json'}")
