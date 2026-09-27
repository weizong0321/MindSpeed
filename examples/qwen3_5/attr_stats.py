#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""闭词表属性的两个关键统计：多数类基线 + 逐图区分度。

为什么要先算这两个
  1) 多数类基线：如果某个属性 6 张图里有 5 张是"系带"，那模型全答"系带"
     就有 0.83 准确率 —— 这个数好看但没意义。
     评测必须报"宏平均 / 平衡准确率"，并且和多数类基线比。
  2) 逐图区分度：如果一个子样式内 15 张图属性完全一样，
     那这些属性不携带任何 per-image 信息，模型靠品类模板就能满分，
     也就无法证明它在看图。
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
# --- path shim (auto-added for portability) ---
import os as _os
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
MM_DATA = _Path(_os.environ.get("MM_DATA", _REPO_ROOT / "data" / "ecommerce_multimodal"))
MM_WORK = _Path(_os.environ.get("MM_WORK", _REPO_ROOT / "work"))
MM_EX = _REPO_ROOT / "examples" / "qwen3_5"
# --- end shim ---


for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

U = MM_WORK
keep = {k["attr"] for k in json.loads((U / "attr_keep.json").read_text("utf-8"))}
rows = json.loads((U / "attr_t4b_big.json").read_text("utf-8"))
sample = json.loads((U / "attr_sample.json").read_text("utf-8"))
img2cat = {s["image"]: s["category"] for s in sample}

# 按属性汇总取值分布
vals = defaultdict(Counter)
for r in rows:
    cat = r["category"]
    j = r.get("attrs") or {}
    for k, v in j.items():
        full = f"{cat}.{k}"
        if full not in keep or v is None:
            continue
        vals[full][str(v).strip()] += 1

print(f"可用属性 {len(keep)} 个\n")
print(f"{'属性':<30s}{'样本':>5s}{'取值数':>7s}{'最多取值':>12s}{'占比':>7s}"
      f"{'多数类基线':>11s}")
print("-" * 80)
base_sum = 0.0
n_attr = 0
for k in sorted(vals):
    c = vals[k]
    tot = sum(c.values())
    top, tn = c.most_common(1)[0]
    base = tn / tot
    base_sum += base
    n_attr += 1
    print(f"{k:<30s}{tot:>5d}{len(c):>7d}{top:>12s}{tn / tot:>7.2f}{base:>11.2f}")
print("-" * 80)
print(f"{'平均多数类基线':<30s}{'':>5s}{'':>7s}{'':>12s}{'':>7s}"
      f"{base_sum / n_attr:>11.3f}")

# 逐图区分度：同一品类内，6 张图的属性组合有多少种
print("\n=== 同一品类内，6 张图的属性组合多样性 ===")
combo = defaultdict(list)
for r in rows:
    cat = r["category"]
    j = r.get("attrs") or {}
    sig = tuple(sorted((k, str(v).strip()) for k, v in j.items()
                       if f"{cat}.{k}" in keep and v is not None))
    if sig:
        combo[cat].append(sig)

tot_sig = tot_uniq = 0
for cat in sorted(combo):
    sigs = combo[cat]
    uniq = len(set(sigs))
    tot_sig += len(sigs)
    tot_uniq += uniq
    print(f"  {cat:22s} {uniq}/{len(sigs)} 种不同属性组合")
print(f"\n  合计 {tot_uniq}/{tot_sig} = {tot_uniq / max(tot_sig,1):.3f}"
      f"（若接近 1.0，说明属性真的逐图不同）")
