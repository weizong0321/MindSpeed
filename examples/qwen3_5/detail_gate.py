#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐图对比：老师/学生/基线 在人工真值上错在哪。"""

import json
import sys
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
runs = {}
for t in ("base", "s6", "t4b_cap", "t4b_ans"):
    p = U / f"gate_{t}.json"
    if p.exists():
        runs[t] = {r["id"]: r for r in json.loads(p.read_text())["rows"]}

truth = {}
gp = Path(str(_REPO_ROOT / "data/ecommerce_multimodal/sample5/gate_truth.json"))
for t in json.loads(gp.read_text("utf-8"))["items"]:
    truth[t["id"]] = t

ids = sorted(truth)
hdr = f"{'ID':4s} {'子样式':30s} {'真值色':>5s}"
for t in runs:
    hdr += f" {t:>9s}"
print(hdr)
print("-" * len(hdr))

for i in ids:
    t = truth[i]
    row = f"{i}{'?' if t['ambiguous'] else ' '} {t['sku']:30s} {t['truth_color']:>5s}"
    for name, rs in runs.items():
        r = rs.get(i)
        if r is None:
            row += f" {'--':>9s}"
        else:
            row += f" {'✓' if r['color_hit'] else '✗'}{r['parts_hit']:.2f}".rjust(10)
    print(row)

print("\n=== 老师（4B）说错颜色的那些图 ===")
for i in ids:
    r = runs.get("t4b_cap", {}).get(i)
    if r and not r["color_hit"]:
        print(f"  {i} {truth[i]['sku']:30s} 真值={truth[i]['truth_color']:4s} "
              f"老师={r['said']}")
        print(f"      看图备注: {truth[i]['see']}")

print("\n=== 学生（sample6）说错颜色的那些图 ===")
for i in ids:
    r = runs.get("s6", {}).get(i)
    if r and not r["color_hit"]:
        print(f"  {i} {truth[i]['sku']:30s} 真值={truth[i]['truth_color']:4s} "
              f"学生答: {r['raw'][:90]}")

print("\n=== 部件命中为 0 的（老师）===")
for i in ids:
    r = runs.get("t4b_cap", {}).get(i)
    if r and r["parts_hit"] == 0:
        print(f"  {i} 真值部件{truth[i]['parts']} 老师说={r['said']}")
