#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""打印 eval7 三个模型的对照表。"""

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
order = [("base", "eval7_base.json"), ("s6", "eval7_s6.json"),
         ("s7", "eval7_s7.json")]
got = {}
for tag, f in order:
    p = U / f
    if p.exists():
        got[tag] = json.loads(p.read_text("utf-8"))["summary"]

if not got:
    print("还没有结果")
    sys.exit(0)

cols = ["n", "grounded_own", "grounded_swap", "leak_own_after_swap",
        "same_answer_rate", "color_in_clause_rate", "bg_in_clause_rate",
        "avg_answer_len", "elapsed_min"]
hdr = f"{'模型':<6s}" + "".join(f"{c[:13]:>15s}" for c in cols)
print(hdr)
print("-" * len(hdr))
for tag, _ in order:
    if tag not in got:
        continue
    s = got[tag]
    print(f"{tag:<6s}" + "".join(f"{s.get(c, 0):>15.3f}" for c in cols))

print("\n说明：")
print("  grounded_own      自己的部件出现在自己答案里（越高越好）")
print("  grounded_swap     换图后描述的是新图的部件（越高=越在按图说话）")
print("  leak_own_after_swap 换图后还在说原图部件（越低越好）")
print("  same_answer_rate  换图前后答案完全相同（越低越好，高=背模板）")
print("  color/bg_in_clause 从句里出现颜色词/背景词（目标 0）")

# sample6 vs sample7 的逐图胜负
if "s6" in got and "s7" in got:
    r6 = {r["image"]: r for r in json.loads(
        (U / "eval7_s6.json").read_text("utf-8"))["rows"]}
    r7 = {r["image"]: r for r in json.loads(
        (U / "eval7_s7.json").read_text("utf-8"))["rows"]}
    common = sorted(set(r6) & set(r7))
    print(f"\n=== s6 vs s7 逐图（{len(common)} 张共同图）===")
    w7 = w6 = tie = 0
    for img in common:
        a, b = r6[img]["grounded_own"], r7[img]["grounded_own"]
        if b > a + 1e-9:
            w7 += 1
        elif a > b + 1e-9:
            w6 += 1
        else:
            tie += 1
    print(f"  接地率 s7 更好 {w7} 张，s6 更好 {w6} 张，持平 {tie} 张")
    print("\n  s6 出现颜色断言的图占比: "
          f"{sum(1 for r in r6.values() if r['color_in_clause']) / len(r6):.3f}")
    print("  s7 出现颜色断言的图占比: "
          f"{sum(1 for r in r7.values() if r['color_in_clause']) / len(r7):.3f}")

    print("\n=== 换图测试样例（s7）===")
    for img in common[:3]:
        r = r7[img]
        print(f"\n[{img}]  自己的部件={r['parts_own']}")
        print(f"  喂自己的图: {r['a_own'][:150]}")
        print(f"  喂换的图({r['other']}, 部件={r['parts_other']}): "
              f"{r['a_swap'][:150]}")
