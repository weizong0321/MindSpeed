#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""开跑前自检：30 张图都在吗？每个子样式在 eval.json 里都有问题吗？"""

import json
import os
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

BASE = Path(str(MM_DATA))
items = json.loads((BASE / "sample5/gate_images.json").read_text("utf-8"))
truth = json.loads((BASE / "sample5/gate_truth.json").read_text("utf-8"))["items"]
ev = json.loads((BASE / "sample6/eval.json").read_text("utf-8"))

q_by_sku = {}
for s in ev:
    sub = s["image"].split("/")[2]
    q_by_sku.setdefault(sub, s["conversations"][0]["value"])

print(f"gate 图 {len(items)}  真值 {len(truth)}  eval 子样式 {len(q_by_sku)}")
miss_img, miss_q = [], []
for i, it in enumerate(items):
    p = BASE / "sample5" / it["image"]
    if not p.exists():
        miss_img.append(it["image"])
    if it["sku"].split("/")[-1] not in q_by_sku:
        miss_q.append(it["sku"])

print(f"缺图 {len(miss_img)}: {miss_img}")
print(f"缺问题(会用通用问题) {len(miss_q)}: {miss_q}")

print("\n各子样式被问到的问题：")
seen = set()
for it in items:
    if it["sku"] in seen:
        continue
    seen.add(it["sku"])
    q = q_by_sku.get(it["sku"], "(通用)")
    print(f"  {it['sku']:38s} {q[:80]}")

# 多数类基线
from collections import Counter
# --- path shim (auto-added for portability) ---
import os as _os
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
MM_DATA = _Path(_os.environ.get("MM_DATA", _REPO_ROOT / "data" / "ecommerce_multimodal"))
MM_WORK = _Path(_os.environ.get("MM_WORK", _REPO_ROOT / "work"))
MM_EX = _REPO_ROOT / "examples" / "qwen3_5"
# --- end shim ---

c = Counter(t["truth_color"] for t in truth)
print(f"\n真值颜色分布: {dict(c)}")
print(f"多数类基线（全答{ c.most_common(1)[0][0] }）= "
      f"{c.most_common(1)[0][1] / len(truth):.3f}")
print(f"歧义图 {sum(1 for t in truth if t['ambiguous'])} 张")
