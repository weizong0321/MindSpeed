#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 855 张图整理成属性标注用的清单（每品类样本量足够大）。

为什么要全量跑
    上一轮每品类只有 6 张，导致：
      - "取值单一"可能只是这 6 张碰巧一样（误杀有信息量的属性）
      - 一致率 1.000 在 n=6 上置信区间很宽
    这里用 sample5 的全部 855 张（21 个品类，每类 14–89 张），
    让每个属性能有几十个样本，才谈得上"这个属性有没有信息量"。
"""

import json
import sys
from collections import defaultdict
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
by_cat = defaultdict(list)
seen = set()
for line in (U / "captions_all.jsonl").open(encoding="utf-8"):
    d = json.loads(line)
    if not d.get("attrs"):
        continue
    p = d["image"].split("/")          # .../sample5/images/cat/sku/xxx.jpg
    rel = "/".join(p[-4:])             # images/cat/sku/xxx.jpg
    cat = p[-3]
    if rel in seen:
        continue
    seen.add(rel)
    by_cat[cat].append(rel)

out = []
for cat in sorted(by_cat):
    for i, rel in enumerate(sorted(by_cat[cat])):
        out.append({"id": f"{cat}__{i:03d}", "image": rel, "category": cat})

p = U / "attr_list_855.json"
p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"共 {len(out)} 张，{len(by_cat)} 个品类")
for cat in sorted(by_cat):
    print(f"  {cat:22s} {len(by_cat[cat])}")
print(f"→ {p}")
