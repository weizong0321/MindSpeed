#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""挑一批属性验证用的图：按品类分层，每类 ~6 张。

为什么不能用之前那 30 张
    30 张铺在 19 个品类上，每个属性只有 1-2 个样本，
    "一致率 1.000" 在 n=2 上没有任何统计意义。
    这里按品类各取 6 张，让每个属性能有 ~6-12 个样本（同品类多属性）。

样本来源：captions_all.jsonl（855 张，含品类），
        并排除门槛实验那 30 张，保证是独立样本。
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
B = Path(str(_REPO_ROOT / "data/ecommerce_multimodal/sample5"))

gate_imgs = {it["image"] for it in json.loads(
    (B / "gate_images.json").read_text("utf-8"))}

by_cat = defaultdict(list)
for line in (U / "captions_all.jsonl").open(encoding="utf-8"):
    d = json.loads(line)
    if not d.get("attrs"):
        continue
    p = d["image"].split("/")
    cat = p[-3]
    rel = "/".join(p[-4:])                      # images/cat/sku/xxx.jpg
    if rel in gate_imgs:
        continue
    by_cat[cat].append(rel)

PER_CAT = 6
picked = []
for cat in sorted(by_cat):
    imgs = sorted(set(by_cat[cat]))
    step = max(1, len(imgs) // PER_CAT)
    chosen = imgs[::step][:PER_CAT]
    for i, im in enumerate(chosen):
        picked.append({"id": f"{cat[:6]}_{i:02d}", "image": im, "category": cat})

out = U / "attr_sample.json"
out.write_text(json.dumps(picked, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"选中 {len(picked)} 张，{len(by_cat)} 个品类")
for cat in sorted(by_cat):
    n = sum(1 for p in picked if p["category"] == cat)
    print(f"  {cat:22s} {n} 张（可选 {len(set(by_cat[cat]))}）")
print(f"→ {out}")
