#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分清"模型的问题"和"模板的问题"。

模型是照着训练数据学的。如果训练答案里本来就有
  「短板是图中看不到…」这种逻辑别扭的句子，
那学生复现它不算模型的错，改模板才是解法。
"""

import json
import re
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
B = Path(str(_REPO_ROOT / "data/ecommerce_multimodal/sample8b"))


def stats(rows, label):
    n = len(rows)
    short = short_bad = disc = 0
    for r in rows:
        a = r["conversations"][1]["value"]
        m = re.search(r"短板是([^。]*)。", a)
        if m:
            short += 1
            if "看不到" in m.group(1) or "图中没有" in m.group(1):
                short_bad += 1
        if "图中看不到" in a or "图中没有" in a:
            disc += 1
    print(f"  {label:<14s} n={n:4d}  「短板是…」出现 {short/n:.3f}  "
          f"其中逻辑别扭(短板=看不到) {short_bad/n:.3f}  "
          f"含免责句 {disc/n:.3f}")


print("=== 训练数据 vs 模型输出：这些缺陷是不是学来的 ===")
for f in ("train.json", "eval.json"):
    stats(json.loads((B / f).read_text("utf-8")), f"训练集 {f}")

d = json.loads((U / "eval8_s8b.json").read_text("utf-8"))
stats([{"conversations": [{"value": ""}, {"value": r["a_own"]}]}
       for r in d["rows"]], "模型输出")

# 模板重复度：有多少答案的"非从句部分"完全相同
print("\n=== 模板重复度（去掉「图中为…」从句后，答案还剩多少种）===")
for f in ("train.json",):
    rows = json.loads((B / f).read_text("utf-8"))
    rest = []
    for r in rows:
        a = r["conversations"][1]["value"]
        rest.append(re.sub(r"图中为.*?。", "", a))
    print(f"  训练集 {len(rows)} 条 → 去掉从句后只有 {len(set(rest))} 种不同文本"
          f" = {len(set(rest))/len(rows):.3f}")
    from collections import Counter
    top = Counter(rest).most_common(1)[0]
    print(f"  最高频的一种出现 {top[1]} 次（占 {top[1]/len(rows):.3f}）")
    print(f"  例: {top[0][:150]}")
