#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""校验 sample8：从句能不能被精确反解成属性（这是评测成立的前提）。"""
# --- path shim (portable paths; auto-generated) ---
import os as _os
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
MM_DATA = _Path(_os.environ.get("MM_DATA", _REPO_ROOT / "data" / "ecommerce_multimodal"))
MM_WORK = _Path(_os.environ.get("MM_WORK", _REPO_ROOT / "work"))
MM_EX = _REPO_ROOT / "examples" / "qwen3_5"
_skill_scripts = _REPO_ROOT / "skill" / "ecommerce_guide_skill" / "scripts"
if str(_skill_scripts) not in _sys.path:
    _sys.path.insert(0, str(_skill_scripts))
# --- end shim ---


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
from attr_render import parse_values  # noqa: E402
import attr_spec  # noqa: E402

D = Path(str(_REPO_ROOT / "data/ecommerce_multimodal/sample8"))
truth = json.loads((D / "attrs_truth.json").read_text("utf-8"))
selected = json.loads((D / "attr_spec.json").read_text("utf-8"))

full_spec = {}
for cat, attrs in selected.items():
    m = {k: o for k, _, o in attr_spec.ATTRS.get(cat, [])}
    full_spec[cat] = [{"key": k, "options": m.get(k, [])} for k in attrs]

rows = []
for f in ("train.json", "eval.json"):
    rows += json.loads((D / f).read_text("utf-8"))
print(f"sample8: {len(rows)} 条问答，覆盖 {len({r['image'] for r in rows})} 张图")

import re

ok = bad = 0
bad_ex = []
n_attr = Counter()
for img, t in truth.items():
    cat = t["category"]
    spec = full_spec.get(cat, [])
    got = parse_values(cat, t["clause"], spec)
    want = t["attrs"]
    n_attr[len(want)] += 1
    if got == want:
        ok += 1
    else:
        bad += 1
        if len(bad_ex) < 5:
            bad_ex.append((img, t["clause"], want, got))

print(f"\n反解往返: 正确 {ok} / 失败 {bad} = {ok / max(ok+bad,1):.3f}")
for img, c, w, g in bad_ex:
    print(f"  {img}\n    从句: {c}\n    期望: {w}\n    反解: {g}")

print(f"\n每张图的属性个数分布: {dict(sorted(n_attr.items()))}")

# 逐图区分度
by_sku = defaultdict(set)
for img, t in truth.items():
    by_sku[img.rsplit("/", 1)[0]].add(t["clause"])
d = [len(v) for v in by_sku.values()]
print(f"每个子样式内不同从句数: 平均 {sum(d)/len(d):.1f}，"
      f"最少 {min(d)}，最多 {max(d)}（共 {len(d)} 个子样式）")

# 颜色词泄漏（多字颜色词，避免"金属"里的"金"误判）
COLORS = ["黑色", "白色", "灰色", "银色", "金色", "棕色", "红色", "粉色",
          "橙色", "黄色", "绿色", "蓝色", "紫色", "透明"]
n_col = 0
for img, t in truth.items():
    core = t["clause"]
    for s in ("白底商品图", "实拍场景", "手持展示", "带包装盒"):
        core = core.replace(s, "")
    if any(c in core for c in COLORS):
        n_col += 1
print(f"从句里出现颜色词: {n_col}/{len(truth)} = {n_col/len(truth):.4f}")

print("\n=== 样例 ===")
seen = set()
for img in sorted(truth):
    k = img.rsplit("/", 1)[0]
    if k in seen:
        continue
    seen.add(k)
    if len(seen) > 6:
        break
    print(f"  {img}\n    从句: {truth[img]['clause']}")
    rec = next((r for r in rows if r["image"] == img), None)
    if rec:
        print(f"    答案: {rec['conversations'][1]['value'][:130]}")
