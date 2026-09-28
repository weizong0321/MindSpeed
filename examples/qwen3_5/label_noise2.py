#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""标签稳定性（归一化版）。

上一版直接用原始字符串算 Jaccard，太苛刻：
    '网面鞋面' vs '鞋面'  → 判为完全不同，其实是一回事。
训练目标是**白名单归一化之后**的部件，所以稳定性也必须按归一化后的形式来量。
"""
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
import re
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(_Path(__file__).resolve().parent))
from part_vocab import filter_parts  # noqa: E402

U = MM_WORK
B = Path(str(_REPO_ROOT / "data/ecommerce_multimodal/sample5"))

gi = json.loads((B / "gate_images.json").read_text("utf-8"))
sku2cat = {it["sku"]: it["category"] for it in gi}
id2cat = {f"A{i:02d}": it["category"] for i, it in enumerate(gi, 1)}

def jac(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def parts_from_raw(raw):
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return []
    try:
        j = json.loads(m.group(0))
    except Exception:                                          # noqa: BLE001
        return []
    return [p for p in (j.get("parts") or []) if isinstance(p, str)]

def load(path, **kw):
    p = U / path
    if not p.exists():
        return {}
    d = json.loads(p.read_text("utf-8"))
    if "rows" in d:
        return {r["id"]: parts_from_raw(r.get("raw", "")) for r in d["rows"]}
    return {}

src = {}
pp = U / "gate_prompt_probe.json"
if pp.exists():
    d = json.loads(pp.read_text("utf-8"))
    for name in ("v1_current", "v2_area", "v4_deliberate"):
        if name in d:
            src[f"4B-{name}"] = {r["id"]: parts_from_raw(r["raw"])
                                 for r in d[name]["rows"]}
for f, tag in (("gate_t4b_cap.json", "4B-v1"), ("gate_t30b_cap.json", "30B-v1")):
    v = load(f)
    if v:
        src[tag] = v

# 归一化
norm_src = {}
for name, m in src.items():
    norm_src[name] = {k: filter_parts(v, id2cat.get(k, "")) for k, v in m.items()}

names = list(norm_src)
print(f"来源: {names}\n")
ids = sorted(set().union(*[set(v) for v in norm_src.values()]))

print(f"{'来源A':<16s}{'来源B':<16s}{'归一化后Jaccard':>16s}{'完全相同':>10s}{'样本':>6s}")
print("-" * 68)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = norm_src[names[i]], norm_src[names[j]]
        js, same, n = [], 0, 0
        for k in ids:
            if k not in a or k not in b:
                continue
            n += 1
            js.append(jac(a[k], b[k]))
            same += 1 if set(a[k]) == set(b[k]) else 0
        if n:
            print(f"{names[i]:<16s}{names[j]:<16s}{sum(js) / n:>16.3f}"
                  f"{same / n:>10.3f}{n:>6d}")

print("\n=== 同一张图，两个老师（同一问法）归一化后的部件 ===")
a, b = norm_src.get("4B-v1", {}), norm_src.get("30B-v1", {})
for k in ids[:12]:
    if k in a and k in b:
        print(f"  {k} {id2cat.get(k,''):18s} 4B={a[k]}  30B={b[k]}")

# 单老师的上限：同一问法重跑（v1_current vs gate_t4b_cap 是同一 prompt）
print("\n=== 参考：同一 prompt 重跑的稳定性（上界）===")
a, b = norm_src.get("4B-v1_current", {}), norm_src.get("4B-v1", {})
if a and b:
    js = [jac(a[k], b[k]) for k in ids if k in a and k in b]
    print(f"  4B-v1_current vs 4B-v1: 平均 {sum(js) / len(js):.3f} "
          f"（采样数 {len(js)}）")
