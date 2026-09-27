#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""标签噪声上限：同一张图，两次同等合理的标注能一致到什么程度？

为什么关键
    检索测试显示模型只有 0.578 的区分度。但在下结论"模型没看图"之前，
    必须先知道**标签本身可不可复现**：
      - 如果同一张图让另一个同样合理的标注者来标，部件集合只有 0.55 重合，
        那任何模型都不可能稳定命中 —— 0.578 已经接近天花板，
        真正该修的是标签，而不是模型结构。
      - 如果标签高度稳定（0.9+），那 0.578 就是模型的锅，值得做结构化输出。

这里用两个独立的"标注者"当代理：
    ① 同一个 4B 老师，换 4 种问法（v1/v2/v3/v4）
    ② 另一个 30B 老师，同一问法
    看同一张图在这些标注之间的部件集合重合度（Jaccard）。
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


def rows_parts(path, key="raw"):
    p = U / path
    if not p.exists():
        return {}
    d = json.loads(p.read_text("utf-8"))
    return {r["id"]: parts_from_raw(r.get(key, "")) for r in d["rows"]}


# ---- 来源 ----
src = {}
pp = U / "gate_prompt_probe.json"
if pp.exists():
    d = json.loads(pp.read_text("utf-8"))
    for name in ("v1_current", "v2_area", "v3_two", "v4_deliberate"):
        if name in d:
            src[f"4B-{name}"] = {r["id"]: parts_from_raw(r["raw"])
                                 for r in d[name]["rows"]}
mc = U / "gate_multicolor.json"
if mc.exists():
    d = json.loads(mc.read_text("utf-8"))
    for name in ("list_all", "by_part"):
        if name in d:
            src[f"4B-{name}"] = {r["id"]: [str(x) for x in (r.get("colors") or [])]
                                 for r in d[name]["rows"]} if False else \
                {r["id"]: parts_from_raw(r["raw"]) for r in d[name]["rows"]}
t4 = rows_parts("gate_t4b_cap.json")
if t4:
    src["4B-v1(run)"] = t4
t30 = rows_parts("gate_t30b_cap.json")
if t30:
    src["30B-v1"] = t30

names = list(src)
print(f"标注来源 {len(names)} 个: {names}\n")
ids = sorted(set().union(*[set(v) for v in src.values()]))
print(f"覆盖 {len(ids)} 张图\n")

print(f"{'来源A':<18s}{'来源B':<18s}{'平均Jaccard':>12s}{'完全相同比例':>14s}{'样本':>6s}")
print("-" * 70)
tot_j, tot_same, tot_n = 0.0, 0, 0
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = src[names[i]], src[names[j]]
        js, same, n = [], 0, 0
        for k in ids:
            if k not in a or k not in b:
                continue
            n += 1
            v = jac(a[k], b[k])
            js.append(v)
            if set(a[k]) == set(b[k]):
                same += 1
        if not n:
            continue
        mj = sum(js) / n
        tot_j += mj * n
        tot_same += same
        tot_n += n
        print(f"{names[i]:<18s}{names[j]:<18s}{mj:>12.3f}{same / n:>14.3f}{n:>6d}")

print("-" * 70)
print(f"{'加权平均':<36s}{tot_j / tot_n:>12.3f}{tot_same / tot_n:>14.3f}{tot_n:>6d}")

print("\n判读：")
print("  >= 0.85  标签很稳定 → 模型的 0.578 是模型的问题，值得做结构化输出")
print("  ~0.55    标签本身不稳定 → 0.578 已接近天花板，该修的是标签")
