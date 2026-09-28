#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查白名单在真实数据上的效果：会砍掉多少部件？会不会把图砍到没法用？"""
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
from collections import Counter
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(_Path(__file__).resolve().parent))
from part_vocab import filter_parts  # noqa: E402

CAP = Path("/workspace/user_data/captions_all.jsonl")

print("=== 855 条标注：白名单前后的部件数分布 ===")
before, after, dropped = Counter(), Counter(), []
n = 0
for line in CAP.open(encoding="utf-8"):
    d = json.loads(line)
    if not d.get("attrs"):
        continue
    p = d["image"].split("/")
    cat = p[-3]
    raw_parts = [x for x in (d["attrs"].get("parts") or []) if x]
    kept = filter_parts(raw_parts, cat)
    n += 1
    before[min(len(raw_parts), 6)] += 1
    after[min(len(kept), 6)] += 1
    for x in raw_parts:
        if x not in kept:
            dropped.append((cat, x))

print(f"共 {n} 条")
print(f"{'部件数':>6s} {'过滤前':>8s} {'过滤后':>8s}")
for k in range(0, 7):
    print(f"{k if k < 6 else '6+':>6} {before[k]:8d} {after[k]:8d}")

ok = sum(after[k] for k in range(2, 7))
print(f"\n过滤后仍有 >=2 个部件的图: {ok} / {n} = {ok / n:.3f}")
print(f"过滤后只剩 0 个部件的图: {after[0]} = {after[0] / n:.3f}")
print(f"过滤后只剩 1 个部件的图: {after[1]} = {after[1] / n:.3f}")

print("\n=== 被砍掉最多的部件词（前 40）===")
for w, c in Counter(x for _, x in dropped).most_common(40):
    print(f"  {c:5d}  {w}")

print("\n=== 各类被砍比例 ===")
tot, cut = Counter(), Counter()
for line in CAP.open(encoding="utf-8"):
    d = json.loads(line)
    if not d.get("attrs"):
        continue
    cat = d["image"].split("/")[-3]
    rp = [x for x in (d["attrs"].get("parts") or []) if x]
    kp = filter_parts(rp, cat)
    tot[cat] += len(rp)
    cut[cat] += len(rp) - len(kp)
for cat in sorted(tot):
    print(f"  {cat:22s} {cut[cat]:4d}/{tot[cat]:4d} = {cut[cat] / max(tot[cat], 1):.2f}")

# ---- 30 张门槛图的过滤结果（人工核对精确率用）----
gp = Path("/workspace/user_data/gate_t4b_cap.json")
if gp.exists():
    print("\n=== 30 张门槛图：白名单过滤后的部件（人工核对）===")
    rows = json.loads(gp.read_text("utf-8"))["rows"]
    for r in rows:
        cat = r["sku"].split("/")[0]
        m = re.search(r"\{.*\}", r["raw"], re.S)
        j = json.loads(m.group(0)) if m else {}
        raw_parts = [x for x in (j.get("parts") or []) if x]
        kept = filter_parts(raw_parts, cat)
        print(f"  {r['id']} {cat:22s} {raw_parts} -> {kept}")
