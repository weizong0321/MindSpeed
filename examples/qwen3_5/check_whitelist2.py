#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查白名单效果（修正版：区分"改名"和"真删"）。

上一版把 normalize 当成 drop 统计了：
    '厚实中底' -> '中底' 是改名，不是删除，但被算进了"被砍掉的词"。
这版分开统计，才能真正看出白名单砍掉了什么。
"""

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
from part_vocab import BLOCKLIST, WHITELIST, GENERIC, filter_parts  # noqa: E402
# --- path shim (auto-added for portability) ---
import os as _os
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
MM_DATA = _Path(_os.environ.get("MM_DATA", _REPO_ROOT / "data" / "ecommerce_multimodal"))
MM_WORK = _Path(_os.environ.get("MM_WORK", _REPO_ROOT / "work"))
MM_EX = _REPO_ROOT / "examples" / "qwen3_5"
# --- end shim ---



def classify(raw_parts, cat):
    """返回 (kept_normalized, renamed, truly_dropped)"""
    allowed = sorted(set(WHITELIST.get(cat, [])) | set(GENERIC),
                     key=len, reverse=True)
    kept, renamed, dropped = [], [], []
    for p in raw_parts or []:
        if not isinstance(p, str) or not p.strip():
            continue
        p = p.strip()
        if any(b in p for b in BLOCKLIST):
            dropped.append((p, "命中黑名单"))
            continue
        match = None
        for w in allowed:
            if w in p:
                match = w
                break
        if match is None:
            dropped.append((p, "不在白名单"))
        elif match != p:
            renamed.append((p, match))
            kept.append(match)
        else:
            kept.append(match)
    return kept, renamed, dropped


CAP = Path("/workspace/user_data/captions_all.jsonl")
print("=== 855 条标注：白名单效果 ===")
before, after = Counter(), Counter()
ren_all, drop_all = Counter(), Counter()
n = 0
for line in CAP.open(encoding="utf-8"):
    d = json.loads(line)
    if not d.get("attrs"):
        continue
    cat = d["image"].split("/")[-3]
    rp = [x for x in (d["attrs"].get("parts") or []) if x]
    kept, renamed, dropped = classify(rp, cat)
    n += 1
    before[min(len(rp), 6)] += 1
    after[min(len(set(kept)), 6)] += 1
    for _, m in renamed:
        ren_all[m] += 1
    for p, why in dropped:
        drop_all[p] += 1

print(f"共 {n} 条   （去重后计数）")
print(f"{'部件数':>6s} {'过滤前':>8s} {'过滤后':>8s}")
for k in range(0, 7):
    print(f"{k if k < 6 else '6+':>6} {before[k]:8d} {after[k]:8d}")

ok2 = sum(after[k] for k in range(2, 7))
ok1 = ok2 + after[1]
print(f"\n>=2 个部件: {ok2}/{n} = {ok2 / n:.3f}")
print(f">=1 个部件: {ok1}/{n} = {ok1 / n:.3f}")
print(f"0 个部件  : {after[0]}/{n} = {after[0] / n:.3f}")

print(f"\n=== 真正被删掉的词（前 30，共 {sum(drop_all.values())} 次）===")
for w, c in drop_all.most_common(30):
    print(f"  {c:5d}  {w}")

print(f"\n=== 归一化改名的词（前 15，共 {sum(ren_all.values())} 次，不影响信息量）===")
for w, c in ren_all.most_common(15):
    print(f"  {c:5d}  -> {w}")

print("\n=== 各类被删比例 ===")
tot, cut = Counter(), Counter()
for line in CAP.open(encoding="utf-8"):
    d = json.loads(line)
    if not d.get("attrs"):
        continue
    cat = d["image"].split("/")[-3]
    rp = [x for x in (d["attrs"].get("parts") or []) if x]
    _, _, dr = classify(rp, cat)
    tot[cat] += len(rp)
    cut[cat] += len(dr)
for cat in sorted(tot):
    print(f"  {cat:22s} {cut[cat]:4d}/{tot[cat]:4d} = {cut[cat] / max(tot[cat], 1):.2f}")

gp = Path("/workspace/user_data/gate_t4b_cap.json")
gi = Path(str(_REPO_ROOT / "data/ecommerce_multimodal/sample5/gate_images.json"))
if gp.exists() and gi.exists():
    # gate 的行里 sku 没有品类前缀（如 accessories_001），需要从 gate_images 映射回来
    sku2cat = {it["sku"]: it["category"]
               for it in json.loads(gi.read_text("utf-8"))}
    print("\n=== 30 张门槛图：过滤后的部件（人工核对）===")
    for r in json.loads(gp.read_text("utf-8"))["rows"]:
        cat = sku2cat.get(r["sku"], "?")
        m = re.search(r"\{.*\}", r["raw"], re.S)
        j = json.loads(m.group(0)) if m else {}
        rp = [x for x in (j.get("parts") or []) if x]
        kept, renamed, dropped = classify(rp, cat)
        extra = f"   删:{[p for p, _ in dropped]}" if dropped else ""
        print(f"  {r['id']} {cat:20s} {rp} -> {kept}{extra}")
