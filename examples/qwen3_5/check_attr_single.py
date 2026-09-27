#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单模型属性标注体检：弃权率、非法率、解析失败率。"""

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
from attr_spec import ATTRS, UNSURE  # noqa: E402
# --- path shim (auto-added for portability) ---
import os as _os
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
MM_DATA = _Path(_os.environ.get("MM_DATA", _REPO_ROOT / "data" / "ecommerce_multimodal"))
MM_WORK = _Path(_os.environ.get("MM_WORK", _REPO_ROOT / "work"))
MM_EX = _REPO_ROOT / "examples" / "qwen3_5"
# --- end shim ---


U = MM_WORK
tag = sys.argv[1] if len(sys.argv) > 1 else "t4b_big"
rows = json.loads((U / f"attr_{tag}.json").read_text("utf-8"))

tot = unsure = illegal = parse_fail = 0
miss_keys = Counter()
bycat = defaultdict(lambda: [0, 0, 0])
for r in rows:
    cat = r["category"]
    spec = dict((k, opts) for k, _, opts in ATTRS.get(cat, []))
    if not spec:
        continue
    j = r.get("attrs")
    if j is None:
        parse_fail += 1
        continue
    for k, opts in spec.items():
        tot += 1
        v = j.get(k)
        bycat[cat][0] += 1
        if v is None:
            miss_keys[k] += 1
            illegal += 1
            bycat[cat][2] += 1
            continue
        s = str(v).strip()
        if UNSURE in s:
            unsure += 1
            bycat[cat][2] += 1
            continue
        if any(s == o or s.startswith(o) or o.startswith(s) for o in opts):
            bycat[cat][1] += 1
        else:
            illegal += 1
            bycat[cat][2] += 1
            miss_keys[f"非法:{k}={s[:20]}"] += 1

print(f"[{tag}] {len(rows)} 张图，{tot} 个属性槽")
print(f"  JSON 解析失败: {parse_fail}/{len(rows)}")
print(f"  弃权（不确定/缺字段）: {unsure}/{tot} = {unsure / max(tot,1):.3f}")
print(f"  非法/缺失: {illegal}/{tot} = {illegal / max(tot,1):.3f}")
print(f"  有效作答: {(tot - unsure - illegal) / max(tot,1):.3f}")

print("\n=== 按品类：有效作答率 ===")
for c in sorted(bycat):
    t, ok, bad = bycat[c]
    print(f"  {c:22s} 有效 {ok}/{t} = {ok / max(t,1):.3f}  弃/非法 {bad}")

if miss_keys:
    print("\n=== 常见弃权/非法 ===")
    for k, c in miss_keys.most_common(15):
        print(f"  {c:4d}  {k}")
