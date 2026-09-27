#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""看 s6 的换图测试原始样例，判断指标是否真的有区分力。"""

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
D = Path(str(_REPO_ROOT / "data/ecommerce_multimodal/sample7"))
truth = json.loads((D / "attrs_truth.json").read_text("utf-8"))


def clause(a):
    m = re.search(r"图中为(.*?)。", a)
    return m.group(1).strip() if m else None


rows = json.loads((U / "eval7_s6.json").read_text("utf-8"))["rows"]
print(f"s6 共 {len(rows)} 张\n")
for r in rows[:6]:
    print(f"[{r['image']}]")
    print(f"  自己图部件: {r['parts_own']}")
    print(f"  换图({r['other']}) 部件: {r['parts_other']}")
    print(f"  喂自己的图 -> 从句: {clause(r['a_own'])}")
    print(f"  喂换过的图 -> 从句: {clause(r['a_swap'])}")
    print(f"  两个从句相同? {clause(r['a_own']) == clause(r['a_swap'])}")
    print()

n_same_clause = sum(1 for r in rows if clause(r["a_own"]) == clause(r["a_swap"]))
print(f"从句完全相同（不管其他文字）: {n_same_clause}/{len(rows)} = "
      f"{n_same_clause / len(rows):.3f}")
n_same_full = sum(1 for r in rows if r["a_own"] == r["a_swap"])
print(f"整条答案完全相同: {n_same_full}/{len(rows)} = "
      f"{n_same_full / len(rows):.3f}")
