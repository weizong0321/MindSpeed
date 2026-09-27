#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""公平地比较 s6 与 s7 的"逐图复现"能力。

问题
    前面 analyze_swap.py 只用 sample7 的从句格式去比，s6 的从句带颜色，
    格式根本不同，所以 s6 的精确匹配恒为 0 —— 那个比较对 s6 不公平。

做法
    各自用**自己训练时的真值**重建应有的从句：
      s6:  {color}{category}，{parts[:4]}，{scene}   （来自 sample6/attrs_truth.json）
      s7:  {category}，{parts[:4]}，{scene}          （来自 sample7/attrs_truth.json）
    然后看"喂它自己的图时，从句是否逐字等于该图应有的从句"。
    这个数才是两个模型之间可比的"逐图记忆/复现"强度。

另外统计：换图后从句是否跟着换（跟随率），以及不跟随（仍在说原图）的比例。
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
B = Path(str(MM_DATA))
FORBIDDEN = ["品牌", "型号", "价格", "容量", "功率", "尺码", "成分"]


def clause_of(answer):
    m = re.search(r"图中为(.*?)。", answer)
    return m.group(1).strip() if m else None


def exp_s6(t):
    if not t:
        return None
    color = (t.get("color") or "").strip()
    cat = (t.get("category") or "").strip()
    parts = [p.strip() for p in (t.get("parts") or []) if p and p.strip()]
    parts = [p for p in parts if not any(f in p for f in FORBIDDEN)]
    scene = (t.get("scene") or "").strip()
    head = f"{color}{cat}" if color and cat else (cat or color)
    seg = head
    if parts:
        seg += "，" + "、".join(parts[:4])
    if scene:
        seg += "，" + scene
    return seg.strip("，") or None


def exp_s7(t):
    if not t:
        return None
    cat = (t.get("category") or "").strip()
    parts = t.get("parts") or []
    scene = (t.get("scene") or "").strip()
    seg = cat
    if parts:
        seg += "，" + "、".join(parts[:4])
    if scene:
        seg += "，" + scene
    return seg.strip("，") or None


def measure(tag, rows, truth, expfn):
    own = swap_new = swap_old = none = 0
    n = 0
    for r in rows:
        ci, cj = clause_of(r["a_own"]), clause_of(r["a_swap"])
        ei, ej = expfn(truth.get(r["image"])), expfn(truth.get(r["other"]))
        n += 1
        none += 1 if ci is None else 0
        own += 1 if (ci and ei and ci == ei) else 0
        swap_new += 1 if (cj and ej and cj == ej) else 0
        swap_old += 1 if (cj and ei and cj == ei) else 0
    print(f"\n===== {tag}（{n} 张）=====")
    print(f"  从句缺失率                          {none / n:.3f}")
    print(f"  喂自己的图 → 逐字等于该图应有从句    {own / n:.3f}")
    print(f"  喂换过的图 → 逐字等于新图应有从句    {swap_new / n:.3f}")
    print(f"  喂换过的图 → 仍在等于原图应有从句    {swap_old / n:.3f}")
    return own / n, swap_new / n, swap_old / n


def main():
    t6 = json.loads((B / "sample6/attrs_truth.json").read_text("utf-8"))
    t7 = json.loads((B / "sample7/attrs_truth.json").read_text("utf-8"))

    res = {}
    for tag, f, truth, fn in (("s6", "eval7_s6.json", t6, exp_s6),
                              ("s7", "eval7_s7.json", t7, exp_s7)):
        p = U / f
        if not p.exists():
            continue
        rows = json.loads(p.read_text("utf-8"))["rows"]
        res[tag] = measure(tag, rows, truth, fn)

    if len(res) == 2:
        print("\n" + "=" * 70)
        print(f"{'模型':<6s}{'自身从句复现':>14s}{'换图跟随':>12s}{'换图仍说原图':>14s}")
        print("-" * 70)
        for tag in ("s6", "s7"):
            a, b, c = res[tag]
            print(f"{tag:<6s}{a:>14.3f}{b:>12.3f}{c:>14.3f}")
        print("=" * 70)
        print("'自身从句复现' 越高 = 越能把这张图的具体属性写对（不是套品类模板）")
        print("'换图跟随' 高且 '换图仍说原图' 低 = 答案真的跟着图片走")

    # 逐图：s7 自己图 vs 换图，从句是否变化
    p = U / "eval7_s7.json"
    if p.exists():
        rows = json.loads(p.read_text("utf-8"))["rows"]
        same_clause = sum(1 for r in rows
                          if clause_of(r["a_own"]) == clause_of(r["a_swap"]))
        print(f"\ns7 换图前后从句完全相同: {same_clause}/{len(rows)} = "
              f"{same_clause / len(rows):.3f}")
    p = U / "eval7_s6.json"
    if p.exists():
        rows = json.loads(p.read_text("utf-8"))["rows"]
        same_clause = sum(1 for r in rows
                          if clause_of(r["a_own"]) == clause_of(r["a_swap"]))
        print(f"s6 换图前后从句完全相同: {same_clause}/{len(rows)} = "
              f"{same_clause / len(rows):.3f}")


if __name__ == "__main__":
    main()
