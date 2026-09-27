#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""换图测试的强化版：精确匹配。

为什么需要
    grounded_own / grounded_swap 用的是"部件命中率"。但同一子样式的图，
    部件本来就高度重合（都是运动鞋，都有鞋面/鞋底/鞋带），所以换图前后
    命中率都会很高 —— 这个指标区分不出"真看图"和"背这个子样式的模板"。

更硬的判据
    训练数据里，每张图的答案都带着它**自己**的「图中为…」从句。
    所以：
      喂图 i -> 从句应该等于 clause(i)
      喂图 j（换图）-> 从句应该等于 clause(j)，而**不是** clause(i)
    clause 是把 category/parts/scene 拼出来的，可以精确重建并逐字比较。
    一个没看图的模型会一直输出 clause(i)。
"""

import json
import re
import sys
from collections import Counter
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


def clause_of(answer):
    m = re.search(r"图中为(.*?)。", answer)
    return m.group(1).strip() if m else None


def expected_clause(t):
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


def main():
    truth = json.loads((D / "attrs_truth.json").read_text("utf-8"))
    for tag, f in (("s7", "eval7_s7.json"), ("s6", "eval7_s6.json")):
        p = U / f
        if not p.exists():
            continue
        rows = json.loads(p.read_text("utf-8"))["rows"]
        own_exact = swap_exact = swap_is_own = own_none = swap_none = 0
        part_own = part_swap_new = part_swap_old = 0.0
        n = 0
        for r in rows:
            ci = clause_of(r["a_own"])
            cj = clause_of(r["a_swap"])
            ei = expected_clause(truth.get(r["image"]))
            ej = expected_clause(truth.get(r["other"]))
            n += 1
            own_none += 1 if ci is None else 0
            swap_none += 1 if cj is None else 0
            own_exact += 1 if (ci and ei and ci == ei) else 0
            swap_exact += 1 if (cj and ej and cj == ej) else 0
            swap_is_own += 1 if (cj and ei and cj == ei) else 0
            Pi = r["parts_own"]
            Pj = r["parts_other"]
            if Pi:
                part_own += sum(1 for x in Pi if ci and x in ci) / len(Pi)
                part_swap_old += sum(1 for x in Pi if cj and x in cj) / len(Pi)
            if Pj:
                part_swap_new += sum(1 for x in Pj if cj and x in cj) / len(Pj)

        print(f"\n===== {tag}（{n} 张）=====")
        print(f"  从句缺失率：own {own_none / n:.3f} / swap {swap_none / n:.3f}")
        print(f"  喂自己的图 → 从句逐字等于该图的从句： {own_exact / n:.3f}")
        print(f"  喂换过的图 → 从句逐字等于新图的从句： {swap_exact / n:.3f}  ← 关键")
        print(f"  喂换过的图 → 从句仍在等于原图的从句： {swap_is_own / n:.3f}  "
              f"← 越低越好（高=背模板）")
        print(f"  部件命中：自己答案×自己部件 {part_own / n:.3f} | "
              f"换图答案×新图部件 {part_swap_new / n:.3f} | "
              f"换图答案×原图部件 {part_swap_old / n:.3f}")

    # 举几个例子
    p = U / "eval7_s7.json"
    if p.exists():
        rows = json.loads(p.read_text("utf-8"))["rows"]
        print("\n===== s7 例子（看换图后描述的是哪张图）=====")
        for r in rows[:4]:
            ei = expected_clause(truth.get(r["image"]))
            ej = expected_clause(truth.get(r["other"]))
            print(f"\n[{r['image']}]")
            print(f"  该图应有从句: {ei}")
            print(f"  喂自己的图 -> {clause_of(r['a_own'])}")
            print(f"[换图 {r['other']}] 该图应有从句: {ej}")
            print(f"  喂换过的图 -> {clause_of(r['a_swap'])}")


if __name__ == "__main__":
    main()
