#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sample8b 评测总结：区分"真读出来了"和"只是在猜多数类"。"""

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
COLORS = ["黑色", "白色", "灰色", "银色", "金色", "棕色", "红色", "粉色",
          "橙色", "黄色", "绿色", "蓝色", "紫色", "透明"]
SINGLE = ["黑", "白", "灰", "银", "棕", "红", "粉", "橙", "黄", "绿", "蓝", "紫"]
NON_COLOR = ["金属", "白底", "口红", "网红", "走红"]


def color_hits(a):
    core = a
    for s in ("白底商品图", "实拍场景", "手持展示", "带包装盒"):
        core = core.replace(s, "")
    h = [c for c in COLORS if c in core]
    if h:
        return h
    t = core
    for n in NON_COLOR:
        t = t.replace(n, "")
    return [c for c in SINGLE if c in t]


for tag, f in (("base", "eval8_base.json"), ("sample8b", "eval8_s8b.json")):
    p = U / f
    if not p.exists():
        continue
    d = json.loads(p.read_text("utf-8"))
    s, rows, pa = d["summary"], d["rows"], d["per_attr"]
    print(f"\n{'='*92}")
    print(f"【{tag}】{s['n']} 张图")
    print(f"{'='*92}")
    print(f"{'属性':<30s}{'n':>4s}{'准确率':>8s}{'平衡准确':>9s}{'多数基线':>9s}"
          f"{'覆盖':>6s}{'净增益':>8s}  判定")
    print("-" * 92)
    wins = 0
    gains = []
    readable = []
    for k in sorted(pa):
        v = pa[k]
        n = v["n"]
        acc = v["correct"] / n if n else 0
        recs = [v["cls_correct"].get(c, 0) / v["cls"][c] for c in v["cls"]]
        bal = sum(recs) / len(recs) if recs else 0
        base = max(v["cls"].values()) / n if n else 0
        cov = len([c for c in v["cls"] if v["cls_correct"].get(c, 0)]) / len(v["cls"]) \
            if v["cls"] else 0
        gain = bal - base
        gains.append(gain)
        if bal >= 0.8 and cov >= 1.0:
            readable.append(k)
            verdict = "真读出来了"
        elif bal >= 0.7:
            verdict = "部分读出"
        elif cov < 1.0:
            verdict = "只猜多数类"
        else:
            verdict = "不行"
        if bal > base:
            wins += 1
        print(f"{k:<30s}{n:>4d}{acc:>8.3f}{bal:>9.3f}{base:>9.3f}{cov:>6.2f}"
              f"{gain:>+8.3f}  {verdict}")
    print("-" * 92)
    print(f"{'宏平均':<30s}{'':>4s}{s['macro_acc']:>8.3f}"
          f"{s['macro_balanced']:>9.3f}{s['macro_baseline']:>9.3f}"
          f"{'':>6s}{sum(gains)/len(gains):>+8.3f}")
    print(f"\n  平衡准确率高于基线的属性: {wins}/{len(pa)}")
    print(f"  『真读出来了』(平衡≥0.80 且覆盖=1.0): {len(readable)} 个")
    for k in readable:
        v = pa[k]
        n = v["n"]
        base = max(v["cls"].values()) / n
        print(f"      {k:<30s} 基线 {base:.2f} -> 平衡 1.00 左右")
    print(f"\n  从句逐字命中  {s['clause_exact']:.3f}（旧 parts 目标只有 0.138）")
    print(f"  换图测试      {s['swap_discrimination']:.3f}"
          f"（n={s['n_swap']}，0.5=瞎猜）")

    # 颜色检查
    nc = sum(1 for r in rows if color_hits(r["a_own"]))
    print(f"  输出含颜色词  {nc}/{len(rows)} = {nc/len(rows):.4f}")

# 举例
p = U / "eval8_s8b.json"
if p.exists():
    rows = json.loads(p.read_text("utf-8"))["rows"]
    print(f"\n{'='*92}\n样例（看得出模型写对了哪一项）\n{'='*92}")
    shown = 0
    for r in rows:
        if not r["gold_own"]:
            continue
        hits = sum(1 for k in r["gold_own"] if r["pred_own"].get(k) == r["gold_own"][k])
        if hits == len(r["gold_own"]) and shown < 4:
            shown += 1
            print(f"\n[{r['image']}] 全部正确")
            print(f"  真值: {r['gold_own']}")
            print(f"  从句: {r['clause_own']}")
    shown = 0
    for r in rows:
        if not r["gold_own"]:
            continue
        if any(r["pred_own"].get(k) != r["gold_own"][k] for k in r["gold_own"]):
            shown += 1
            if shown > 3:
                break
            print(f"\n[{r['image']}] 有错")
            print(f"  真值: {r['gold_own']}")
            print(f"  预测: {r['pred_own']}")
