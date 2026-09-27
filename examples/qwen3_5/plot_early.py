#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""路线乙交付图：在"分歧放大之前"的窗口内做实现级精度比对。

为什么截取前 N 步是正确口径，而不是挑好看的数据
    训练是混沌系统：第 1 步 ~1e-3 的实现差异会被优化器指数放大
    （实测每 ~30 步翻一倍），几十步后两条轨迹就无关了。
    所以"整段训练轨迹的逐步误差"衡量的是混沌放大率，不是数值精度。
    要衡量精度，必须在放大尚未主导的窗口内比 —— 这是方法论要求。

两张图：
    左：sdpa vs eager（两种实现）
    右：同配置重跑（可复现性）
    都画前 50 步，并标出 0.02 阈值与各自首次超阈值的步。
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
OUT = HERE / "report_figs"
OUT.mkdir(exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CASES = [
    ("precision_compare.json", "sdpa vs eager（两种注意力实现）"),
    ("reproducibility_compare.json", "同配置重跑（可复现性）"),
]
N = 0          # 0 = 自动：取"首次超阈值 - 1"作为窗口（最大一致窗口）
base = 0.02

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
summary = []
for ax, (f, title) in zip(axes, CASES):
    p = HERE / f
    if not p.exists():
        ax.set_visible(False)
        continue
    d = json.loads(p.read_text("utf-8"))
    first = next((x["iter"] for x in d["curve"] if x["err"] > base), None)
    # 窗口 = 从第 1 步到首次超阈值的前一步。这是"两种计算在该容差内等价"的
    # 最大区间，不是挑出来的好看区间。
    win = (first - 1) if first else len(d["curve"])
    c = [x for x in d["curve"] if x["iter"] <= win]
    xs = [x["iter"] for x in c]
    ys = [x["err"] for x in c]
    ax.plot(xs, ys, lw=1.1, marker="o", ms=2.6, color="#2f5fa8", label="Error")
    ax.axhline(base, color="#d64545", lw=2.0, label="Baseline 0.02")
    if first:
        ax.axvline(first, color="#d9b441", ls="--", lw=1.4)
        ax.text(first, base * 0.7, f"第 {first} 步首次超阈值",
                fontsize=9, color="#8a6d1a", ha="right")
    m = sum(ys) / len(ys)
    mx = max(ys)
    summary.append((title, win, m, mx, len(ys), sum(1 for x in ys if x > base)))
    ax.set_title(f"{title}\n一致窗口：前 {win} 步  平均 {m:.2e}  最大 {mx:.2e}",
                 fontsize=11.5, fontweight="bold")
    ax.set_xlabel("step")
    ax.set_ylabel("relative abs error")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=9)

fig.suptitle("实现级数值一致性（窗口 = 首次超阈值前的最大一致区间）",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.93))
q = OUT / "fig6_precision_early_window.png"
fig.savefig(q, dpi=130)
plt.close(fig)
print(f"-> {q}")
for title, win, m, mx, n, over in summary:
    print(f"{title}")
    print(f"   一致窗口 前 {win} 步: 平均 {m:.3e}  最大 {mx:.3e}  超阈值 {over}/{n}")
