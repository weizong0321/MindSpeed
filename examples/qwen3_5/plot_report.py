#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""出图：训练曲线 + 真正的精度报告图。

注意区分两件事
  训练曲线（loss / 数值偏差）回答的是"训练跑得健不健康"；
  精度报告（属性准确率 vs 基线）回答的是"模型答得对不对"。
  截图里那张 Comparison relative abs Chart 属于前者，不能当精度报告用。

产出
  fig1_training.png    loss 曲线
  fig2_accuracy.png    分属性平衡准确率 vs 多数类基线
  fig3_headline.png    三个核心结论的对比（带 95% 置信区间）
"""

import json
import math
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

for _s in (__import__("sys").stdout, __import__("sys").stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
OUT = HERE / "report_figs"
OUT.mkdir(exist_ok=True)

# 让中文能显示（Windows 自带字体）
for f in ("Microsoft YaHei", "SimHei", "DejaVu Sans"):
    try:
        plt.rcParams["font.sans-serif"] = [f]
        break
    except Exception:                                           # noqa: BLE001
        pass
plt.rcParams["axes.unicode_minus"] = False


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


# ============ 图 1：训练曲线 ============
lc = json.loads((HERE / "loss_curves.json").read_text(encoding="utf-8"))
fig, ax = plt.subplots(figsize=(10, 4.6))
for name, rows in lc.items():
    ax.plot([r["iter"] for r in rows], [r["loss"] for r in rows],
            lw=0.5, alpha=0.45, label=f"{name} (raw)")
    # 滑动均值让趋势看得清
    ys = [r["loss"] for r in rows]
    w = 25
    sm = np.convolve(ys, np.ones(w) / w, mode="valid")
    ax.plot(range(w, len(ys) + 1), sm, lw=2.0, label=f"{name} (滑动均值)")
ax.set_title("训练损失曲线 Loss / Loss Chart", fontsize=14, fontweight="bold")
ax.set_xlabel("iteration")
ax.set_ylabel("loss")
ax.grid(True, alpha=0.35)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "fig1_training.png", dpi=130)
plt.close(fig)
print("→ fig1_training.png")

# ============ 图 2：分属性平衡准确率 vs 基线 ============
d = json.loads((HERE / "eval8_s8b.json").read_text(encoding="utf-8"))
pa, rows = d["per_attr"], d["rows"]


def bal(key, rows_sub):
    cat, attr = key.split(".", 1)
    cls, cc = defaultdict(int), defaultdict(int)
    for r in rows_sub:
        if r.get("cat") != cat:
            continue
        g = r["gold_own"].get(attr)
        if g is None:
            continue
        cls[g] += 1
        if r["pred_own"].get(attr) == g:
            cc[g] += 1
    if not cls:
        return None
    return sum(cc[c] / cls[c] for c in cls) / len(cls)


random.seed(0)
keys = sorted(pa)
labels, bals, bases, los, his = [], [], [], [], []
for k in keys:
    n = pa[k]["n"]
    b = bal(k, rows) or 0
    base = max(pa[k]["cls"].values()) / n if n else 0
    boots = []
    for _ in range(800):
        sub = [rows[random.randrange(len(rows))] for _ in range(len(rows))]
        v = bal(k, sub)
        if v is not None:
            boots.append(v)
    boots.sort()
    lo = boots[int(0.025 * len(boots))] if boots else b
    hi = boots[int(0.975 * len(boots)) - 1] if boots else b
    labels.append(f"{k}\n(n={n})")
    bals.append(b)
    bases.append(base)
    los.append(b - lo)
    his.append(hi - b)

fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(labels))
ax.bar(x - 0.2, bals, 0.4, yerr=[los, his], capsize=3,
       label="模型·平衡准确率（带 95% bootstrap 区间）", color="#3b6fb6")
ax.bar(x + 0.2, bases, 0.4, label="对照·多数类基线（不看图）", color="#c0504d")
ax.axhline(0.5, ls="--", lw=1, color="gray")
ax.text(len(labels) - 0.5, 0.51, "0.5 = 瞎猜", ha="right", fontsize=8, color="gray")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7.5)
ax.set_ylabel("准确率")
ax.set_ylim(0, 1.05)
ax.set_title("分属性：模型平衡准确率 vs 不看图的多数类基线（sample8b, 78 张）",
             fontsize=13, fontweight="bold")
ax.grid(True, axis="y", alpha=0.3)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "fig2_accuracy.png", dpi=130)
plt.close(fig)
print("→ fig2_accuracy.png")

# ============ 图 3：三个核心结论 ============
s = d["summary"]
# 整组全对 & 众数对照
allright = sum(1 for r in rows if r["gold_own"]
               and all(r["pred_own"].get(k) == v
                       for k, v in r["gold_own"].items()))
# 众数对照（用训练集众数，完全不看图）。训练集真值文件本地取一份即可；
# 取不到就用之前算好的 40/78（report_ci.py 的输出）。
tr_path = HERE / "attrs_truth_sample8b.json"
maj_hit = 40
if tr_path.exists():
    tr = json.loads(tr_path.read_text(encoding="utf-8"))
    ev = {r["image"] for r in rows}
    cnt = defaultdict(lambda: defaultdict(int))
    for img, t in tr.items():
        if img in ev:
            continue
        for k, v in (t.get("attrs") or {}).items():
            cnt[f"{t['category']}.{k}"][v] += 1
    maj = {k: max(v, key=v.get) for k, v in cnt.items()}
    maj_hit = sum(1 for r in rows
                  if r["gold_own"] and all(
                      maj.get(f"{r['cat']}.{k}") == v
                      for k, v in r["gold_own"].items()))
    print(f"（用训练集众数重算对照 = {maj_hit}/{len(rows)}）")

n = len(rows)
items = [("整组属性全对\n(模型)", allright, n, "#3b6fb6"),
         ("整组属性全对\n(不看图/众数)", maj_hit if maj_hit is not None else 40,
          n, "#c0504d"),
         ("换图测试\n(更像自己那张)", s.get("n_swap") and
          round(s["swap_discrimination"] * s["n_swap"]), s.get("n_swap") or 0,
          "#4f8a4f")]
fig, ax = plt.subplots(figsize=(8, 5))
xs = np.arange(len(items))
vals = [k / m if m else 0 for _, k, m, _ in items]
errs = [wilson(k, m) for _, k, m, _ in items]
los = [v - e[0] for v, e in zip(vals, errs)]
his = [e[1] - v for v, e in zip(vals, errs)]
ax.bar(xs, vals, 0.5, yerr=[los, his], capsize=5,
       color=[c for *_, c in items])
for i, (lab, k, m, _) in enumerate(items):
    ax.text(i, vals[i] + 0.02, f"{k}/{m}\n{vals[i]:.3f}", ha="center",
            fontsize=10)
ax.axhline(0.5, ls="--", lw=1, color="gray")
ax.set_xticks(xs)
ax.set_xticklabels([lab for lab, *_ in items], fontsize=9)
ax.set_ylim(0, 1.0)
ax.set_ylabel("比例")
ax.set_title("核心结论（误差棒为 95% 置信区间，样本小所以区间宽）",
             fontsize=12, fontweight="bold")
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig3_headline.png", dpi=130)
plt.close(fig)
print("→ fig3_headline.png")
print(f"\n整组全对 {allright}/{n} = {allright/n:.3f}   众数对照 "
      f"{maj_hit}/{n} = {(maj_hit or 0)/n:.3f}")
