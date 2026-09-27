#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按截图版式出图：上 Loss/Loss Chart，中 Comparison relative abs Chart，
下 同一误差的对数纵轴（用来判断是"算子误差"还是"训练混沌放大"）。

用法
    python plot_precision.py [输入json] [输出png] [参考标签] [对比标签]
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

SRC = sys.argv[1] if len(sys.argv) > 1 else "precision_compare.json"
DST = sys.argv[2] if len(sys.argv) > 2 else "fig4_precision_compare.png"
L_REF = sys.argv[3] if len(sys.argv) > 3 else "参考运行"
L_CMP = sys.argv[4] if len(sys.argv) > 4 else "对比运行"

d = json.loads((HERE / SRC).read_text(encoding="utf-8"))
curve = d["curve"]
ref = d["loss_ref"]
cmp_ = d["loss_cmp"]
base = d.get("baseline", 0.02)

fig = plt.figure(figsize=(9.8, 13.6))
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)

# ---- 上：Loss/Loss Chart ----
ax1.plot([r["iter"] for r in ref], [r["loss"] for r in ref],
         lw=0.55, color="#8ab4e8", label=L_REF)
ax1.plot([r["iter"] for r in cmp_], [r["loss"] for r in cmp_],
         lw=0.55, color="#d64545", alpha=0.85, label=L_CMP)
ax1.set_title("Loss/Loss Chart", fontsize=16, fontweight="bold", pad=10)
ax1.set_ylabel("loss")
ax1.grid(True, alpha=0.35)
ax1.legend(fontsize=9)

# ---- 中：Comparison relative abs Chart（按截图原样）----
xs = [c["iter"] for c in curve]
ys = [c["err"] for c in curve]
ax2.plot(xs, ys, lw=0.55, color="#2f5fa8", label="Error")
ax2.axhline(base, color="#d64545", lw=1.8, label="Baseline")
ax2.set_title("Comparison relative abs Chart", fontsize=16,
              fontweight="bold", pad=10)
ax2.set_ylabel("relative abs error")
top = max(max(ys), base) * 1.35
ax2.set_ylim(-0.2 * top, top)
ax2.grid(True, alpha=0.35)
ax2.legend(fontsize=9, loc="upper right")
first_over = next((c["iter"] for c in curve if c["err"] > base), 0)
head20 = sum(c["err"] for c in curve[:20]) / max(len(curve[:20]), 1)
ax2.annotate(f"前 20 步平均 {head20:.1e}\n首次超阈值：第 {first_over} 步",
             xy=(0.02, 0.76), xycoords="axes fraction", fontsize=9.5,
             bbox=dict(boxstyle="round", fc="#fff8e1", ec="#d9b441"))

# ---- 下：对数纵轴 ----
ax3.semilogy(xs, ys, lw=0.6, color="#2f5fa8", label="Error (log scale)")
ax3.axhline(base, color="#d64545", lw=1.8, label="Baseline 0.02")
ax3.set_title("同一误差取对数纵轴", fontsize=12.5, fontweight="bold", pad=10)
ax3.set_xlabel("step")
ax3.set_ylabel("relative abs error (log)")
ax3.grid(True, which="both", alpha=0.3)
ax3.legend(fontsize=9)

fig.text(0.5, 0.008,
         f"Mean Error:{d['mean_error']:.15f}, "
         f"Mean Square Error:{d['mean_square_error']:.15f}\n"
         f"Max Error:{d['max_error']:.15f}, "
         f"Min Error:{d['min_error']:.15f}",
         ha="center", fontsize=9.5, color="#333333")

fig.tight_layout(rect=(0, 0.030, 1, 1))
p = OUT / DST
fig.savefig(p, dpi=130)
plt.close(fig)
print(f"-> {p}")
print(f"   对齐步数 {d['n_steps']}，超阈值 {d['n_over_baseline']} 步")
print(f"   每步耗时比 {d['cmp_ms_avg'] / max(d['ref_ms_avg'], 1e-9):.3f}x")
