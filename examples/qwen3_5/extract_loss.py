#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从训练日志里抽出逐步 loss 曲线（为出图准备）。

日志行格式：
  ... =>  [2026-09-26 10:14:44] iteration      741/    2120 | consumed samples: 741
       | elapsed time per iteration (ms): 626.3 | learning rate: 8.22E-06
       | global batch size: 1 | loss: 1.301974E+00 | grad norm: 61.761 |
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

PAT = re.compile(
    r"iteration\s+(\d+)/\s*(\d+).*?"
    r"elapsed time per iteration \(ms\):\s*([\d.]+).*?"
    r"learning rate:\s*([\d.eE+-]+).*?"
    r"loss:\s*([\d.eE+-]+).*?"
    r"grad norm:\s*([\d.eE+-]+)")

U = MM_WORK
LOGS = {
    "sample6": "train_sample6.log",
    "sample7": "train_sample7.log",
    "sample8b": "train_sample8b.log",
}

out = {}
for name, f in LOGS.items():
    p = U / f
    if not p.exists():
        print(f"[skip] {f} 不存在")
        continue
    rows = []
    for line in p.open(encoding="utf-8", errors="replace"):
        if "iteration" not in line:
            continue
        m = PAT.search(line)
        if m:
            rows.append({
                "iter": int(m.group(1)), "total": int(m.group(2)),
                "ms": float(m.group(3)), "lr": float(m.group(4)),
                "loss": float(m.group(5)), "gnorm": float(m.group(6)),
            })
    if not rows:
        print(f"[skip] {f} 没解析出数据")
        continue
    seen = {}
    for r in rows:
        seen[r["iter"]] = r
    rows = [seen[k] for k in sorted(seen)]
    out[name] = rows
    losses = [r["loss"] for r in rows]
    print(f"{name:10s} {len(rows):5d} 步  loss 首 {losses[0]:.4f} "
          f"末 {losses[-1]:.4f}  最小 {min(losses):.4f} 最大 {max(losses):.4f}")

(U / "loss_curves.json").write_text(
    json.dumps(out, ensure_ascii=False), encoding="utf-8")
print(f"\n→ {U / 'loss_curves.json'}")

# 画图（纯 matplotlib，风格对齐截图的两块布局）
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9, 11))
    ax = axes[0]
    for name, rows in out.items():
        ax.plot([r["iter"] for r in rows], [r["loss"] for r in rows],
                lw=0.6, label=name)
    ax.set_title("Loss/Loss Chart", fontsize=16, fontweight="bold")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.4)
    ax.legend()

    # 下图：把"数值精度比对"换成我们真正能做的东西——
    # 用 loss 的滑动均值当作基准，画逐步相对偏差（说明训练稳定度）
    ax = axes[1]
    for name, rows in out.items():
        xs = [r["iter"] for r in rows]
        ys = [r["loss"] for r in rows]
        w = max(5, len(ys) // 50)
        base = []
        for i in range(len(ys)):
            lo = max(0, i - w)
            base.append(sum(ys[lo:i + 1]) / (i + 1 - lo))
        rel = [abs(ys[i] - base[i]) / max(abs(base[i]), 1e-9)
               for i in range(len(ys))]
        ax.plot(xs, rel, lw=0.6, label=f"{name} 逐步相对偏差")
    ax.set_title("Step-wise relative deviation (proxy)", fontsize=13)
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)
    plt.tight_layout()
    png = U / "loss_curve.png"
    plt.savefig(png, dpi=110)
    print(f"→ {png}")
except Exception as e:                                          # noqa: BLE001
    print(f"画图失败: {type(e).__name__}: {e}")
