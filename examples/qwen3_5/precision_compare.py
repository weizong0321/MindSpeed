#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""精度比对：sdpa 跑 vs eager 跑，逐步 loss 的相对绝对误差。

对应截图里的 "Comparison relative abs Chart"：
    Error    = |loss_eager - loss_sdpa| / |loss_sdpa|      （相对绝对误差）
    Baseline = 0.02                                        （阈值线）
    统计量   = Mean / MeanSquare / Max / Min Error

两次运行只差 attn_implementation，数据、seed、步数、精度全同，
所以数学上应当几乎一致；不一致的程度就是实现差异带来的数值误差。
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
# 允许通过命令行指定两条日志：参考  对比
REF_LOG = sys.argv[1] if len(sys.argv) > 1 else "train_sample8b.log"
CMP_LOG = sys.argv[2] if len(sys.argv) > 2 else "train_sample8b_eager.log"
OUT_JSON = sys.argv[3] if len(sys.argv) > 3 else "precision_compare.json"


def parse(path):
    out = {}
    p = Path(path)
    if not p.exists():
        return out
    for line in p.open(encoding="utf-8", errors="replace"):
        if "iteration" not in line:
            continue
        m = PAT.search(line)
        if m:
            it = int(m.group(1))
            out[it] = {"iter": it, "total": int(m.group(2)),
                       "ms": float(m.group(3)), "lr": float(m.group(4)),
                       "loss": float(m.group(5)), "gnorm": float(m.group(6))}
    return out


a = parse(U / REF_LOG)            # 参考
b = parse(U / CMP_LOG)            # 对比
print(f"参考({REF_LOG}) {len(a)} 步，对比({CMP_LOG}) {len(b)} 步")

common = sorted(set(a) & set(b))
print(f"可逐步对齐 {len(common)} 步")

if not common:
    print("没有可对齐的步，无法比对")
    sys.exit(1)

err, sq = [], []
curve = []
for it in common:
    la, lb = a[it]["loss"], b[it]["loss"]
    e = abs(lb - la) / max(abs(la), 1e-12)
    err.append(e)
    sq.append(e * e)
    curve.append({"iter": it, "ref": la, "cmp": lb, "err": e})

mean_e = sum(err) / len(err)
mse = sum(sq) / len(sq)
res = {
    "n_steps": len(common),
    "mean_error": mean_e,
    "mean_square_error": mse,
    "max_error": max(err),
    "min_error": min(err),
    "baseline": 0.02,
    "n_over_baseline": sum(1 for e in err if e > 0.02),
    "ref_loss_first": a[common[0]]["loss"],
    "ref_loss_last": a[common[-1]]["loss"],
    "cmp_loss_first": b[common[0]]["loss"],
    "cmp_loss_last": b[common[-1]]["loss"],
    "ref_ms_avg": sum(a[i]["ms"] for i in common) / len(common),
    "cmp_ms_avg": sum(b[i]["ms"] for i in common) / len(common),
    "curve": curve,
}
# 同时把两条 loss 曲线整条存下（画上面那张 Loss 图要用）
res["loss_ref"] = [{"iter": i, "loss": a[i]["loss"]} for i in sorted(a)]
res["loss_cmp"] = [{"iter": i, "loss": b[i]["loss"]} for i in sorted(b)]

Path(U / OUT_JSON).write_text(
    json.dumps(res, ensure_ascii=False), encoding="utf-8")

print("\n" + "=" * 60)
print(f"精度比对结果（{CMP_LOG} vs {REF_LOG}）")
print("=" * 60)
print(f"  对齐步数            {res['n_steps']}")
print(f"  Mean Error          {mean_e:.15f}")
print(f"  Mean Square Error   {mse:.15f}")
print(f"  Max Error           {res['max_error']:.15f}")
print(f"  Min Error           {res['min_error']:.15f}")
print(f"  超过阈值 0.02 的步   {res['n_over_baseline']}/{res['n_steps']}")
print(f"\n  loss 首/末  参考 {res['ref_loss_first']:.4f}/{res['ref_loss_last']:.4f}"
      f"   对比 {res['cmp_loss_first']:.4f}/{res['cmp_loss_last']:.4f}")
print(f"  每步耗时    参考 {res['ref_ms_avg']:.1f} ms   "
      f"对比 {res['cmp_ms_avg']:.1f} ms"
      f"  （eager 慢 {res['cmp_ms_avg']/max(res['ref_ms_avg'],1e-9):.2f} 倍）")
print("=" * 60)
print(f"→ {U / OUT_JSON}")
print("PRECISION_COMPARE_DONE")
