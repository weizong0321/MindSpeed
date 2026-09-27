#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定位两次同配置运行的分岔点 —— 判断训练是否可复现。

判读逻辑
    同配置、同 seed、同数据顺序的两次运行：
      · 若框架确定 → 逐步 loss 应恒等，误差全程 0
      · 若存在非确定性 → 最初若干步完全相同（误差恰好 0），
        在某个算子触发非确定性后开始分岔，再被混沌放大
    所以"误差恰好为 0 的步数"和"第一个非零误差出现的步"是关键证据。
"""

import json
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
d = json.loads((HERE / "reproducibility_compare.json").read_text("utf-8"))
curve = d["curve"]
base = d.get("baseline", 0.02)

exact = [c["iter"] for c in curve if c["err"] == 0.0]
nonzero = [c for c in curve if c["err"] > 0.0]

print("=" * 78)
print("可复现性诊断（同配置原样重跑 vs 原始运行）")
print("=" * 78)
print(f"  总步数                 {len(curve)}")
print(f"  loss 完全相同的步数    {len(exact)}"
      f"{f'（第 {exact[0]}–{exact[-1]} 步）' if exact else ''}")
if nonzero:
    print(f"  第一个出现差异的步     第 {nonzero[0]['iter']} 步"
          f"   （参考 {nonzero[0]['ref']:.10f} vs "
          f"对比 {nonzero[0]['cmp']:.10f}，误差 {nonzero[0]['err']:.3e}）")
    print(f"  从第 1 步到分岔点的 loss 值: "
          f"{curve[0]['ref']:.10f}  (两次相同)")
else:
    print("  两次运行逐步完全一致 → 训练可复现")

print(f"\n  loss 首  参考 {d['ref_loss_first']:.10f}  对比 "
      f"{d['cmp_loss_first']:.10f}  "
      f"→ {'相同' if abs(d['ref_loss_first']-d['cmp_loss_first'])<1e-12 else '不同'}")
print(f"  loss 末  参考 {d['ref_loss_last']:.6f}  对比 "
      f"{d['cmp_loss_last']:.6f}")
print(f"  Min/Max Error  {d['min_error']:.3e} / {d['max_error']:.3e}")
print(f"  超阈值 {base} 的步   {d['n_over_baseline']}/{d['n_steps']}")

# 分岔前后的误差增长
if nonzero:
    k = nonzero[0]["iter"]
    print(f"\n{'='*78}\n分岔后的误差增长\n{'='*78}")
    print(f"{'窗口':>16s}{'平均误差':>14s}{'最大误差':>14s}")
    print("-" * 78)
    for w in (10, 20, 50, 100, 200, 500, 1000, 1816):
        sub = [c for c in curve if k <= c["iter"] < k + w]
        if not sub:
            continue
        e = [c["err"] for c in sub]
        print(f"{'第%d–%d步' % (k, k + len(sub) - 1):>16s}"
              f"{sum(e)/len(e):>14.3e}{max(e):>14.3e}")

print("\n" + "=" * 78)
print("结论")
print("=" * 78)
if exact and nonzero:
    print(f"  两次运行在第 {exact[-1]} 步之前**逐位完全相同**，"
          f"之后第 {nonzero[0]['iter']} 步出现差异。")
    print("  → 这不是'精度不够'，而是**运行间非确定性**：")
    print("     同一个配置重复跑，会得到不同的训练轨迹。")
    print("  → 常见成因：归约里的原子加、多流执行、" 
          "非确定 kernel、数据加载顺序。")
    print("  → 影响：任何'逐步数值比对'都无法达标；模型无法精确复现。")
    print("  → 解决：开 use_deter_comp=true 或关闭多流"
          "（MULTI_STREAM_MEMORY_REUSE / TASK_QUEUE_ENABLE）。")
elif not nonzero:
    print("  两次运行逐步完全一致 → 框架确定性良好，可直接出'全程达标'的精度图。")
