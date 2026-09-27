#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多色方案被 precision 否掉了，那还有没有别的出路？

思路：既然单色问法 precision 0.867（说一个色时 87% 是真的），
      那就看看能不能"只在有把握时才说颜色"——用多个 prompt 的投票当置信度。
      如果几个问法都同意某个颜色，这条样本就保留；不同意就弃权（答案里不写颜色）。

产出：
  保留率（还有多少图能给出颜色）
  保留样本上的准确率（precision）
  以及整体覆盖 = 保留率 × 保留样本准确率
"""

import json
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
BASE = Path(str(_REPO_ROOT / "data/ecommerce_multimodal/sample5"))

truth = {t["id"]: t for t in json.loads(
    (BASE / "gate_truth.json").read_text("utf-8"))["items"]}
prod = json.loads((BASE / "gate_product_colors.json")
                  .read_text("utf-8"))["colors"]

NORM = [("透明", "透明"), ("无色", "透明"), ("藏青", "蓝"), ("深蓝", "蓝"),
        ("蓝", "蓝"), ("青", "绿"), ("蓝绿", "绿"), ("墨绿", "绿"),
        ("深绿", "绿"), ("绿", "绿"), ("银白", "银"), ("银灰", "银"),
        ("银", "银"), ("灰白", "灰"), ("深灰", "灰"), ("灰", "灰"),
        ("黑色", "黑"), ("纯黑", "黑"), ("黑", "黑"), ("白色", "白"),
        ("米白", "白"), ("白", "白"), ("正红", "红"), ("酒红", "红"),
        ("红", "红"), ("粉", "粉"), ("橙", "橙"), ("黄", "黄"),
        ("金", "金"), ("棕", "棕"), ("咖啡", "棕"), ("褐", "棕")]


def norm(w):
    for pat, c in NORM:
        if w and pat in w:
            return c
    return None


# 收集所有"单色"投票来源
votes = {}          # id -> {source: color}


def add(src, rows, getter):
    for r in rows:
        c = norm(getter(r))
        if c:
            votes.setdefault(r["id"], {})[src] = c


pp = json.loads((U / "gate_prompt_probe.json").read_text("utf-8"))
for name in ("v1_current", "v2_area", "v3_two", "v4_deliberate"):
    if name in pp:
        add(name, pp[name]["rows"], lambda r: r.get("said"))

mc = json.loads((U / "gate_multicolor.json").read_text("utf-8"))
if "single" in mc:
    add("single", mc["single"]["rows"], lambda r: (r.get("colors") or [None])[0])
if "list_all" in mc:
    add("list_all", mc["list_all"]["rows"], lambda r: (r.get("colors") or [None])[0])

print(f"投票来源数: {len({s for d in votes.values() for s in d})}")
print(f"参与图片: {len(votes)}\n")

# 按"最少几个来源同意"扫描
print(f"{'门槛':>12s} {'保留图数':>8s} {'保留率':>7s} {'保留集准确':>10s} "
      f"{'总体覆盖':>8s}")
print("-" * 56)
best = None
for k in range(1, 6):
    kept, right = 0, 0
    for i, t in truth.items():
        d = votes.get(i, {})
        if len(d) < k:
            continue
        from collections import Counter
        c, cnt = Counter(d.values()).most_common(1)[0]
        if cnt < k:
            continue
        kept += 1
        if c in set(prod[i]):
            right += 1
    rate = kept / len(truth)
    acc = right / kept if kept else 0
    print(f"{'>=%d 票' % k:>12s} {kept:8d} {rate:7.3f} {acc:10.3f} "
          f"{rate * acc:8.3f}")
    if best is None or rate * acc > best[1]:
        best = (k, rate * acc)

print(f"\n小结：单色问法本身的 precision 是 0.867，覆盖率 0.667 → 总体 0.578。")
print(f"      投票提纯的最好结果：门槛 {best[0]} 票，总体覆盖 {best[1]:.3f}")

# 逐图展示 4 票以上的情况
from collections import Counter
print("\n=== 逐图投票（只列有 >=3 票的）===")
for i in sorted(truth):
    d = votes.get(i, {})
    if len(d) < 3:
        print(f"  {i} 票数不足({len(d)}) {d}")
        continue
    c, cnt = Counter(d.values()).most_common(1)[0]
    real = set(prod[i])
    mark = "OK" if c in real else "XX"
    same = "一致" if cnt == len(d) else f"{cnt}/{len(d)}"
    print(f"  {i} {mark} 多数={c} ({same}) 真实集合={sorted(real)} 明细={d}")
