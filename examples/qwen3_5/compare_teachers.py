#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""4B 与 30B 到底错在哪 —— 是"同款难度"还是"各错各的"。

如果两个模型错在**同一批图**上，说明是图本身难/真值有歧义/任务定义有问题，
换更大的老师没有用（实测确实没变）。
如果各错各的，说明可以做投票/集成把准确率抬上去。
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
G = Path(str(_REPO_ROOT / "data/ecommerce_multimodal/sample5/gate_truth.json"))

truth = {t["id"]: t for t in json.loads(G.read_text("utf-8"))["items"]}
runs = {}
for tag in ("t4b_cap", "t30b_cap", "s6"):
    p = U / f"gate_{tag}.json"
    if p.exists():
        runs[tag] = {r["id"]: r for r in json.loads(p.read_text())["rows"]}

COLOR_ALTS = {
    "银色": ["银色", "银白", "银灰", "银"], "棕色": ["棕色", "棕", "褐色", "咖啡色", "深棕"],
    "灰色": ["灰色", "灰", "深灰", "银灰", "灰白"], "黑色": ["黑色", "黑", "纯黑"],
    "白色": ["白色", "白"], "蓝色": ["蓝色", "蓝", "深蓝", "藏青"],
    "红色": ["红色", "红", "正红", "酒红"], "绿色": ["绿色", "绿", "墨绿", "深绿", "青色", "蓝绿"],
    "黄色": ["黄色", "黄"], "粉色": ["粉色", "粉", "蓝", "蓝色"], "透明": ["透明", "无色", "白色", "白"],
}


def said_color(tag, i):
    r = runs.get(tag, {}).get(i)
    if not r:
        return None
    j = None
    import re
    m = re.search(r"\{.*\}", r["raw"], re.S)
    if m:
        try:
            j = json.loads(m.group(0))
        except Exception:                                       # noqa: BLE001
            j = None
    return (j or {}).get("color")


print(f"{'ID':4s} {'真值':>5s} {'4B':>5s} {'30B':>5s} {'学生':>4s}   判定")
print("-" * 62)
both_wrong, only4b, only30b, both_right = [], [], [], []
for i in sorted(truth):
    t = truth[i]
    alts = COLOR_ALTS.get(t["truth_color"], [t["truth_color"]])
    a = said_color("t4b_cap", i)
    b = said_color("t30b_cap", i)
    st = runs.get("s6", {}).get(i)
    ok4 = bool(a) and any(x in a for x in alts)
    ok30 = bool(b) and any(x in b for x in alts)
    oks = bool(st and st["color_hit"])
    if ok4 and ok30:
        both_right.append(i)
        verdict = "都对"
    elif not ok4 and not ok30:
        both_wrong.append(i)
        verdict = "都错"
    elif not ok4:
        only4b.append(i)
        verdict = "只 4B 错"
    else:
        only30b.append(i)
        verdict = "只 30B 错"
    amb = "?" if t["ambiguous"] else " "
    print(f"{i}{amb} {t['truth_color']:>5s} {str(a):>5s} {str(b):>5s} "
          f"{'OK' if oks else 'XX':>4s}   {verdict}")

n = len(truth)
print(f"\n共 {n} 张：都对 {len(both_right)}，都错 {len(both_wrong)}，"
      f"只 4B 错 {len(only4b)}，只 30B 错 {len(only30b)}")
print(f"两者一致率 {(len(both_right) + len(both_wrong)) / n:.3f}")
print(f"都错的图: {both_wrong}")

# 如果按"两个老师投票取并集"算，能到多少
union = len(both_right) + len(only4b) + len(only30b)
print(f"\n如果把 4B+30B 的答案并起来判（任一命中即算对）: {union / n:.3f}")
print(f"（这说明剩余错误里有多少是'两个模型都看不出来'）")
print(f"都错的里面，歧义图有 {sum(1 for i in both_wrong if truth[i]['ambiguous'])} 张")
