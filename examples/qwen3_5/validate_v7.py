#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""校验 sample7 的答案：路线乙是否真的做到了"不断言颜色、不带背景"。

三个指标
  1) 颜色词泄漏率：答案里出现颜色词的条数占比（目标 ≈ 0）
  2) 背景词泄漏率：出现袜子/椅子/桌面/地面… 的条数占比（目标 ≈ 0）
  3) 结构部件接地率：该图白名单部件有多少出现在答案里（目标 1.0，说明确实用了这张图的信息）
"""

import json
import re
import sys
from collections import Counter, defaultdict
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

D = Path(str(_REPO_ROOT / "data/ecommerce_multimodal/sample7"))

# 只保留**多字**颜色词。单字会误伤：
#   "金属扣" 里的 "金" 不是颜色   "白底商品图" 里的 "白" 是场景枚举
COLOR_WORDS = [
    "黑色", "白色", "灰色", "银色", "金色", "棕色", "红色", "粉色", "橙色",
    "黄色", "绿色", "蓝色", "紫色", "透明", "藏青", "墨绿", "酒红", "米色",
    "卡其", "驼色", "深蓝", "浅蓝", "深灰", "浅灰", "银灰",
]
BACKGROUND_WORDS = [
    "袜子", "椅子", "桌面", "地面", "地板", "床单", "床品", "沙发", "墙壁",
    "模特", "手指", "手臂", "鞋盒", "纸箱", "窗帘", "地毯",
]
# 场景枚举本身不是"背景泄漏"，检查前先摘掉
SCENE_VALUES = ["白底商品图", "实拍场景", "手持展示", "带包装盒"]


def main():
    truth = json.loads((D / "attrs_truth.json").read_text("utf-8"))
    rows = []
    for f in ("train.json", "eval.json"):
        rows += json.loads((D / f).read_text("utf-8"))
    print(f"sample7 共 {len(rows)} 条问答，覆盖 {len({r['image'] for r in rows})} 张图")

    color_hit, bg_hit = Counter(), Counter()
    color_rows, bg_rows = [], []
    ground_ok = ground_tot = 0
    per_img_color = defaultdict(set)

    for r in rows:
        a = r["conversations"][1]["value"]
        # 只看「图中为…」从句，因为那是我们注入的视觉断言
        m = re.search(r"图中为(.*?)。", a)
        clause = m.group(1) if m else ""
        # 场景枚举（白底商品图/带包装盒…）不属于商品断言，先摘掉再查
        core = clause
        for s in SCENE_VALUES:
            core = core.replace(s, "")
        cs = [w for w in COLOR_WORDS if w in core]
        if cs:
            color_hit["n"] += 1
            color_rows.append((r["image"], clause, cs))
        bg = [w for w in BACKGROUND_WORDS if w in core]
        if bg:
            bg_hit["n"] += 1
            bg_rows.append((r["image"], clause, bg))
        t = truth.get(r["image"]) or {}
        for p in t.get("parts") or []:
            ground_tot += 1
            if p in clause:
                ground_ok += 1

    n = len(rows)
    print(f"\n1) 视觉从句里出现颜色词的条数: {color_hit['n']}/{n} = "
          f"{color_hit['n'] / n:.4f}")
    print(f"2) 视觉从句里出现背景词的条数: {bg_hit['n']}/{n} = "
          f"{bg_hit['n'] / n:.4f}")
    print(f"3) 结构部件接地率: {ground_ok}/{ground_tot} = "
          f"{ground_ok / max(ground_tot, 1):.4f}")

    if color_rows:
        print("\n=== 颜色泄漏样例（前 10）===")
        for img, clause, cs in color_rows[:10]:
            print(f"  {img}\n     {clause}   <- {cs}")
    if bg_rows:
        print("\n=== 背景词样例（前 10）===")
        for img, clause, bg in bg_rows[:10]:
            print(f"  {img}\n     {clause}   <- {bg}")

    # 逐图差异：同一子样式的答案是不是真的不同
    by_sku = defaultdict(set)
    for r in rows:
        m = re.search(r"images/[^/]+/([^/]+)/", r["image"])
        if not m:
            continue
        clause = re.search(r"图中为(.*?)。", r["conversations"][1]["value"])
        by_sku[r["image"].rsplit("/", 1)[0]].add(
            clause.group(1) if clause else "")
    diffs = [len(v) for v in by_sku.values()]
    print(f"\n4) 每个子样式内侧视觉从句的不同取值数："
          f"平均 {sum(diffs) / len(diffs):.1f}（={sum(diffs)}/{len(diffs)}），"
          f"最少 {min(diffs)}，最多 {max(diffs)}")
    print("   （若为 1，说明该子样式所有图用了同一句话，没有逐图信息）")

    # 样例展示
    print("\n=== 合成答案样例 ===")
    seen = set()
    for r in rows:
        k = r["image"].rsplit("/", 1)[0]
        if k in seen:
            continue
        seen.add(k)
        if len(seen) > 8:
            break
        print(f"\n[{r['image']}]\nQ: {r['conversations'][0]['value']}\n"
              f"A: {r['conversations'][1]['value'][:220]}")


if __name__ == "__main__":
    main()
