#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""门槛实验结果分析：扣掉"乱说一堆颜色也能蒙中"的运气成分。

问题
    判据是子串匹配。如果模型输出了"黑色 白色 灰色 银色 …"一串颜色词，
    几乎必然命中真值。所以必须报告：
      1) 每条答案里出现了几个不同颜色词（mention count）
      2) 在这个曝光量下，随机命中概率有多大（超几何近似）
      3) 多数类基线（永远回答"黑色"）
    只有"实测命中 ≫ 随机命中"才说明模型真的在看图。
"""

import json
import sys
from math import comb
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

LEX = ["黑", "白", "灰", "银", "金", "棕", "红", "粉", "橙", "黄", "绿", "蓝",
       "紫", "透明", "米", "卡其", "驼", "青", "咖啡", "褐", "杏", "玫",
       "藏青", "墨绿", "军绿", "酒红", "裸色", "香槟"]


def colors_in(text):
    """答案里出现的颜色词（互不包含的取最长的若干，粗略去重）"""
    hit = set()
    for w in LEX:
        if w in text:
            hit.add(w)
    # 去冗余：'墨绿' 命中时 '绿' 也算命中，但只算一个"绿系"
    order = sorted(hit, key=len, reverse=True)
    kept = []
    for w in order:
        if not any(w in k for k in kept):
            kept.append(w)
    return kept


def chance_hit(n_lex, n_alt, n_mention):
    """从 n_lex 个颜色里随口说 n_mention 个，至少命中 n_alt 个候选之一的概率"""
    if n_mention <= 0:
        return 0.0
    if n_mention >= n_lex - n_alt + 1:
        return 1.0
    miss = comb(n_lex - n_alt, n_mention) / comb(n_lex, n_mention)
    return 1.0 - miss


def main():
    files = sys.argv[1:] or [
        "/workspace/user_data/gate_base.json",
        "/workspace/user_data/gate_s6.json",
        "/workspace/user_data/gate_t4b_cap.json",
        "/workspace/user_data/gate_t4b_ans.json",
    ]
    print(f"{'模型':<10s} {'色命中':>7s} {'随机基线':>8s} {'净增益':>7s} "
          f"{'部件':>6s} {'颜色词数':>8s} {'多数类':>7s}")
    print("-" * 62)
    for f in files:
        p = Path(f)
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        s, rows = d["summary"], d["rows"]
        exp, nm = [], []
        for r in rows:
            cs = colors_in(r["raw"])
            nm.append(len(cs))
            exp.append(chance_hit(len(LEX), 4, len(cs)))   # 平均 4 个同色说法
        avg = lambda x: sum(x) / len(x) if x else 0.0
        obs = s["color_hit"]
        base_rate = 1.0 - chance_hit(len(LEX), 4, 0)
        print(f"{s['tag']:<10s} {obs:7.3f} {avg(exp):8.3f} {obs - avg(exp):+7.3f} "
              f"{s['parts_hit']:6.3f} {avg(nm):8.1f} "
              f"{sum(1 for r in rows if r['truth_color'] == '黑色') / len(rows):7.3f}")

    # 逐条：露出最多的那几条最可疑
    print("\n最可疑（颜色词 >= 4 个）：")
    for f in files:
        p = Path(f)
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for r in d["rows"]:
            cs = colors_in(r["raw"])
            if len(cs) >= 4:
                print(f"  [{d['summary']['tag']}] {r['id']} 说了{len(cs)}个色"
                      f"{cs} 真值{r['truth_color']}")
                print(f"      {r['raw'][:150]}")


if __name__ == "__main__":
    main()
