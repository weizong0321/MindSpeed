#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可用性体检：光看指标会漏掉的东西。

指标只统计"属性值答对没有"，但用户看到的是整段文字。已知至少有这些问题：
  1) 品类名词可能写错（例：真值是"配饰/皮带"，模型从句写成"钱包，皮带"）
     —— 属性值判对了，但句子本身自相矛盾，指标完全看不出来。
  2) "短板是图中看不到…"这类句式常常逻辑不通（短板不能是"看不到"）
  3) 每段都有固定的免责模板，可能显得机械
"""

import json
import re
import sys
from collections import Counter
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
d = json.loads((U / "eval8_s8b.json").read_text("utf-8"))
rows = d["rows"]

CAT_CN = {
    "accessories": "配饰", "apparel_pants": "长裤", "apparel_shorts": "短裤",
    "apparel_top": "上衣", "appliance_large": "家电", "baby_products": "婴儿用品",
    "bag_backpack": "双肩包", "cosmetics": "化妆品", "health_supplement": "保健品",
    "household_detergent": "洗护用品", "jewelry": "珠宝首饰", "luggage": "行李箱",
    "personal_care": "个护用品", "purse_wallet": "钱包", "shoes_kids": "童鞋",
    "shoes_leather": "皮鞋", "shoes_other": "鞋", "shoes_sneaker": "运动鞋",
    "toy_vehicle": "玩具车", "watch": "手表",
}


def clause_of(a):
    m = re.search(r"图中为(.*?)。", a)
    return m.group(1).strip() if m else None


print("=" * 92)
print(f"完整答案抽样（共 {len(rows)} 条，展示 8 条）")
print("=" * 92)
for r in rows[:8]:
    print(f"\n[{'/'.join(r['image'].split('/')[-3:])}]")
    print(f"  真值属性: {r['gold_own']}")
    print(f"  从句: {r['clause_own']}")
    print(f"  全文: {r['a_own']}")

# ---- 缺陷统计 ----
n = len(rows)
cat_wrong = 0
cat_ex = []
no_disc = 0
short_board = 0
board_ex = []
lens = []
dup_first = Counter()

for r in rows:
    a = r["a_own"]
    lens.append(len(a))
    c = clause_of(a) or ""
    head = c.split("，")[0] if c else ""
    want = CAT_CN.get(r["cat"], "")
    if want and head and head != want:
        cat_wrong += 1
        if len(cat_ex) < 5:
            cat_ex.append((r["image"], want, head, c))
    if "图中看不到" not in a and "图中没有" not in a:
        no_disc += 1
    m = re.search(r"短板是([^。]*)。", a)
    if m:
        short_board += 1
        if ("看不到" in m.group(1) or "图中没有" in m.group(1)) and len(board_ex) < 5:
            board_ex.append(m.group(0))

print("\n" + "=" * 92)
print("缺陷统计")
print("=" * 92)
print(f"  平均答案长度          {sum(lens)/n:.0f} 字（最短 {min(lens)}，最长 {max(lens)}）")
print(f"  从句品类名词写错      {cat_wrong}/{n} = {cat_wrong/n:.3f}   ← 指标查不到，但用户一眼看到")
for img, want, got, c in cat_ex:
    print(f"      真值品类「{want}」→ 模型写「{got}」: {c}")
print(f"  缺免责句(图中看不到…)  {no_disc}/{n} = {no_disc/n:.3f}")
print(f"  「短板是图中看不到…」这类逻辑别扭的句子: {short_board}/{n} = {short_board/n:.3f}")
for b in board_ex:
    print(f"      {b}")
