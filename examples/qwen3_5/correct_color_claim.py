#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""订正：用"整条答案"重新统计 s6 / s7 的颜色断言率。

之前 eval_v7 的 color_in_clause 只看「图中为…」那一段，
得出 s7 = 0.000，并据此声称"不再输出颜色"。那个结论是错的，
因为 subagent 写的子样式级模板本来就带颜色词，散落在从句之外。
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

U = MM_WORK
COLORS = ["黑色", "白色", "灰色", "银色", "金色", "棕色", "红色", "粉色",
          "橙色", "黄色", "绿色", "蓝色", "紫色", "透明", "藏青", "墨绿",
          "酒红", "深蓝", "浅蓝", "深灰", "浅灰", "银灰", "米色", "卡其"]
SINGLE = ["黑", "白", "灰", "银", "棕", "红", "粉", "橙", "黄", "绿", "蓝", "紫"]
NON_COLOR = ["金属", "白底", "黑科技"]


def clause_of(a):
    m = re.search(r"图中为(.*?)。", a)
    return m.group(1) if m else ""


def hits(text):
    core = text
    for s in ("白底商品图", "实拍场景", "手持展示", "带包装盒"):
        core = core.replace(s, "")
    h = [c for c in COLORS if c in core]
    if h:
        return h
    tmp = core
    for n in NON_COLOR:
        tmp = tmp.replace(n, "")
    return [c for c in SINGLE if c in tmp]


for tag in ("s6", "s7"):
    p = U / f"eval7_{tag}.json"
    if not p.exists():
        continue
    rows = json.loads(p.read_text("utf-8"))["rows"]
    n = len(rows)
    c_clause = sum(1 for r in rows if hits(clause_of(r["a_own"])))
    c_full = sum(1 for r in rows if hits(r["a_own"]))
    c_swap_full = sum(1 for r in rows if hits(r["a_swap"]))
    print(f"\n===== eval7 {tag}（{n} 张真实推理输出）=====")
    print(f"  从句有颜色: {c_clause}/{n} = {c_clause/n:.3f}")
    print(f"  整条答案有颜色: {c_full}/{n} = {c_full/n:.3f}")
    print(f"  换图后整条有颜色: {c_swap_full}/{n} = {c_swap_full/n:.3f}")
    ex = [(r["image"], hits(r["a_own"]), r["a_own"]) for r in rows
          if hits(r["a_own"])][:3]
    for img, h, a in ex:
        print(f"    例 {h}: {a[:150]}")
