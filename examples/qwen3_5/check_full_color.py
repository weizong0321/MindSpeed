#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补一个我之前漏掉的检查：整条答案里有没有颜色断言（不只是「图中为…」从句）。

为什么重要
    eval_v7 里的 color_in_clause 只看了「图中为…」那一段，
    于是得出"sample7 颜色断言 0%"。但答案的其余部分来自
    subagent 写的**子样式级**模板，里面可能本来就带颜色词，
    例如"黑色好搭也耐脏"、"三张图里出现了黑色光面带和棕色压印花纹带两种"。
    如果这部分没清掉，那"不再输出不可靠颜色"这个目标只完成了一半。
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

B = Path(str(MM_DATA))

# 多字颜色词（避免"金属"里的"金"误判）
COLORS = ["黑色", "白色", "灰色", "银色", "金色", "棕色", "红色", "粉色",
          "橙色", "黄色", "绿色", "蓝色", "紫色", "透明", "藏青", "墨绿",
          "酒红", "深蓝", "浅蓝", "深灰", "浅灰", "银灰", "米色", "卡其"]
# 单字颜色词，但要排除常见非颜色词
SINGLE = ["黑", "白", "灰", "银", "棕", "红", "粉", "橙", "黄", "绿", "蓝", "紫"]
NON_COLOR = ["金属", "金色属", "白银", "黑科技", "白底", "黑框眼镜",
             "口红", "网红", "走红", "红人", "黑色素"]


def clause_of(a):
    m = re.search(r"图中为(.*?)。", a)
    return m.group(1) if m else ""


def scan(text):
    core = text
    for s in ("白底商品图", "实拍场景", "手持展示", "带包装盒"):
        core = core.replace(s, "")
    hits = [c for c in COLORS if c in core]
    if not hits:
        # 退一步看单字，但排除已知非颜色词
        tmp = core
        for n in NON_COLOR:
            tmp = tmp.replace(n, "")
        hits = [c for c in SINGLE if c in tmp]
    return hits


names = sys.argv[1:] or ["sample6", "sample7", "sample8"]
for name in names:
    d = B / name
    if not d.exists():
        continue
    rows = []
    for f in ("train.json", "eval.json"):
        p = d / f
        if p.exists():
            rows += json.loads(p.read_text("utf-8"))
    if not rows:
        continue
    n_clause = n_full = 0
    ex = []
    for r in rows:
        a = r["conversations"][1]["value"]
        if scan(clause_of(a)):
            n_clause += 1
        h = scan(a)
        if h:
            n_full += 1
            if len(ex) < 3:
                ex.append((r["image"], h, a[:110]))
    print(f"\n===== {name}（{len(rows)} 条）=====")
    print(f"  「图中为…」从句里有颜色词: {n_clause}/{len(rows)} = "
          f"{n_clause / len(rows):.4f}")
    print(f"  **整条答案**里有颜色词  : {n_full}/{len(rows)} = "
          f"{n_full / len(rows):.4f}")
    for img, h, a in ex:
        print(f"    例 {h}: {a}")
