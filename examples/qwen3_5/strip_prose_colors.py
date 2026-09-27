#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把答案**散文部分**里的颜色句也去掉（从句早就清干净了）。

背景（订正之前的错误结论）
    eval_v7 的 color_in_clause 只看「图中为…」一段，于是得出 s7 颜色断言 = 0。
    但真实推理输出里整条答案有颜色的占 19.1% —— 残留来自
    subagent 按**子样式**写的模板散文，例如：
        "三张图里出现了黑色光面带和棕色压印花纹带两种"
        "黑色好搭也耐脏。"
    这些是按子样式说的，不是按这张图说的，仍然属于"没看图就断言颜色"。

做法
    按 。；！？ 切句，丢掉含颜色词的句子，其余原样保留。
    必须保住两类关键句子：
      - 「图中为…」视觉从句（现在已无颜色）
      - 「…图中看不到…以商品页标注为准」这类免责句
"""

import argparse
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

COLORS = ["黑色", "白色", "灰色", "银色", "金色", "棕色", "红色", "粉色",
          "橙色", "黄色", "绿色", "蓝色", "紫色", "透明", "藏青", "墨绿",
          "酒红", "深蓝", "浅蓝", "深灰", "浅灰", "银灰", "米色", "卡其"]
SINGLE = ["黑", "白", "灰", "银", "棕", "红", "粉", "橙", "黄", "绿", "蓝", "紫"]
NON_COLOR = ["金属", "白底", "黑科技", "口红", "网红", "走红", "红人",
             "白银", "黑色素"]
SCENES = ["白底商品图", "实拍场景", "手持展示", "带包装盒"]
SPLIT = re.compile(r"(?<=[。；！？])")


def has_color(seg):
    core = seg
    for s in SCENES:
        core = core.replace(s, "")
    if any(c in core for c in COLORS):
        return True
    tmp = core
    for n in NON_COLOR:
        tmp = tmp.replace(n, "")
    return any(c in tmp for c in SINGLE)


def strip_colors(text):
    parts = [p for p in SPLIT.split(text) if p]
    kept, dropped = [], []
    for p in parts:
        # 关键句永不丢
        if "图中为" in p or "图中看不到" in p or "图中没有" in p:
            kept.append(p)
            continue
        if has_color(p):
            dropped.append(p)
        else:
            kept.append(p)
    return "".join(kept), dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="sample8")
    ap.add_argument("--dst", default="sample8b")
    a = ap.parse_args()

    B = Path(str(MM_DATA))
    src, dst = B / a.src, B / a.dst
    dst.mkdir(parents=True, exist_ok=True)

    stats = {"n": 0, "changed": 0, "len_before": 0, "len_after": 0}
    examples = []
    for f in ("train.json", "eval.json"):
        rows = json.loads((src / f).read_text("utf-8"))
        for r in rows:
            ans = r["conversations"][1]["value"]
            new, dropped = strip_colors(ans)
            stats["n"] += 1
            stats["len_before"] += len(ans)
            stats["len_after"] += len(new)
            if dropped:
                stats["changed"] += 1
                if len(examples) < 4:
                    examples.append((ans, dropped, new))
            r["conversations"][1]["value"] = new
        (dst / f).write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    for f in ("questions.json", "attrs_truth.json", "attr_spec.json"):
        p = src / f
        if p.exists():
            (dst / f).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")

    n = stats["n"]
    print(f"{a.src} -> {a.dst}：{n} 条")
    print(f"  有句子被删掉的: {stats['changed']}/{n} = "
          f"{stats['changed'] / n:.3f}")
    print(f"  平均长度 {stats['len_before'] / n:.0f} -> "
          f"{stats['len_after'] / n:.0f} 字")
    for old, dr, new in examples:
        print(f"\n  删除: {dr}")
        print(f"  改后: {new[:170]}")


if __name__ == "__main__":
    main()
