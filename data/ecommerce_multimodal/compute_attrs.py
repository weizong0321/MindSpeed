#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从图片实际像素计算"可见属性"，供答案合成与客观评测使用。

为什么这么做
    之前的失败模式是"同一子样式的 10-15 张图共用一段答案" → 模型不看图也能把 loss 压到 0。
    如果答案里的颜色/背景描述是**从这张图真实算出来的**，那么：
      ① 同一子样式内部答案也会随图变化 → 模型没法靠背模板过关
      ② 评测变成客观的：模型输出的颜色词 vs 图片真实颜色词，可以直接算命中率

用法
    python compute_attrs.py --root sample3/images --out image_attrs.csv
    python compute_attrs.py --root sample3/images --report      # 只看分布，便于校验取色是否靠谱
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import numpy as np
from PIL import Image

# 色相 → 颜色词（饱和度足够时用）
HUE_BINS = [
    (0, 15, "红"), (15, 40, "橙"), (40, 70, "黄"), (70, 165, "绿"),
    (165, 200, "青"), (200, 265, "蓝"), (265, 300, "紫"), (300, 345, "粉"),
    (345, 361, "红"),
]


def _hue_word(h):
    for lo, hi, name in HUE_BINS:
        if lo <= h < hi:
            return name
    return "其他"


def _gray_word(v):
    if v >= 0.82:
        return "白"
    if v >= 0.62:
        return "浅灰"
    if v >= 0.40:
        return "灰"
    if v >= 0.20:
        return "深灰"
    return "黑"


def image_attrs(path: Path):
    """返回 dict：color（主色词）、is_colored、bg（白底/实拍）、brightness"""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        # 取中心 60%，尽量避开背景
        cx, cy = int(w * 0.2), int(h * 0.2)
        cw, ch = max(8, int(w * 0.6)), max(8, int(h * 0.6))
        center = im.crop((cx, cy, cx + cw, cy + ch))
        small = center.resize((48, 48), Image.LANCZOS)
        hsv = np.asarray(small.convert("HSV")).astype(np.float32)
        H = hsv[..., 0] * 360.0 / 255.0
        S = hsv[..., 1] / 255.0
        V = hsv[..., 2] / 255.0

        # 背景判断：外圈靠近白色的像素比例（用整图缩小后的边框）
        tiny = np.asarray(im.resize((32, 32), Image.LANCZOS)).astype(np.float32)
        border = np.concatenate([tiny[0], tiny[-1], tiny[:, 0], tiny[:, -1]])
        near_white = float(((border > 225).all(axis=1)).mean())
        bg = "白底商品图" if near_white > 0.5 else "实拍背景"

    colored = S > 0.25
    if colored.sum() >= 0.12 * colored.size:
        hue = float(np.median(H[colored]))
        color = _hue_word(hue)
        # 偏暗的橙/黄更接近棕色
        if color in ("橙", "黄") and float(np.median(V[colored])) < 0.45:
            color = "棕"
        is_colored = True
    else:
        color = _gray_word(float(np.median(V)))
        is_colored = False
    return {"color": color, "is_colored": is_colored, "bg": bg,
            "brightness": round(float(np.median(V)), 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="image_attrs.csv")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    files = sorted(p for p in root.rglob("*.jpg"))
    if not files:
        print("[错误] 没找到图片")
        return 1
    print(f"处理 {len(files)} 张 ...")

    rows, per_style = [], defaultdict(list)
    for p in files:
        try:
            a = image_attrs(p)
        except Exception as e:                                  # noqa: BLE001
            print(f"  [跳过] {p.name}: {e}")
            continue
        style = f"{p.parent.parent.name}/{p.parent.name}"
        rows.append({"path": str(p), "style": style, **a})
        per_style[style].append(a)

    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "style", "color", "is_colored",
                                          "bg", "brightness"])
        w.writeheader()
        w.writerows(rows)
    print(f"逐图属性 → {args.out}")

    if args.report:
        print("\n每个子样式的颜色分布（校验取色是否靠谱）")
        print("-" * 78)
        print(f"{'子样式':<44}{'张数':>5}  颜色分布")
        for style in sorted(per_style):
            c = Counter(a["color"] for a in per_style[style])
            bg = Counter(a["bg"] for a in per_style[style])
            dist = " ".join(f"{k}×{v}" for k, v in c.most_common())
            bgs = " ".join(f"{k}×{v}" for k, v in bg.most_common())
            print(f"{style:<44}{len(per_style[style]):>5}  {dist}")
            print(f"{'':<44}{'':>5}  {bgs}")
        print("-" * 78)
        allc = Counter(a["color"] for a in rows)
        print("全库颜色分布：" + " ".join(f"{k}×{v}" for k, v in allc.most_common()))
        n_multi = sum(1 for s in per_style.values() if len({a['color'] for a in s}) > 1)
        print(f"子样式内部颜色不唯一的：{n_multi} / {len(per_style)}"
              f"（越高说明答案越不可能靠背）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
