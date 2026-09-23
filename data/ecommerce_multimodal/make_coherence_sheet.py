#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为每个品类生成"纯度检查表"：10 个子样式 × 每个取 6 张图。
一眼就能看出每个子样式是不是"同一款商品"，还是要拆开。

用法
    python make_coherence_sheet.py --root sample3/images --out-dir . --per-style 6
"""

import argparse
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from PIL import Image, ImageDraw


def sheet(style_dirs, out, per_style=6, thumb=150, label_w=150):
    rows = len(style_dirs)
    pad, head = 6, 28
    W = label_w + per_style * (thumb + pad) + pad * 2
    H = head + rows * (thumb + pad) + pad
    canvas = Image.new("RGB", (W, H), (18, 20, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 8), out.stem + "  每行一个子样式，看这 6 张是不是同一款商品",
              fill=(240, 243, 250))
    for r, d in enumerate(style_dirs):
        y = head + pad + r * (thumb + pad)
        files = sorted(p for p in d.iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        draw.text((pad, y + thumb // 2 - 6), f"{d.name}\n({len(files)}张)",
                  fill=(255, 214, 120))
        step = max(1, len(files) // per_style)
        picks = files[::step][:per_style]
        for c, p in enumerate(picks):
            x = label_w + pad + c * (thumb + pad)
            try:
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    im.thumbnail((thumb - 4, thumb - 4), Image.LANCZOS)
                    canvas.paste(im, (x + (thumb - im.width) // 2,
                                      y + (thumb - im.height) // 2))
            except Exception:                                # noqa: BLE001
                draw.rectangle([x, y, x + thumb, y + thumb], fill=(48, 30, 30))
    canvas.save(out)
    print(f"→ {out} ({W}x{H})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--per-style", type=int, default=6)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir)
    for cat in sorted(p for p in root.iterdir() if p.is_dir()):
        dirs = sorted(p for p in cat.iterdir() if p.is_dir())
        sheet(dirs, out_dir / f"coherence_{cat.name}.png", per_style=args.per_style)
    return 0


if __name__ == "__main__":
    sys.exit(main())
