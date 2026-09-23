#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把某个子样式的全部图片摊成一张联络表（带序号），用于人工核验属性分组。"""

import argparse
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from PIL import Image, ImageDraw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="子样式目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--thumb", type=int, default=230)
    args = ap.parse_args()

    d = Path(args.dir)
    files = sorted(p for p in d.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not files:
        print(f"[错误] {d} 里没有图片")
        return 1

    cols = args.cols
    rows = (len(files) + cols - 1) // cols
    pad, head = 8, 30
    W = cols * args.thumb + (cols + 1) * pad
    H = rows * args.thumb + (rows + 1) * pad + head
    canvas = Image.new("RGB", (W, H), (18, 20, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 9), f"{d.parent.name}/{d.name}  共 {len(files)} 张"
                        f"（序号 = 文件名，用于人工分组）", fill=(240, 243, 250))
    for i, p in enumerate(files):
        r, c = divmod(i, cols)
        x = pad + c * (args.thumb + pad)
        y = head + pad + r * (args.thumb + pad)
        with Image.open(p) as im:
            im = im.convert("RGB")
            im.thumbnail((args.thumb - 4, args.thumb - 4), Image.LANCZOS)
            canvas.paste(im, (x + (args.thumb - im.width) // 2,
                              y + (args.thumb - im.height) // 2))
        draw.text((x + 4, y + args.thumb - 15), p.name,
                  fill=(255, 214, 120))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"联络表 → {args.out} ({W}x{H}, {len(files)} 张)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
