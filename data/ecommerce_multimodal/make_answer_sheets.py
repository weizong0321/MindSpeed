#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 sample5 的 57 个子样式生成"写答案用的检查表"（每表 10 个子样式 × 3 张图）。

用法
    python make_answer_sheets.py --root sample5/images --out-dir ans_sheets
"""

import argparse
import json
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
    ap.add_argument("--root", default="sample5/images")
    ap.add_argument("--selected", default="selected_styles.json")
    ap.add_argument("--out-dir", default="ans_sheets")
    ap.add_argument("--per-sheet", type=int, default=10)
    ap.add_argument("--per-style", type=int, default=3)
    ap.add_argument("--thumb", type=int, default=210)
    args = ap.parse_args()

    sel = json.loads(Path(args.selected).read_text(encoding="utf-8"))["styles"]
    keys = sorted({(s["category"], s["sku"]) for s in sel})
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    thumb, pad, head, label_w = args.thumb, 6, 30, 230
    n_sheet = (len(keys) + args.per_sheet - 1) // args.per_sheet
    manifest = []
    for s in range(n_sheet):
        chunk = keys[s * args.per_sheet:(s + 1) * args.per_sheet]
        W = label_w + args.per_style * (thumb + pad) + pad
        H = head + len(chunk) * (thumb + pad) + pad
        canvas = Image.new("RGB", (W, H), (18, 20, 24))
        draw = ImageDraw.Draw(canvas)
        draw.text((pad, 9), f"写答案用 · 批次 {s + 1}/{n_sheet}"
                            f"   每行 = 一个已核验的同款子样式（3 张代表图）",
                  fill=(240, 243, 250))
        for r, (cat, sku) in enumerate(chunk):
            d = root / cat / sku
            files = sorted(p for p in d.iterdir() if p.suffix.lower() == ".jpg")
            y = head + pad + r * (thumb + pad)
            draw.text((pad, y + thumb // 2 - 18),
                      f"[{cat}]\n{sku}\n{len(files)}张",
                      fill=(255, 214, 120))
            step = max(1, len(files) // args.per_style)
            for c, p in enumerate(files[::step][:args.per_style]):
                x = label_w + pad + c * (thumb + pad)
                try:
                    with Image.open(p) as im:
                        im = im.convert("RGB")
                        im.thumbnail((thumb - 4, thumb - 4), Image.LANCZOS)
                        canvas.paste(im, (x + (thumb - im.width) // 2,
                                          y + (thumb - im.height) // 2))
                except Exception:                            # noqa: BLE001
                    draw.rectangle([x, y, x + thumb, y + thumb], fill=(48, 30, 30))
            manifest.append({"sheet": s + 1, "row": r + 1,
                             "category": cat, "sku": sku, "n": len(files)})
        canvas.save(out_dir / f"ans_{s + 1:02d}.png")
        print(f"  ans_{s + 1:02d}.png  ({len(chunk)} 个子样式)")

    import csv
    with open(out_dir / "manifest.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sheet", "row", "category", "sku", "n"])
        w.writeheader()
        w.writerows(manifest)
    print(f"\n共 {len(manifest)} 个子样式 → {out_dir}/（{n_sheet} 张表）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
