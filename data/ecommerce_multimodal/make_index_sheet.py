#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把聚类结果汇成"总览图"：每个簇取若干张最接近簇心的图，拼成一张大图，
人（或我）看一眼就能给整个数据集做品类盘点。

用法
    python make_index_sheet.py --csv clusters.csv --sheet 1 --out index_sheet_1.png
    python make_index_sheet.py --csv clusters.csv --sheet 2 --from-rank 31 --to-rank 60
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="clusters.csv")
    ap.add_argument("--out", default="index_sheet.png")
    ap.add_argument("--from-rank", type=int, default=1)
    ap.add_argument("--to-rank", type=int, default=30)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--per-cluster", type=int, default=4)
    ap.add_argument("--thumb", type=int, default=148)
    args = ap.parse_args()

    import csv
    rows = []
    with open(args.csv, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append((r["path"], int(r["cluster"]), int(r["rank"]), float(r["purity"])))

    by_rank = {}
    for path, cl, rk, pur in rows:
        if args.from_rank <= rk <= args.to_rank:
            by_rank.setdefault(rk, []).append((pur, cl, path))

    ranks = sorted(by_rank)
    if not ranks:
        print("[错误] CSV 里没有指定 rank 区间的数据")
        return 1

    # 每簇按纯度排序，取最像簇心的若干张
    picks = []
    for rk in ranks:
        items = sorted(by_rank[rk], reverse=True)[:args.per_cluster]
        cl = items[0][1]
        size = len(by_rank[rk])
        purity = sum(p for p, _, _ in by_rank[rk]) / len(by_rank[rk])
        picks.append((rk, cl, size, purity, [p for _, _, p in items]))

    n = len(picks)
    cols = args.cols
    rows_n = (n + cols - 1) // cols
    per_row = 2 if args.per_cluster == 4 else 1
    cell_w = args.thumb * per_row + 8
    cell_h = args.thumb * (2 if args.per_cluster == 4 else 1) + 20
    pad = 8
    W = cols * cell_w + (cols + 1) * pad
    H = rows_n * cell_h + (rows_n + 1) * pad + 30

    canvas = Image.new("RGB", (W, H), (18, 20, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 9),
              f"数据集品类总览  rank {args.from_rank}-{args.to_rank}"
              f"（每格 = 一个簇，取最接近簇心的 {args.per_cluster} 张；"
              f"标注 = rank · 簇号 · 张数 · 纯度）",
              fill=(240, 243, 250))

    for i, (rk, cl, size, purity, paths) in enumerate(picks):
        r, c = divmod(i, cols)
        x0 = pad + c * (cell_w + pad)
        y0 = 30 + pad + r * (cell_h + pad)
        draw.rectangle([x0 - 2, y0 - 2, x0 + cell_w, y0 + cell_h - 18],
                       outline=(60, 66, 78))
        for j, p in enumerate(paths):
            rr, cc = divmod(j, per_row)
            x = x0 + 4 + cc * args.thumb
            y = y0 + 2 + rr * args.thumb
            try:
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    im.thumbnail((args.thumb - 4, args.thumb - 4), Image.LANCZOS)
                    canvas.paste(im, (x + (args.thumb - im.width) // 2,
                                      y + (args.thumb - im.height) // 2))
            except Exception:                                # noqa: BLE001
                draw.rectangle([x, y, x + args.thumb, y + args.thumb], fill=(45, 30, 30))
        draw.text((x0 + 4, y0 + cell_h - 16),
                  f"#{rk} 簇{cl}  {size}张  纯度{purity:.2f}", fill=(255, 214, 120))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"总览图 → {args.out}  ({W}x{H}, {n} 个簇)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
