#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 Step 2 批量生成候选子样式的「纯度检查表」。

背景：实测三种自动纯度指标（簇心纯度 / 二分轮廓 / 严格阈值连通分量）都无法区分
"同一款商品"和"看起来像的一堆商品"，只能靠人眼看联络表。
本脚本把候选子样式按每表 10 个、每个 4 张图排好，供人工 30 秒判读。

用法
    python make_candidate_sheets.py --per-sheet 10 --per-style 4 --sheets 20
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from PIL import Image, ImageDraw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assign", default="assignments.csv")
    ap.add_argument("--keep", default="keep_categories.json")
    ap.add_argument("--labels", default="cluster_labels.json")
    ap.add_argument("--out-dir", default="cand_sheets")
    ap.add_argument("--per-sheet", type=int, default=10)
    ap.add_argument("--per-style", type=int, default=4)
    ap.add_argument("--sheets", type=int, default=20)
    ap.add_argument("--min-size", type=int, default=8)
    args = ap.parse_args()

    keep_cfg = json.loads(Path(args.keep).read_text(encoding="utf-8"))
    keep = set(keep_cfg["keep"])
    cat_names = json.loads(Path(args.labels).read_text(encoding="utf-8")).get(
        "categories", {})

    by_sku = defaultdict(list)
    cat_of = {}
    with open(args.assign, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            by_sku[r["sku"]].append(r["path"])
            cat_of[r["sku"]] = r["category"]

    # 候选：保留品类里、图片数 ≥ min-size 的子样式；按类分层轮取，保证多样性
    per_cat = defaultdict(list)
    for sku, paths in by_sku.items():
        if cat_of.get(sku) not in keep or len(paths) < args.min_size:
            continue
        per_cat[cat_of[sku]].append((sku, sorted(paths)))
    for c in per_cat:
        per_cat[c].sort(key=lambda x: -len(x[1]))

    ordered, i = [], 0
    while True:
        added = False
        for c in sorted(per_cat, key=lambda k: -len(per_cat[k])):
            if i < len(per_cat[c]):
                ordered.append((c, *per_cat[c][i]))
                added = True
        if not added:
            break
        i += 1

    total = args.sheets * args.per_sheet
    ordered = ordered[:total]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    thumb, pad, head, label_w = 190, 6, 30, 190
    for s in range(args.sheets):
        chunk = ordered[s * args.per_sheet:(s + 1) * args.per_sheet]
        if not chunk:
            break
        rows = len(chunk)
        W = label_w + args.per_style * (thumb + pad) + pad
        H = head + rows * (thumb + pad) + pad
        canvas = Image.new("RGB", (W, H), (18, 20, 24))
        draw = ImageDraw.Draw(canvas)
        draw.text((pad, 9), f"候选批次 {s + 1}/{args.sheets}   "
                            f"判读：这一行的 {args.per_style} 张是不是同一款商品？",
                  fill=(240, 243, 250))
        for r, (cat, sku, paths) in enumerate(chunk):
            y = head + pad + r * (thumb + pad)
            draw.text((pad, y + thumb // 2 - 14),
                      f"{s + 1}.{r + 1}  {sku}\n{len(paths)}张",
                      fill=(255, 214, 120))
            step = max(1, len(paths) // args.per_style)
            for c, p in enumerate(paths[::step][:args.per_style]):
                x = label_w + pad + c * (thumb + pad)
                try:
                    with Image.open(p) as im:
                        im = im.convert("RGB")
                        im.thumbnail((thumb - 4, thumb - 4), Image.LANCZOS)
                        canvas.paste(im, (x + (thumb - im.width) // 2,
                                          y + (thumb - im.height) // 2))
                except Exception:                            # noqa: BLE001
                    draw.rectangle([x, y, x + thumb, y + thumb], fill=(48, 30, 30))
            manifest.append({"sheet": s + 1, "slot": r + 1, "sku": sku,
                             "category": cat, "n": len(paths)})
        canvas.save(out_dir / f"cand_{s + 1:02d}.png")
        print(f"  cand_{s + 1:02d}.png  ({len(chunk)} 个候选)")

    with open(out_dir / "candidates.csv", "w", encoding="utf-8-sig",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sheet", "slot", "sku", "category", "n"])
        w.writeheader()
        w.writerows(manifest)

    print(f"\n共 {len(manifest)} 个候选 → {out_dir}/")
    print("按类别分布：")
    cnt = defaultdict(int)
    for m in manifest:
        cnt[cat_names.get(m['category'], m['category'])] += 1
    for k, v in sorted(cnt.items(), key=lambda x: -x[1]):
        print(f"  {k:<12}{v:>4}")
    print("\n判读方式：每行 4 张，若确为同一款商品 → 记为「纯」；明显混了多款 → 记「混」。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
