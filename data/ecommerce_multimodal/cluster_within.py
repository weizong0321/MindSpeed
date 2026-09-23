#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
品类内再聚类 → 分出"款"（SKU）。

为什么需要
    外层 k=200 的簇是"品类级"（运动鞋跨 7 个簇），而写答案的单位是"款"
    （同一款的多张图参数一致，可以共用一段回答）。所以要在每个品类内部再聚一次。
    特征已缓存在 embeddings.npy，这一步不需要 GPU 重新提特征。

用法
    python cluster_within.py --manifest kept_images.csv --per-sku 40
    python cluster_within.py --manifest kept_images.csv --per-sku 15 \
        --montage-dir sku_montages --montage-per-category 12

关键参数
    --per-sku  目标"每款多少张图"。越小 → 款越多越纯，但需要写的答案越多。
               经验：15-20 接近"同一商品"；40 更像"同一类商品"。

输出
    assignments.csv        逐图分配（path, category, sku, purity）
    sku_summary.json       每个款的规模/纯度/代表图
    sku_montages/          每个品类下最纯的若个款的拼图，供人工命名
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

import numpy as np

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("[错误] 需要 Pillow")
    sys.exit(1)


def montage(paths, out_path, cols=5, thumb=150, title=""):
    rows = (len(paths) + cols - 1) // cols
    pad, head = 6, 24
    W = cols * thumb + (cols + 1) * pad
    H = rows * thumb + (rows + 1) * pad + head
    canvas = Image.new("RGB", (W, H), (22, 24, 28))
    d = ImageDraw.Draw(canvas)
    d.text((pad, 7), title, fill=(238, 241, 248))
    for i, p in enumerate(paths):
        r, c = divmod(i, cols)
        x = pad + c * (thumb + pad)
        y = head + pad + r * (thumb + pad)
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail((thumb - 4, thumb - 4), Image.LANCZOS)
                canvas.paste(im, (x + (thumb - im.width) // 2,
                                  y + (thumb - im.height) // 2))
        except Exception:                                    # noqa: BLE001
            d.rectangle([x, y, x + thumb, y + thumb], fill=(48, 30, 30))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="品类内聚类分款")
    ap.add_argument("--manifest", default="kept_images.csv")
    ap.add_argument("--clusters", default="clusters.csv")
    ap.add_argument("--emb", default="embeddings.npy")
    ap.add_argument("--labels", default="cluster_labels.json")
    ap.add_argument("--per-sku", type=int, default=40)
    ap.add_argument("--min-sku-size", type=int, default=8)
    ap.add_argument("--out", default="assignments.csv")
    ap.add_argument("--summary", default="sku_summary.json")
    ap.add_argument("--montage-dir", default="")
    ap.add_argument("--montage-per-category", type=int, default=12)
    ap.add_argument("--montage-per-sku", type=int, default=9)
    args = ap.parse_args()

    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    cat_names = labels.get("categories", {})

    # clusters.csv 的行序 == embeddings.npy 的行序（同一份 paths 顺序写出）
    order, row_of = [], {}
    with open(args.clusters, encoding="utf-8-sig") as f:
        for i, r in enumerate(csv.DictReader(f)):
            order.append(r["path"])
            row_of[r["path"]] = i

    by_cat = defaultdict(list)
    with open(args.manifest, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            p = r["path"]
            if p not in row_of:
                print(f"[警告] {p} 不在 clusters.csv 里，跳过")
                continue
            by_cat[r["category"]].append(p)

    emb = np.load(args.emb, mmap_mode="r")
    if emb.shape[0] != len(order):
        print(f"[错误] embeddings 有 {emb.shape[0]} 行，clusters.csv 有 {len(order)} 行，不一致")
        return 1
    print(f"特征 {emb.shape}，匹配 {len(order)} 行")

    from sklearn.cluster import MiniBatchKMeans

    all_rows, summary = [], {}
    print()
    print("=" * 88)
    print(f"{'品类':<20}{'图片':>7}{'款数':>7}{'中位款':>8}{'去尾后':>8}{'需写答案':>9}")
    print("-" * 88)
    total_sku = 0
    for cat, paths in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        idx = np.array([row_of[p] for p in paths])
        X = np.asarray(emb[idx], dtype=np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        ok = norms.ravel() > 1e-6
        X = X[ok] / norms[ok]
        paths_ok = [p for p, k in zip(paths, ok) if k]

        k = max(2, int(round(len(paths_ok) / args.per_sku)))
        km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3,
                             batch_size=2048, max_iter=200)
        lab = km.fit_predict(X)
        cent = km.cluster_centers_
        cent /= np.maximum(np.linalg.norm(cent, axis=1, keepdims=True), 1e-6)
        pur = np.einsum("ij,ij->i", X, cent[lab])

        sizes = np.bincount(lab, minlength=k)
        keep = [c for c in range(k) if sizes[c] >= args.min_sku_size]
        total_sku += len(keep)

        skus = []
        for rank, c in enumerate(sorted(keep, key=lambda c: -sizes[c]), start=1):
            m = np.where(lab == c)[0]
            order_p = m[np.argsort(-pur[m])]
            sku_id = f"{cat}_{rank:03d}"
            skus.append({
                "sku": sku_id, "category": cat, "size": int(sizes[c]),
                "purity": round(float(pur[m].mean()), 3),
                "samples": [Path(paths_ok[i]).name for i in order_p[:5]],
                "paths": [paths_ok[i] for i in order_p],
            })
            for i in m:
                all_rows.append((paths_ok[i], cat, sku_id, round(float(pur[i]), 4)))
        summary[cat] = skus

        if args.montage_dir:
            md = Path(args.montage_dir)
            for s in skus[:args.montage_per_category]:
                montage(s["paths"][:args.montage_per_sku],
                        md / f"{cat}__{s['sku']}.png",
                        title=f"{cat_names.get(cat, cat)} · {s['sku']} · "
                              f"{s['size']}张 · 纯度{s['purity']}")

        med = int(np.median([s["size"] for s in skus])) if skus else 0
        print(f"{cat_names.get(cat, cat) + ' (' + cat + ')':<20}"
              f"{len(paths_ok):>7}{k:>7}{med:>8}{len(keep):>8}{len(keep):>9}")

    print("-" * 88)
    print(f"{'合计':<20}{sum(len(v) for v in by_cat.values()):>7}"
          f"{'':>7}{'':>8}{'':>8}{total_sku:>9}")
    print("=" * 88)
    print(f"\nper-sku 目标 {args.per_sku} 张/款 → 共 {total_sku} 个款")
    print(f"如果要给每个款写一段回答，需要写 {total_sku} 段。")

    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "category", "sku", "purity"])
        w.writerows(all_rows)
    print(f"\n逐图分配 → {args.out}（{len(all_rows)} 条）")

    slim = {c: [{k2: v2 for k2, v2 in s.items() if k2 != "paths"} for s in v]
            for c, v in summary.items()}
    Path(args.summary).write_text(
        json.dumps({"per_sku": args.per_sku,
                    "categories": {c: cat_names.get(c, c) for c in summary},
                    "skus": slim}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"款汇总 → {args.summary}")
    if args.montage_dir:
        print(f"款拼图 → {args.montage_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
