#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按 keep_categories.json 过滤出要保留的图片，生成清单。

输入
    clusters.csv          逐图聚类结果（path, cluster, rank, purity）
    cluster_labels.json   簇 → 品类标签
    keep_categories.json  保留/丢弃清单
输出
    kept_images.csv       保留的图片清单（path, cluster, category, purity）
    dropped_images.csv    丢弃的图片清单（含丢弃原因）

用法
    python select_dataset.py --manifest kept_images.csv
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


def main() -> int:
    ap = argparse.ArgumentParser(description="按保留清单过滤图片")
    ap.add_argument("--clusters", default="clusters.csv")
    ap.add_argument("--labels", default="cluster_labels.json")
    ap.add_argument("--keep", default="keep_categories.json")
    ap.add_argument("--manifest", default="kept_images.csv")
    ap.add_argument("--dropped", default="dropped_images.csv")
    args = ap.parse_args()

    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    cl = labels["clusters"]
    cat_names = labels.get("categories", {})
    keep_cfg = json.loads(Path(args.keep).read_text(encoding="utf-8"))
    keep = set(keep_cfg["keep"])
    drop = set(keep_cfg["drop"])
    drop_unclear = bool(keep_cfg.get("drop_unclear", True))
    min_conf = float(keep_cfg.get("drop_low_confidence", 0.0) or 0.0)

    kept_rows, drop_rows = [], []
    per_cat = defaultdict(int)
    reason = defaultdict(int)

    with open(args.clusters, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            cid = r["cluster"]
            info = cl.get(cid)
            if info is None:
                drop_rows.append((r["path"], cid, "", "簇无标签"))
                reason["簇无标签"] += 1
                continue
            cat = info.get("cat", "")
            conf = info.get("conf")
            conf = float(conf) if isinstance(conf, (int, float)) else 1.0

            if cat == "unclear" and drop_unclear:
                drop_rows.append((r["path"], cid, cat, "unclear"))
                reason["unclear"] += 1
            elif cat in drop:
                drop_rows.append((r["path"], cid, cat, "杂项桶"))
                reason["杂项桶"] += 1
            elif cat not in keep:
                drop_rows.append((r["path"], cid, cat, "不在保留清单"))
                reason["不在保留清单"] += 1
            elif conf < min_conf:
                drop_rows.append((r["path"], cid, cat, f"置信度<{min_conf}"))
                reason[f"置信度<{min_conf}"] += 1
            else:
                kept_rows.append((r["path"], cid, cat, r.get("purity", "")))
                per_cat[cat] += 1

    with open(args.manifest, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "cluster", "category", "purity"])
        w.writerows(kept_rows)

    with open(args.dropped, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "cluster", "category", "reason"])
        w.writerows(drop_rows)

    total = len(kept_rows) + len(drop_rows)
    print("=" * 70)
    print(f"保留 {len(kept_rows)} 张 / 丢弃 {len(drop_rows)} 张 / 合计 {total} 张")
    print(f"保留率 {len(kept_rows) / max(1, total) * 100:.1f}%")
    print("=" * 70)
    print(f"\n{'品类':<22}{'张数':>8}{'簇数':>6}")
    print("-" * 40)
    clusters_per_cat = defaultdict(set)
    with open(args.clusters, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            info = cl.get(r["cluster"])
            if info and info.get("cat") in per_cat:
                clusters_per_cat[info["cat"]].add(r["cluster"])
    for cat, n in sorted(per_cat.items(), key=lambda kv: -kv[1]):
        name = cat_names.get(cat, cat)
        print(f"{name + ' (' + cat + ')':<22}{n:>8}{len(clusters_per_cat[cat]):>6}")
    print("-" * 40)
    print(f"{'合计':<22}{sum(per_cat.values()):>8}")

    print("\n丢弃原因分布：")
    for k, v in sorted(reason.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<16}{v:>8}")

    print(f"\n保留清单 → {args.manifest}")
    print(f"丢弃清单 → {args.dropped}")
    print("\n下一步：python cluster_within.py --manifest kept_images.csv --per-sku 40")
    return 0


if __name__ == "__main__":
    sys.exit(main())
