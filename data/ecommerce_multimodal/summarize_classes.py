#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按"品类"合并聚类簇，算出每个品类的真实规模 —— 这是"选出最多的类"的核心一步。

为什么需要它
    聚类只按视觉相似度分组，同一个品类会散落在多个簇里
    （实测：裤子跨 3 个簇、鞋跨 6 个簇、零食跨 6 个簇）。
    只看单个簇的规模会严重低估品类规模，必须合并。

用法
    # 1) 生成待命名的模板（每个簇一行）
    python summarize_classes.py --init

    # 2) 人工/模型填写 cluster_labels.json 的 cat / zh 字段

    # 3) 汇总
    python summarize_classes.py --min-images 500
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def load_sizes(path: Path):
    """从 cluster_sizes.json 读 {cluster_id: {size, rank, purity, samples}}"""
    if not path.is_file():
        return None
    rep = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for c in rep.get("clusters", []):
        out[int(c["cluster"])] = {
            "size": int(c["size"]), "rank": int(c["rank"]),
            "purity": float(c["purity"]), "samples": c.get("samples", []),
        }
    return out


def sizes_from_csv(path: Path):
    """备选：直接从 clusters.csv 统计每个簇的张数"""
    import csv
    from collections import Counter
    cnt = Counter()
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            cnt[int(r["cluster"])] += 1
    return {c: {"size": n, "rank": 0, "purity": 0.0, "samples": []} for c, n in cnt.items()}


def do_init(sizes, labels_path: Path):
    if labels_path.exists():
        print(f"[跳过] {labels_path} 已存在，未覆盖")
        return 0
    ordered = sorted(sizes.items(), key=lambda kv: -kv[1]["size"])
    obj = {
        "_readme": [
            "把每个簇的 cat / zh 填上，然后用 `python summarize_classes.py` 汇总。",
            "cat  英文小写下划线，统一的一套键（同品类的不同簇必须用同一个 cat）",
            "zh   中文品类名",
            "看不清/太杂的填 unclear，不计入保留清单",
            "size/rank/samples 是聚类结果，仅供参考，不要改",
        ],
        "clusters": {},
    }
    for i, (cid, info) in enumerate(ordered, start=1):
        obj["clusters"][str(cid)] = {
            "rank": info["rank"] or i,
            "size": info["size"],
            "purity": round(info["purity"], 3),
            "samples": info["samples"],
            "cat": "",
            "zh": "",
        }
    labels_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已生成待命名模板 → {labels_path}（{len(ordered)} 个簇）")
    print("请填写每个簇的 cat / zh，然后重跑本脚本汇总。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="按品类合并簇，算真实品类规模")
    ap.add_argument("--labels", default="cluster_labels.json")
    ap.add_argument("--sizes", default="cluster_sizes.json")
    ap.add_argument("--csv", default="clusters.csv", help="没有 cluster_sizes.json 时的备选数据源")
    ap.add_argument("--init", action="store_true", help="生成待命名模板")
    ap.add_argument("--min-images", type=int, default=0,
                    help="只列张数 ≥ N 的品类（用于生成保留清单）")
    ap.add_argument("--top", type=int, default=0, help="只列前 N 个品类")
    args = ap.parse_args()

    labels_path = Path(args.labels)
    sizes = load_sizes(Path(args.sizes)) or sizes_from_csv(Path(args.csv))
    if not sizes:
        print("[错误] 既没有 cluster_sizes.json 也没有 clusters.csv，先跑 embed_cluster.py")
        return 1

    if args.init:
        return do_init(sizes, labels_path)

    if not labels_path.is_file():
        print(f"[错误] 找不到 {labels_path}，先跑 `python summarize_classes.py --init`")
        return 1

    obj = json.loads(labels_path.read_text(encoding="utf-8"))
    clusters = obj.get("clusters", {})
    if isinstance(clusters, list):        # 兼容 {list} 形式
        clusters = {str(c["cluster"]): c for c in clusters}
    # 品类级规范中文名（优先用这个，避免拿某个簇的名字代表整个品类）
    cat_names = obj.get("categories", {}) or {}

    per_cat, unlabeled, unclear = defaultdict(lambda: {"images": 0, "clusters": []}), [], []
    for cid_str, info in clusters.items():
        cid = int(cid_str)
        size = sizes.get(cid, {}).get("size", info.get("size", 0))
        cat = (info.get("cat") or "").strip()
        zh = (info.get("zh") or "").strip()
        if not cat:
            unlabeled.append((cid, size))
            continue
        if cat == "unclear":
            unclear.append((cid, size))
            continue
        per_cat[cat]["zh"] = zh or cat
        per_cat[cat]["images"] += size
        per_cat[cat]["clusters"].append((cid, size))

    total = sum(v["size"] for v in sizes.values())
    ranked = sorted(per_cat.items(), key=lambda kv: -kv[1]["images"])

    print("=" * 84)
    print(f"品类规模（按张数降序）    数据集总计 {total} 张 / {len(sizes)} 个簇")
    print("-" * 84)
    print(f"{'品类':<26}{'簇数':>6}{'张数':>9}{'占比':>8}{'累计':>8}   代表簇")
    print("-" * 84)
    cum = 0
    shown = 0
    for cat, v in ranked:
        if args.min_images and v["images"] < args.min_images:
            continue
        if args.top and shown >= args.top:
            break
        cum += v["images"]
        top_cl = ", ".join(str(c) for c, _ in
                           sorted(v["clusters"], key=lambda x: -x[1])[:4])
        name = cat_names.get(cat) or v["zh"] or cat
        print(f"{name + ' (' + cat + ')':<26}{len(v['clusters']):>6}{v['images']:>9}"
              f"{v['images'] / total * 100:>7.1f}%{cum / total * 100:>7.1f}%   {top_cl}")
        shown += 1
    print("-" * 84)

    if args.min_images:
        kept = [(c, v) for c, v in ranked if v["images"] >= args.min_images]
        n_img = sum(v["images"] for _, v in kept)
        print(f"\n保留标准：张数 ≥ {args.min_images}")
        print(f"  ⇒ 保留 {len(kept)} 个品类 / {len(ranked)} 个，覆盖 {n_img} 张"
              f"（占全部 {n_img / total * 100:.1f}%）")
        print(f"  ⇒ 丢弃长尾 {len(ranked) - len(kept)} 个品类，共 "
              f"{total - n_img - sum(s for _, s in unclear)} 张")

    if unclear:
        print(f"\n标记为 unclear 的簇：{len(unclear)} 个，共 {sum(s for _, s in unclear)} 张"
              f"（这些图不进训练集）")
    if unlabeled:
        print(f"\n[警告] 还有 {len(unlabeled)} 个簇没有填 cat，共 "
              f"{sum(s for _, s in unlabeled)} 张未被统计：")
        print("        " + ", ".join(f"簇{c}({s})" for c, s in
                                      sorted(unlabeled, key=lambda x: -x[1])[:20]))
        print("        请补齐 cluster_labels.json 里的 cat / zh 后重跑。")

    print(f"\n下一步：保留的品类写进 questions.json（按品类写问法），"
          f"再用 classify_tool.html 做款级分组与写回答。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
