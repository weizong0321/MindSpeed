#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 cluster_categories.json（按 rank 组织的标注）转成 cluster_labels.json
（按 cluster id 索引，供 summarize_classes.py 使用），并做独立校验。

校验项
    1. 标注条数 == 簇数
    2. cluster id 唯一、且与 cluster_sizes.json / clusters.csv 完全一致
    3. 标注里的 size 与从 clusters.csv 真实统计的张数一致
    4. size 之和 == 图片总数
    5. rank 1..N 连续无缺
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="cluster_categories.json")
    ap.add_argument("--csv", default="clusters.csv")
    ap.add_argument("--out", default="cluster_labels.json")
    args = ap.parse_args()

    src, csvp, out = Path(args.src), Path(args.csv), Path(args.out)
    if not src.is_file():
        print(f"[错误] 找不到 {src}")
        return 1

    data = json.loads(src.read_text(encoding="utf-8"))
    entries = data["clusters"] if isinstance(data, dict) else data
    print(f"读入 {len(entries)} 条标注（{src}）")

    # ---- 从 CSV 独立统计真实张数 ----
    truth = Counter()
    with open(csvp, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            truth[int(r["cluster"])] += 1
    total_rows = sum(truth.values())
    print(f"从 {csvp} 统计：{len(truth)} 个簇，共 {total_rows} 行")

    problems = []
    seen, ranks = {}, []
    for e in entries:
        cid, rk, sz = int(e["cluster"]), int(e["rank"]), int(e.get("size", -1))
        if cid in seen:
            problems.append(f"簇 {cid} 重复出现（rank {seen[cid]} 与 {rk}）")
        seen[cid] = rk
        ranks.append(rk)
        real = truth.get(cid)
        if real is None:
            problems.append(f"簇 {cid} 在 clusters.csv 里不存在")
        elif sz != real:
            problems.append(f"簇 {cid} 标注 size={sz}，实际 {real}")

    # ---- 校验结论 ----
    print()
    print("=" * 72)
    print("独立校验")
    print("-" * 72)
    print(f"  标注条数        : {len(entries)}")
    print(f"  真实簇数        : {len(truth)}   {'一致' if len(entries) == len(truth) else '不一致 ←'}")
    print(f"  cluster id 唯一 : {'是' if len(seen) == len(entries) else '否 ←'}")
    print(f"  id 集合一致     : {'是' if set(seen) == set(truth) else '否 ←'}")
    miss = sorted(set(range(1, max(ranks) + 1)) - set(ranks)) if ranks else []
    print(f"  rank 连续       : {'是（1..' + str(max(ranks)) + '）' if not miss else '否，缺 ' + str(miss[:10])}")
    size_sum = sum(int(e.get("size", 0)) for e in entries)
    print(f"  张数之和        : {size_sum}   {'== CSV 总行数' if size_sum == total_rows else '!= CSV 总行数 ←'}")
    if problems:
        print(f"\n  发现 {len(problems)} 个问题：")
        for p in problems[:20]:
            print(f"    - {p}")
    else:
        print("\n  未发现问题：标注的簇编号与张数与 clusters.csv 完全吻合。")

    # ---- 转换 ----
    sizes = {int(c["cluster"]): c for c in
             json.loads(Path("cluster_sizes.json").read_text(encoding="utf-8"))["clusters"]} \
        if Path("cluster_sizes.json").is_file() else {}

    obj = {
        "_readme": [
            "由 build_cluster_labels.py 从 cluster_categories.json 自动生成，",
            "字段：cat=英文品类键，zh=中文品类名，conf=标注置信度（<0.5 建议人工复核）。",
            "categories 是品类级的规范中文名（合并簇后显示用，避免用某个簇的名字代表整个品类）。",
            "修改后重跑：python summarize_classes.py --min-images 2000",
        ],
        "categories": {s["category_key"]: s["category_zh"]
                       for s in (data.get("summary") or []) if isinstance(data, dict)},
        "clusters": {},
    }
    for e in sorted(entries, key=lambda x: int(x["cluster"])):
        cid = str(int(e["cluster"]))
        obj["clusters"][cid] = {
            "rank": int(e["rank"]),
            "size": truth.get(int(e["cluster"]), int(e.get("size", 0))),
            "purity": round(float(sizes.get(int(e["cluster"]), {}).get("purity", 0.0)), 3),
            "samples": sizes.get(int(e["cluster"]), {}).get("samples", []),
            "cat": e.get("category_key", ""),
            "zh": e.get("category_zh", ""),
            "conf": e.get("confidence", None),
        }

    out.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已生成 → {out}（{len(obj['clusters'])} 个簇）")

    low = [e for e in entries if (e.get("confidence") or 1) < 0.5]
    unclear = [e for e in entries if e.get("category_key") == "unclear"]
    print(f"  低置信度（<0.5）: {len(low)} 个簇，"
          f"{sum(int(e.get('size', 0)) for e in low)} 张")
    print(f"  unclear         : {len(unclear)} 个簇，"
          f"{sum(int(e.get('size', 0)) for e in unclear)} 张")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
