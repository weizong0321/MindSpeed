#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用「严格阈值连通分量」代替 KMeans 做同款分组。

思路
    KMeans/簇心纯度衡量的是"看起来像"，8 双不同的白鞋可以聚得很紧 —— 所以做不出"同款"。
    改用：把簇内每张图连到所有相似度 ≥ τ 的邻居，取连通分量。
    · 同一款商品的不同照片相似度通常很高 → 连成一组
    · 不同商品之间相似度低 → 断开，自然分到不同组
    代价是组会更小，但"同款"是构造性保证的。

用法
    python strict_group.py --taus 0.85,0.90,0.95 --report       # 在已知 30 个子样式上看分组行为
    python strict_group.py --tau 0.90 --min-size 6 --out strict_groups.csv   # 对全部子样式分组
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import numpy as np

PURE = ["watch_001", "watch_010", "watch_086", "watch_108", "watch_109",
        "watch_116", "watch_127", "watch_146",
        "bag_backpack_001", "bag_backpack_044", "bag_backpack_119", "bag_backpack_155",
        "shoes_sneaker_123", "shoes_sneaker_176"]
MIXED = ["watch_037", "watch_140",
         "bag_backpack_018", "bag_backpack_052", "bag_backpack_096",
         "bag_backpack_110", "bag_backpack_149",
         "shoes_sneaker_012", "shoes_sneaker_031", "shoes_sneaker_054",
         "shoes_sneaker_080", "shoes_sneaker_166", "shoes_sneaker_168",
         "shoes_sneaker_174"]


def components(S, tau):
    """相似度矩阵 -> 连通分量（标签数组）"""
    n = S.shape[0]
    adj = S >= tau
    np.fill_diagonal(adj, False)
    lab = -np.ones(n, dtype=int)
    c = 0
    for i in range(n):
        if lab[i] >= 0:
            continue
        stack, lab[i] = [i], c
        while stack:
            u = stack.pop()
            for v in np.where(adj[u] & (lab < 0))[0]:
                lab[v] = c
                stack.append(v)
        c += 1
    return lab, c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="embeddings.npy")
    ap.add_argument("--clusters", default="clusters.csv")
    ap.add_argument("--assign", default="assignments.csv")
    ap.add_argument("--taus", default="0.85,0.90,0.95")
    ap.add_argument("--tau", type=float, default=0.90)
    ap.add_argument("--min-size", type=int, default=6)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--out", default="strict_groups.csv")
    args = ap.parse_args()

    row_of = {}
    with open(args.clusters, encoding="utf-8-sig") as f:
        for i, r in enumerate(csv.DictReader(f)):
            row_of[r["path"]] = i
    emb = np.load(args.emb, mmap_mode="r")

    by_sku = defaultdict(list)
    with open(args.assign, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            by_sku[r["sku"]].append(r["path"])

    def feats(paths):
        idx = [row_of[p] for p in paths if p in row_of]
        if len(idx) < 2:
            return None, None
        X = np.asarray(emb[idx], dtype=np.float32)
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        ok = nrm.ravel() > 1e-6
        X = X[ok] / nrm[ok]
        return X, [p for p, k in zip(idx, ok) if k]

    if args.report:
        taus = [float(t) for t in args.taus.split(",")]
        print("=" * 96)
        print("在人工核验的 28 个子样式上看「严格阈值连通分量」的分组行为")
        print("-" * 96)
        print(f"{'子样式':<28}{'张数':>5}{'人工':>6}" +
              "".join(f"{'τ=' + str(t):>16}" for t in taus))
        print(f"{'':<28}{'':>5}{'':>6}" +
              "".join(f"{'组数/最大组':>16}" for _ in taus))
        stats = {t: {"纯": [], "混": []} for t in taus}
        for sku in PURE + MIXED:
            paths = by_sku.get(sku) or []
            X, _ = feats(paths)
            if X is None or len(X) < 4:
                continue
            S = X @ X.T
            label = "纯" if sku in PURE else "混"
            cells = []
            for t in taus:
                lab, c = components(S, t)
                sizes = np.bincount(lab)
                big = int(sizes.max())
                stats[t][label].append((c, big))
                cells.append(f"{c:>7}/{big:<8}")
            print(f"{sku:<28}{len(X):>5}{label:>6}" + "".join(cells))
        print("-" * 96)
        for t in taus:
            p = stats[t]["纯"]
            m = stats[t]["混"]
            if not p or not m:
                continue
            gp = sum(x[0] for x in p) / len(p)
            gm = sum(x[0] for x in m) / len(m)
            print(f"  τ={t}: 纯样本平均组数 {gp:.2f}（{len(p)} 个）   "
                  f"混样本平均组数 {gm:.2f}（{len(m)} 个）   比值 {gm / max(gp, 1e-9):.2f}")
        print("\n  判读：混样本的组数应明显多于纯样本；比值越大，该 τ 越能自动识别混合子样式")
        return 0

    # ---- 对全部子样式分组 ----
    print(f"按 τ={args.tau}、最小组 {args.min_size} 张 对全部子样式分组 ...")
    out_rows, n_grp, n_img = [], 0, 0
    for sku, paths in by_sku.items():
        X, keep = feats(paths)
        if X is None or len(X) < args.min_size:
            continue
        lab, c = components(X @ X.T, args.tau)
        sizes = np.bincount(lab)
        gi = 0
        for g in np.argsort(-sizes):
            if sizes[g] < args.min_size:
                continue
            gi += 1
            gid = f"{sku}_g{gi}"
            for j in np.where(lab == g)[0]:
                out_rows.append({"path": paths[keep[j]], "sku": gid,
                                 "src_sku": sku, "group_size": int(sizes[g])})
            n_grp += 1
            n_img += int(sizes[g])
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "sku", "src_sku", "group_size"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"→ {args.out}：{n_grp} 个同款组 / {n_img} 张图")
    szs = np.array([r["group_size"] for r in out_rows])
    print(f"  组大小：中位 {int(np.median(szs))}，最大 {szs.max()}，"
          f"≥10 张的组 {len({r['sku'] for r in out_rows if r['group_size'] >= 10})} 个")
    return 0


if __name__ == "__main__":
    sys.exit(main())
