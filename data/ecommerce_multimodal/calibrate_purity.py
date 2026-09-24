#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标定"自动纯度分"：用手工核验过的 30 个子样式当基准，找出能区分「纯 / 混」的指标。

背景
    之前的选样按"到簇心的平均余弦相似度(purity)"排序，但那个指标只衡量"看起来像"，
    结果 shoes_sneaker_031 一个子样式里塞了 8 款不同的鞋。
    现在有 30 个**人工看联络表核验过**的子样式，可以拿来当标定集。

候选指标
    mean_pair   簇内两两余弦相似度的均值
    sil2        把簇内强行分成 2 组时的轮廓系数（越高说明簇内其实有两坨）
    sil3        分成 3 组的轮廓系数
    d_to_cent   到簇心的平均相似度（旧指标，作对照）

用法
    python calibrate_purity.py --top 120 --out purity_ranking.csv
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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 人工核验结论（来自 coherence_*.png 逐行判定）
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
BORDER = ["bag_backpack_007", "shoes_sneaker_041"]


def metrics(X):
    """X: 已归一化的 (n,d) 特征"""
    n = len(X)
    if n < 6:
        return None
    S = X @ X.T
    iu = np.triu_indices(n, k=1)
    mean_pair = float(S[iu].mean())
    d_to_cent = float((X * X.mean(axis=0, keepdims=True)).sum(axis=1).mean())
    out = {"mean_pair": mean_pair, "d_to_cent": d_to_cent}
    for k in (2, 3):
        if n < k + 2:
            out[f"sil{k}"] = 0.0
            continue
        lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
        try:
            out[f"sil{k}"] = float(silhouette_score(X, lab, metric="cosine"))
        except Exception:                                    # noqa: BLE001
            out[f"sil{k}"] = 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="embeddings.npy")
    ap.add_argument("--clusters", default="clusters.csv")
    ap.add_argument("--assign", default="assignments.csv")
    ap.add_argument("--top", type=int, default=120)
    ap.add_argument("--out", default="purity_ranking.csv")
    ap.add_argument("--min-size", type=int, default=8)
    args = ap.parse_args()

    # clusters.csv 的行序 == embeddings.npy 的行序
    row_of = {}
    with open(args.clusters, encoding="utf-8-sig") as f:
        for i, r in enumerate(csv.DictReader(f)):
            row_of[r["path"]] = i
    emb = np.load(args.emb, mmap_mode="r")
    print(f"特征 {emb.shape}，索引 {len(row_of)} 条")

    by_sku = defaultdict(list)
    with open(args.assign, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            by_sku[r["sku"]].append(r["path"])
    print(f"子样式共 {len(by_sku)} 个")

    def feats(paths):
        idx = [row_of[p] for p in paths if p in row_of]
        if len(idx) < 6:
            return None
        X = np.asarray(emb[idx], dtype=np.float32)
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        ok = nrm.ravel() > 1e-6
        X = X[ok] / nrm[ok]
        return X if len(X) >= 6 else None

    # ---------- 1) 标定 ----------
    print("\n" + "=" * 92)
    print("标定：人工核验的 30 个子样式")
    print("-" * 92)
    print(f"{'子样式':<30}{'张数':>5}{'mean_pair':>11}{'sil2':>9}{'sil3':>9}"
          f"{'d_to_cent':>11}   人工判定")
    rows = []
    for sku in PURE + MIXED + BORDER:
        paths = by_sku.get(sku) or []
        X = feats(paths)
        if X is None:
            print(f"{sku:<30}{len(paths):>5}  特征不足")
            continue
        m = metrics(X)
        label = ("纯" if sku in PURE else "混" if sku in MIXED else "边界")
        rows.append({"sku": sku, "n": len(X), "label": label, **m})
        print(f"{sku:<30}{len(X):>5}{m['mean_pair']:>11.3f}{m['sil2']:>9.3f}"
              f"{m['sil3']:>9.3f}{m['d_to_cent']:>11.3f}   {label}")

    # ---------- 2) 哪个指标最能区分 ----------
    print("\n" + "=" * 92)
    print("指标区分能力（纯 vs 混，边界样本不计入）")
    print("-" * 92)
    lab = np.array([r["label"] for r in rows])
    strict = lab != "边界"
    best = None
    for key in ("mean_pair", "sil2", "sil3", "d_to_cent"):
        v = np.array([r[key] for r in rows])[strict]
        y = (lab[strict] == "混").astype(int)
        # 用该指标预测"混"：sil 越大越混；mean_pair/d_to_cent 越小越混
        score = v if key.startswith("sil") else -v
        order = np.argsort(-score)
        # 最优阈值下的准确率
        acc = max(((score >= t).astype(int) == y).mean()
                  for t in np.unique(score))
        pure_v = v[y == 0]
        mixed_v = v[y == 1]
        gap = abs(pure_v.mean() - mixed_v.mean()) / (v.std() + 1e-9)
        print(f"  {key:<11} 纯均值 {pure_v.mean():+.3f}  混均值 {mixed_v.mean():+.3f}"
              f"   最优阈值准确率 {acc * 100:5.1f}%   标准化间隔 {gap:.2f}")
        if best is None or acc > best[1]:
            best = (key, acc)

    print(f"\n  ⇒ 最佳指标：{best[0]}（准确率 {best[1] * 100:.1f}%）")

    # ---------- 3) 给全部子样式打分排序 ----------
    key = best[0]
    print(f"\n对全部 {len(by_sku)} 个子样式按 {key} 打分 ...")
    allrows = []
    for sku, paths in by_sku.items():
        if len(paths) < args.min_size:
            continue
        X = feats(paths)
        if X is None:
            continue
        m = metrics(X)
        if m is None:
            continue
        allrows.append({"sku": sku, "n": len(X), **m,
                        "known": ("纯" if sku in PURE else
                                  "混" if sku in MIXED else
                                  "边界" if sku in BORDER else "")})
    score = lambda r: r["sil2"] if key.startswith("sil") else -r[key]
    allrows.sort(key=score)
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rank", "sku", "n", "mean_pair",
                                          "sil2", "sil3", "d_to_cent", "known"])
        w.writeheader()
        for i, r in enumerate(allrows, 1):
            w.writerow({"rank": i, **r})
    print(f"排序结果 → {args.out}（{len(allrows)} 个子样式）")

    print(f"\n最像「纯」的前 {args.top} 个（前 20 展示）")
    print("-" * 78)
    for i, r in enumerate(allrows[:20], 1):
        tag = f"  ← 人工判定：{r['known']}" if r["known"] else ""
        print(f"  #{i:<4}{r['sku']:<32}{r['n']:>4}张  mean_pair {r['mean_pair']:.3f}"
              f"  sil2 {r['sil2']:.3f}{tag}")
    known_in = [r for r in allrows[:args.top] if r["known"] == "纯"]
    print(f"\n  前 {args.top} 名里包含 {len(known_in)} 个已知纯样本"
          f"（共 {len(PURE)} 个）→ 说明排序{'有效' if len(known_in) >= 10 else '无效'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
