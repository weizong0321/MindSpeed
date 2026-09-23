#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无标注数据集：视觉嵌入 + 聚类，发现"有哪些品类"并给出各品类规模。
不依赖任何需要新安装的包（只用 torch / torchvision / sklearn / numpy / Pillow）。

思路
    142k 张无标注图 → 预训练视觉骨干提特征 → 归一化 → MiniBatchKMeans 聚类
    → 每个簇输出一张缩略图拼图（montage），人眼扫一遍就能给簇命名
    → 按簇大小排序，保留最大的若干类，丢弃长尾与杂项

为什么不用分类器
    没有标注就无法训练分类器；而"品类很多且清单未知"时，聚类是唯一能在
    不知道答案的情况下发现类别的办法。命名只需人看 montage，每个簇几秒钟。

用法
    # 1) 快速试跑（2000 张，验证流程与显存占用）
    python embed_cluster.py --root ./images --limit 2000 --k 40

    # 2) 全量提特征 + 聚类（特征会缓存，中断可续）
    python embed_cluster.py --root ./images --k 200 --montage-dir ./montages

    # 3) 只看规模分布，不重新提特征
    python embed_cluster.py --root ./images --k 200 --reuse

输出
    embeddings.npy        特征缓存（fp16）
    clusters.csv          path, cluster, purity（到簇心的余弦相似度）
    cluster_sizes.json    各簇规模 + 纯度 + 代表图
    montages/cluster_XXX.png   每簇的代表图拼图，用于人工命名
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from torchvision.models import (
        resnet50, ResNet50_Weights,
        convnext_base, ConvNeXt_Base_Weights,
        vit_b_16, ViT_B_16_Weights,
        swin_b, Swin_B_Weights,
    )
    from PIL import Image, ImageDraw
except ImportError as e:
    print(f"[错误] 缺少依赖：{e}")
    sys.exit(1)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

BACKBONES = {
    "resnet50":     (resnet50,      ResNet50_Weights.IMAGENET1K_V2,       2048, 224),
    "convnext_base": (convnext_base, ConvNeXt_Base_Weights.IMAGENET1K_V1, 1024, 224),
    "vit_b_16":     (vit_b_16,      ViT_B_16_Weights.IMAGENET1K_V1,       768,  224),
    "swin_b":       (swin_b,        Swin_B_Weights.IMAGENET1K_V1,         1024, 224),
}


# ---------------------------------------------------------------- 数据加载
def build_transform(size):
    return transforms.Compose([
        transforms.Resize(int(size * 1.14), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_one(path, tf):
    try:
        with Image.open(path) as im:
            return tf(im.convert("RGB"))
    except Exception:                                        # noqa: BLE001
        return None


def batch_iter(paths, bs):
    for i in range(0, len(paths), bs):
        yield i, paths[i:i + bs]


# ---------------------------------------------------------------- 主干
def build_model(name, device):
    fn, weights, dim, size = BACKBONES[name]
    model = fn(weights=weights)
    if name == "resnet50":
        model.fc = nn.Identity()
    elif name == "convnext_base":
        model.classifier[2] = nn.Identity()
    elif name == "vit_b_16":
        model.heads = nn.Identity()
    elif name == "swin_b":
        model.head = nn.Identity()
    model.eval().to(device)
    return model, dim, size


@torch.no_grad()
def extract(paths, model, tf, device, bs, pool, tag=""):
    feats = []
    t0 = time.time()
    for start, chunk in batch_iter(paths, bs):
        tensors = list(pool.map(lambda p: load_one(p, tf), chunk))
        idx = [i for i, t in enumerate(tensors) if t is not None]
        if not idx:
            feats.append(np.zeros((len(chunk), 1), dtype=np.float16))
            continue
        batch = torch.stack([tensors[i] for i in idx]).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            out = model(batch)
        out = out.float()
        # 载入失败的用零向量占位，后面会被标记出来
        full = torch.zeros((len(chunk), out.shape[1]), dtype=torch.float32)
        full[idx] = out.cpu()
        feats.append(full.numpy().astype(np.float16))
        done = start + len(chunk)
        speed = done / max(1e-6, time.time() - t0)
        eta = (len(paths) - done) / max(1e-6, speed)
        print(f"  {tag}特征 {done}/{len(paths)}  {speed:6.1f} 张/秒  剩余 {eta/60:5.1f} 分钟",
              end="\r")
    print(" " * 70, end="\r")
    return np.concatenate(feats, axis=0)


# ---------------------------------------------------------------- 拼图
def make_montage(paths, out_path, cols=5, thumb=160, title="", footer=""):
    rows = (len(paths) + cols - 1) // cols
    pad, header = 6, 26
    W = cols * thumb + (cols + 1) * pad
    H = rows * thumb + (rows + 1) * pad + header + 18
    canvas = Image.new("RGB", (W, H), (24, 26, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 7), title, fill=(235, 238, 245))
    for i, p in enumerate(paths):
        r, c = divmod(i, cols)
        x = pad + c * (thumb + pad)
        y = header + pad + r * (thumb + pad)
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail((thumb, thumb), Image.LANCZOS)
                canvas.paste(im, (x + (thumb - im.width) // 2,
                                  y + (thumb - im.height) // 2))
        except Exception:                                    # noqa: BLE001
            draw.rectangle([x, y, x + thumb, y + thumb], outline=(90, 40, 40))
        draw.text((x + 3, y + thumb - 13), Path(p).name[:20], fill=(255, 220, 120))
    draw.text((pad, H - 15), footer, fill=(150, 160, 175))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description="视觉嵌入 + 聚类，发现无标注数据集的品类")
    ap.add_argument("--root", required=True)
    ap.add_argument("--backbone", default="resnet50", choices=list(BACKBONES))
    ap.add_argument("--k", type=int, default=200, help="聚类簇数")
    ap.add_argument("--k-list", default="", help="扫多个 k，如 50,100,200,400（只报告规模分布）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0, help="跳过前 N 张（配合 --limit 抽样）")
    ap.add_argument("--bs", type=int, default=128, help="推理批大小")
    ap.add_argument("--load-workers", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cache", default="embeddings.npy")
    ap.add_argument("--reuse", action="store_true", help="复用已有特征缓存")
    ap.add_argument("--pca", type=int, default=0, help="先降维到 N 维再聚类（0=不降）")
    ap.add_argument("--montage-dir", default="", help="输出每簇拼图的目录")
    ap.add_argument("--montage-per-cluster", type=int, default=25)
    ap.add_argument("--montage-clusters", type=int, default=60,
                    help="只为最大的前 N 个簇出拼图（默认 60）")
    ap.add_argument("--csv", default="clusters.csv")
    ap.add_argument("--report", default="cluster_sizes.json")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    cache = Path(args.cache)
    if not root.is_dir():
        print(f"[错误] 目录不存在：{root}")
        return 1

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))

    paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    # 文件名全是数字时必须按数值排序 —— 字符串排序会把 100000 排在 10001 前面，
    # 导致 --limit 取到的是一个数值上很窄的片段，样例不具代表性。
    if paths and all(p.stem.isdigit() for p in paths):
        paths.sort(key=lambda p: int(p.stem))
        sort_mode = "数值"
    else:
        paths.sort()
        sort_mode = "字典"
    total_found = len(paths)
    if args.offset:
        paths = paths[args.offset:]
    if args.limit:
        paths = paths[:args.limit]
    print(f"目录内共 {total_found} 张（按{sort_mode}排序），"
          f"本次 offset={args.offset} limit={args.limit or '全部'} → {len(paths)} 张")
    if not paths:
        print("[错误] 没找到图片")
        return 1
    print(f"图片 {len(paths)} 张")
    print(f"设备 {device}  主干 {args.backbone}  批 {args.bs}")

    # ---------- 特征 ----------
    if args.reuse and cache.is_file():
        emb = np.load(cache)
        if len(emb) != len(paths):
            print(f"[错误] 缓存里有 {len(emb)} 条特征，但当前有 {len(paths)} 张图。"
                  f"换 --cache 文件名或去掉 --reuse。")
            return 1
        print(f"复用特征缓存 {cache}（{emb.shape}）")
    else:
        model, dim, size = build_model(args.backbone, device)
        tf = build_transform(size)
        print(f"特征维度 {dim}，输入 {size}×{size}")
        pool = ThreadPoolExecutor(max_workers=args.load_workers)
        t0 = time.time()
        emb = extract(paths, model, tf, device, args.bs, pool)
        pool.shutdown()
        print(f"提特征完成，用时 {(time.time()-t0)/60:.1f} 分钟，形状 {emb.shape}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, emb)
        print(f"已缓存 → {cache}")

    # 归一化；全零向量 = 读取失败的图片
    X = emb.astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    bad = np.where(norms.ravel() < 1e-6)[0]
    if len(bad):
        print(f"[警告] {len(bad)} 张图读取失败（特征为零向量），将从聚类中排除")
    ok_idx = np.where(norms.ravel() >= 1e-6)[0]
    Xn = X[ok_idx] / norms[ok_idx]

    if args.pca and args.pca < Xn.shape[1]:
        from sklearn.decomposition import PCA
        print(f"PCA 降维 {Xn.shape[1]} → {args.pca} ...")
        Xn = PCA(n_components=args.pca, random_state=0).fit_transform(Xn).astype(np.float32)
        Xn /= np.maximum(np.linalg.norm(Xn, axis=1, keepdims=True), 1e-6)

    # ---------- 聚类 ----------
    from sklearn.cluster import MiniBatchKMeans
    ks = [int(x) for x in args.k_list.split(",") if x.strip()] or [args.k]

    if len(ks) > 1:
        print("\n扫 k（只报告规模分布，不写拼图）：")
        for k in ks:
            km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3,
                                 batch_size=4096, max_iter=100)
            lab = km.fit_predict(Xn)
            cnt = np.bincount(lab, minlength=k)
            cnt.sort()
            cnt = cnt[::-1]
            top = cnt[:max(1, k // 10)]
            print(f"  k={k:<5} 最大簇 {cnt[0]:>6}  前10%簇合计 "
                  f"{top.sum():>7} ({top.sum()/len(lab)*100:4.1f}%)  "
                  f"中位簇 {int(np.median(cnt)):>5}  最小簇 {cnt[-1]:>5}")
        print("\n提示：k 越大簇越纯但越碎；选“中位簇 ≈ 每类期望图片数”的那个 k。")
        return 0

    k = ks[0]
    print(f"\nMiniBatchKMeans  k={k} ...")
    t0 = time.time()
    km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3,
                         batch_size=4096, max_iter=300)
    labels = km.fit_predict(Xn)
    print(f"聚类完成，用时 {time.time()-t0:.1f} 分钟")

    # ---------- 纯度与统计 ----------
    cent = km.cluster_centers_
    cent /= np.maximum(np.linalg.norm(cent, axis=1, keepdims=True), 1e-6)
    purity = np.einsum("ij,ij->i", Xn, cent[labels])

    full_lab = np.full(len(paths), -1, dtype=np.int32)
    full_lab[ok_idx] = labels
    full_pur = np.zeros(len(paths), dtype=np.float32)
    full_pur[ok_idx] = purity

    counts = Counter(labels.tolist())
    order = [c for c, _ in counts.most_common()]

    print()
    print("=" * 78)
    print("簇规模分布（按大小降序）")
    print("-" * 78)
    sizes = [counts[c] for c in order]
    cum = np.cumsum(sizes)
    for rank, c in enumerate(order[:30], start=1):
        m = labels == c
        print(f"  #{rank:<3} 簇 {c:<5} {counts[c]:>6} 张  平均纯度 {purity[m].mean():.3f}"
              f"  占累计 {cum[rank-1]/len(labels)*100:5.1f}%")
    print(f"  ... 共 {k} 个簇")
    print(f"  最大簇 {sizes[0]} 张 / 最小簇 {sizes[-1]} 张 / 中位 {int(np.median(sizes))} 张")
    for pct in (50, 80, 90, 95):
        n = int(np.searchsorted(cum, len(labels) * pct / 100) + 1)
        print(f"  覆盖 {pct}% 的图片需要 {n} 个簇")

    # 相邻文件是否同簇（判断原始来源是否按品类顺序排列）
    if len(labels) > 1:
        same = float((labels[1:] == labels[:-1]).mean())
        print(f"\n  相邻文件同簇比例：{same*100:.1f}%"
              + ("  ← 明显高于随机，说明源数据顺序里带品类聚集" if same > 3 / k else ""))

    # ---------- 拼图 ----------
    if args.montage_dir:
        print(f"\n生成拼图（前 {args.montage_clusters} 个簇）...")
        mdir = Path(args.montage_dir)
        for rank, c in enumerate(order[:args.montage_clusters], start=1):
            idxs = np.where(labels == c)[0]
            pick = idxs[np.argsort(-purity[idxs])][:args.montage_per_cluster]
            sel = [str(paths[ok_idx[i]]) for i in pick]
            title = (f"rank {rank}  簇 {c}  {counts[c]} 张  "
                     f"纯度 {purity[labels == c].mean():.3f}")
            make_montage(sel, mdir / f"rank{rank:03d}_cluster{c:04d}.png",
                         title=title,
                         footer=f"代表图 {len(sel)} 张（按与簇心相似度取最像的）")
            print(f"  rank {rank}/{min(args.montage_clusters, len(order))} "
                  f"簇 {c} ({counts[c]} 张)", end="\r")
        print(" " * 60, end="\r")
        print(f"拼图已写入 {mdir}")

    # ---------- 输出 ----------
    if args.csv:
        import csv as _csv
        with open(args.csv, "w", encoding="utf-8-sig", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["path", "cluster", "rank", "purity"])
            rank_of = {c: i + 1 for i, c in enumerate(order)}
            for i, p in enumerate(paths):
                lab = int(full_lab[i])
                w.writerow([str(p), lab, rank_of.get(lab, -1), round(float(full_pur[i]), 4)])
        print(f"逐图聚类结果 → {args.csv}")

    if args.report:
        rep = {
            "root": str(root), "backbone": args.backbone, "k": k, "total": len(paths),
            "clustered": int(len(ok_idx)), "failed": int(len(bad)),
            "max_cluster": int(sizes[0]), "min_cluster": int(sizes[-1]),
            "median_cluster": int(np.median(sizes)),
            "clusters": [
                {"rank": i + 1, "cluster": int(c), "size": int(counts[c]),
                 "purity": round(float(purity[labels == c].mean()), 4),
                 "samples": [Path(paths[ok_idx[j]]).name
                             for j in np.where(labels == c)[0][:5]]}
                for i, c in enumerate(order)
            ],
        }
        Path(args.report).write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
        print(f"聚类报告 → {args.report}")

    print("\n下一步：打开 montages/ 给最大的若干簇命名 → 得到新的类别清单 →")
    print("        写进 questions.json，再用 classify_tool.html 做『款』级分组与写回答。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
