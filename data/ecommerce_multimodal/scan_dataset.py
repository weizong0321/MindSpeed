#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集审计：不依赖任何深度学习框架，只用 Pillow + 标准库。
回答三个问题：这批图能不能用、能留下多少、有多少是重复的。

用法
    python scan_dataset.py --root ./images
    python scan_dataset.py --root ./images --limit 2000      # 快速试跑
    python scan_dataset.py --root ./images --no-hash         # 跳过重复检测（更快）
    python scan_dataset.py --root ./images --csv sizes.csv   # 额外导出每张图的明细

输出
    stdout 报告 + scan_report.json（可选 sizes.csv）
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

try:
    from PIL import Image
except ImportError:
    print("[错误] 需要 Pillow：pip install Pillow")
    sys.exit(1)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")

# 训练侧约束（来自 qwen3_5_0.8B_config.yaml）
AREA_MAX = 262144        # image_max_pixels = 512x512，超过会被等比压缩
SHORT_TIERS = (224, 336, 448, 512)   # 短边分档
DUP_HAMMING = 0          # 0 = 只算完全相同的 dhash；4 = 近似重复


def dhash(path: Path, size: int = 8):
    """感知哈希（64bit）。用 draft() 让 JPEG 走 DCT 缩放，快很多。"""
    with Image.open(path) as im:
        im.draft("L", (64, 64))
        g = im.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = np.asarray(g).ravel().tolist()
    bits = 0
    w = size + 1
    for r in range(size):
        row = r * w
        for c in range(size):
            bits = (bits << 1) | (1 if px[row + c] > px[row + c + 1] else 0)
    return bits


def probe(path: Path):
    """读一张图的元信息；返回 dict 或 (None, 错误信息)"""
    try:
        with Image.open(path) as im:
            w, h = im.size
            fmt = im.format or path.suffix.lstrip(".").upper()
            mode = im.mode
        return {"path": str(path), "w": w, "h": h, "fmt": fmt, "mode": mode,
                "bytes": path.stat().st_size}
    except Exception as e:                                   # noqa: BLE001
        return {"path": str(path), "error": f"{type(e).__name__}: {e}"}


def quantile(sorted_vals, q):
    if not sorted_vals:
        return 0
    i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


def main() -> int:
    ap = argparse.ArgumentParser(description="数据集审计（无深度学习依赖）")
    ap.add_argument("--root", required=True)
    ap.add_argument("--workers", type=int, default=min(32, (os.cpu_count() or 4) * 2))
    ap.add_argument("--limit", type=int, default=0, help="只扫描前 N 张（试跑用）")
    ap.add_argument("--no-hash", action="store_true", help="跳过感知哈希重复检测")
    ap.add_argument("--csv", default="", help="导出每张图明细的 CSV 路径")
    ap.add_argument("--report", default="", help="导出 JSON 报告的路径")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"[错误] 目录不存在：{root}")
        return 1

    print(f"扫描目录：{root}")
    files = [p for p in sorted(root.rglob("*"))
             if p.is_file() and p.suffix.lower() in IMG_EXTS]
    other = [p for p in root.rglob("*")
             if p.is_file() and p.suffix.lower() not in IMG_EXTS]
    if args.limit:
        files = files[:args.limit]
        print(f"[试跑] 只扫描前 {args.limit} 张")
    total = len(files)
    if not total:
        print("[错误] 没找到图片文件")
        return 1
    print(f"待扫描 {total} 张，线程 {args.workers}\n")

    # ---------- 1) 元信息 ----------
    metas, errors = [], []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(probe, files):
            done += 1
            if "error" in r:
                errors.append(r)
            else:
                metas.append(r)
            if done % 5000 == 0 or done == total:
                print(f"  元信息 {done}/{total}", end="\r")
    print(" " * 40, end="\r")

    per_ext = Counter(p.suffix.lower() for p in files)
    size_total = sum(m["bytes"] for m in metas)
    sizes = sorted(m["bytes"] for m in metas)
    shorts = sorted(min(m["w"], m["h"]) for m in metas)
    longs = sorted(max(m["w"], m["h"]) for m in metas)
    areas = sorted(m["w"] * m["h"] for m in metas)
    aspects = sorted(max(m["w"], m["h"]) / max(1, min(m["w"], m["h"])) for m in metas)

    print("=" * 76)
    print("1. 基本信息")
    print("-" * 76)
    print(f"  图片总数        : {total}")
    print(f"  可读            : {len(metas)}")
    print(f"  无法打开        : {len(errors)}")
    print(f"  总大小          : {size_total / 1024**3:.2f} GB"
          f"（平均 {size_total / max(1, len(metas)) / 1024:.0f} KB，"
          f"中位 {quantile(sizes, .5) / 1024:.0f} KB）")
    print(f"  扩展名          : " + ", ".join(f"{k}×{v}" for k, v in per_ext.most_common()))
    print(f"  非图片文件      : {len(other)}")
    modes = Counter(m["mode"] for m in metas)
    if len(modes) > 1:
        print(f"  色彩模式        : " + ", ".join(f"{k}×{v}" for k, v in modes.most_common()))

    print()
    print("=" * 76)
    print("2. 尺寸分布（分位数）")
    print("-" * 76)
    print(f"  {'':>8}{'p1':>8}{'p5':>8}{'p25':>8}{'p50':>8}{'p75':>8}{'p95':>8}{'p99':>8}{'max':>9}")
    for label, vals in (("短边", shorts), ("长边", longs), ("面积", areas)):
        cells = "".join(f"{quantile(vals, q):>8}" for q in (.01, .05, .25, .5, .75, .95, .99))
        print(f"  {label:>8}{cells}{vals[-1]:>9}")
    print(f"  {'长宽比':>8}" +
          "".join(f"{quantile(aspects, q):>8.2f}" for q in (.01, .05, .25, .5, .75, .95, .99)) +
          f"{aspects[-1]:>9.2f}")

    print()
    print("=" * 76)
    print("3. 训练可用性（Qwen3.5-0.8B 配置：短边过小会被放大，面积超 262144 会被压缩）")
    print("-" * 76)
    for t in SHORT_TIERS:
        n = sum(1 for s in shorts if s < t)
        print(f"  短边 < {t:<4}      : {n:>7}  ({n / len(shorts) * 100:5.1f}%)"
              + ("   ← 会被放大，细节丢失，建议淘汰" if t == 336 else ""))
    n_big = sum(1 for a in areas if a > AREA_MAX)
    print(f"  面积 > {AREA_MAX} : {n_big:>7}  ({n_big / len(areas) * 100:5.1f}%)"
          f"   ← 会被等比压缩，不算缺陷，只是白占磁盘")
    keep = sum(1 for m in metas if min(m["w"], m["h"]) >= 336)
    print(f"  ⇒ 短边 ≥ 336 可直接用 : {keep} ({keep / max(1, len(metas)) * 100:.1f}%)")

    # ---------- 4) 重复检测 ----------
    dup_groups, dup_imgs, reclaim = [], 0, 0
    if not args.no_hash:
        print()
        print("=" * 76)
        print("4. 重复检测（dhash 感知哈希）")
        print("-" * 76)
        hashes, hdone = {}, 0
        for m in metas:
            try:
                h = dhash(Path(m["path"]))
            except Exception:                                # noqa: BLE001
                h = None
            m["dhash"] = h
            if h is not None:
                hashes.setdefault(h, []).append(m)
            hdone += 1
            if hdone % 3000 == 0 or hdone == len(metas):
                print(f"  哈希 {hdone}/{len(metas)}", end="\r")
        print(" " * 40, end="\r")
        for h, group in hashes.items():
            if len(group) > 1:
                dup_groups.append(group)
                dup_imgs += len(group)
                group.sort(key=lambda x: -x["bytes"])
                reclaim += sum(g["bytes"] for g in group[1:])
        print(f"  完全重复组      : {len(dup_groups)}")
        print(f"  涉及图片        : {dup_imgs} 张（{(dup_imgs / max(1, len(metas))) * 100:.1f}%）")
        print(f"  去重后可回收    : {reclaim / 1024**3:.2f} GB")
        if dup_groups:
            dup_groups.sort(key=len, reverse=True)
            print(f"  最大重复组      : {len(dup_groups[0])} 张，"
                  f"例：{Path(dup_groups[0][0]['path']).name}")

    # ---------- 5) 结论 ----------
    print()
    print("=" * 76)
    print("5. 结论")
    print("-" * 76)
    usable = keep
    if not args.no_hash:
        usable = keep - sum(1 for g in dup_groups for m in g[1:]
                            if min(m["w"], m["h"]) >= 336)
    print(f"  · 去重 + 剔除过小图后，预计可用 {usable} 张"
          f"（原始 {total}，可用率 {usable / total * 100:.1f}%）")
    if errors:
        print(f"  · 有 {len(errors)} 张无法打开，需要排查（前 5 个）：")
        for e in errors[:5]:
            print(f"      {Path(e['path']).name}: {e['error']}")
    print("  · 这批图的品类分布需要靠模型聚类/分类才能得出（文件名无类别信息）")

    # ---------- 输出 ----------
    report = {
        "root": str(root), "total": total, "readable": len(metas), "errors": len(errors),
        "size_bytes": size_total, "per_ext": dict(per_ext),
        "short_side": {str(t): sum(1 for s in shorts if s < t) for t in SHORT_TIERS},
        "area_gt_max": n_big, "keep_short_ge_336": keep, "usable_after_dedup": usable,
        "dup_groups": len(dup_groups), "dup_images": dup_imgs,
        "dup_reclaim_bytes": reclaim,
        "quantiles": {
            "short": {q: quantile(shorts, q) for q in (.01, .05, .25, .5, .75, .95, .99)},
            "area": {q: quantile(areas, q) for q in (.01, .05, .25, .5, .75, .95, .99)},
        },
    }
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
        print(f"\n  JSON 报告 → {args.report}")
    if args.csv:
        import csv as _csv
        with open(args.csv, "w", encoding="utf-8-sig", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["path", "width", "height", "short", "area", "bytes", "dhash"])
            for m in metas:
                w.writerow([m["path"], m["w"], m["h"], min(m["w"], m["h"]),
                            m["w"] * m["h"], m["bytes"], m.get("dhash", "")])
        print(f"  明细 CSV → {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
