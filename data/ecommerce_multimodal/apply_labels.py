#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 classify_tool.html 导出的 labels.csv 落地成目录结构：

    images/<category>/<sku>/xxx.jpg

用法
    # 1) 先空跑，只打印计划，不动任何文件
    python apply_labels.py --csv labels.csv --src ./images

    # 2) 确认无误后执行（默认复制，原文件保留）
    python apply_labels.py --csv labels.csv --src ./images --apply

    # 3) 想直接移动（不保留原件）
    python apply_labels.py --csv labels.csv --src ./images --apply --move

参数
    --csv        classify_tool.html 导出的 labels.csv
    --src        图片当前所在目录（CSV 里 relativePath 的根）
    --dst        目标 images 根目录，默认与 --src 相同
    --apply      真正执行；不加则只做空跑
    --move       移动而非复制
    --overwrite  目标已存在同名文件时覆盖；默认是跳过并报警告

执行完请接着跑：
    python generate_dataset.py --check
"""

import argparse
import csv
import json
import re
import shutil
import sys
from collections import Counter, OrderedDict
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")
SKU_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def load_categories(*dirs: Path):
    """从 questions.json 读合法类别（依次在给定目录及其上一级找）；找不到返回 None，不做类别校验"""
    for d in dirs:
        for cand in (d / "questions.json", d.parent / "questions.json"):
            if cand.is_file():
                try:
                    with open(cand, encoding="utf-8") as f:
                        cats = [k for k in json.load(f) if not k.startswith("_")]
                    print(f"类别白名单    : 读取自 {cand}（{len(cats)} 个类别）")
                    return cats
                except Exception:                            # noqa: BLE001
                    return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="把 labels.csv 落地成 images/<category>/<sku>/ 结构")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default=None)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--move", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    csv_path = Path(args.csv).resolve()
    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve() if args.dst else src

    if not csv_path.is_file():
        print(f"[错误] 找不到 CSV：{csv_path}")
        return 1
    if not src.is_dir():
        print(f"[错误] 找不到图片目录：{src}")
        return 1

    valid_cats = load_categories(dst, src)

    rows, errors, warnings = [], [], []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            rel = (row.get("relativePath") or "").strip().replace("\\", "/")
            name = (row.get("file") or "").strip()
            cat = (row.get("category") or "").strip()
            sku = (row.get("sku") or "").strip()

            if not rel:
                rel = name
            if not rel or not cat or not sku:
                errors.append(f"第 {i} 行字段不全：{row}")
                continue
            if valid_cats is not None and cat not in valid_cats:
                errors.append(f"第 {i} 行类别 {cat!r} 不在 questions.json 中")
                continue
            if not SKU_RE.match(sku):
                errors.append(f"第 {i} 行款名 {sku!r} 非法（只允许小写字母/数字/下划线/短横线）")
                continue

            s = src / rel
            if not s.is_file():
                errors.append(f"第 {i} 行源文件不存在：{s}")
                continue
            rows.append((rel, cat, sku, s))

    if not rows and errors:
        for e in errors:
            print(f"[错误] {e}")
        print(f"\n没有可执行的记录。")
        return 1

    # 目标路径 + 冲突检测（同名文件进同一目录）
    plan = OrderedDict()          # (cat, sku) -> [(src, dst), ...]
    seen_dst = {}
    n_skip_exist = 0
    for rel, cat, sku, s in rows:
        target = dst / cat / sku / s.name
        key = target.as_posix().lower()
        if key in seen_dst:
            warnings.append(f"同名冲突：{seen_dst[key]} 与 {rel} 都想落到 {cat}/{sku}/{s.name}，"
                            f"后者自动改名")
            target = dst / cat / sku / f"{s.stem}__dup{s.suffix}"
        seen_dst[target.as_posix().lower()] = rel

        if target.exists() and not args.overwrite:
            n_skip_exist += 1
            warnings.append(f"目标已存在，跳过：{cat}/{sku}/{s.name}")
            continue
        plan.setdefault((cat, sku), []).append((s, target))

    # ---- 打印计划 ----
    print(f"CSV           : {csv_path.name}（{len(rows)} 条有效记录）")
    print(f"源目录        : {src}")
    print(f"目标目录      : {dst}")
    print(f"模式          : {'移动' if args.move else '复制'}"
          f"{'（--apply 实际执行）' if args.apply else '（空跑，未执行）'}")
    print()
    print(f"{'类别/款':<44}{'张数':>6}")
    print("-" * 52)
    for (cat, sku), items in plan.items():
        print(f"{cat + '/' + sku:<44}{len(items):>6}")
    print("-" * 52)
    print(f"{'合计':<44}{sum(len(v) for v in plan.values()):>6}")
    print()

    # 类别汇总
    per_cat = Counter(cat for (cat, _), items in plan.items() for _ in range(len(items)))
    print("按类别：")
    for cat, n in per_cat.most_common():
        skus = [sku for (c, sku) in plan if c == cat]
        print(f"  {cat:<20}{n:>5} 张 / {len(skus)} 款")
    print()

    # 未标注的文件
    labeled = {(src / rel) for rel, _, _, _ in rows}
    leftover = [p for p in src.rglob("*")
                if p.is_file() and p.suffix.lower() in IMG_EXTS and p not in labeled
                and p.parent == src]      # 只看仍在根目录下平铺的
    if leftover:
        print(f"[提示] 源目录下还有 {len(leftover)} 张图片未出现在 CSV 中（未标注），保持原位。")
        print()

    for w in warnings[:20]:
        print(f"[警告] {w}")
    if len(warnings) > 20:
        print(f"[警告] ...另有 {len(warnings) - 20} 条同类警告")
    for e in errors[:20]:
        print(f"[错误] {e}")
    if len(errors) > 20:
        print(f"[错误] ...另有 {len(errors) - 20} 条同类错误")
    if warnings or errors:
        print()

    if errors:
        print(f"有 {len(errors)} 条记录被跳过（见上）。")
    if not args.apply:
        print("[空跑结束] 未改动任何文件。加 --apply 执行。")
        return 0

    # ---- 执行 ----
    done = 0
    for (cat, sku), items in plan.items():
        (dst / cat / sku).mkdir(parents=True, exist_ok=True)
        for s, target in items:
            if args.move:
                shutil.move(str(s), str(target))
            else:
                shutil.copy2(s, target)
            done += 1

    print(f"[完成] 已{'移动' if args.move else '复制'} {done} 张图片到 {dst}")
    if errors:
        print(f"[提示] 另有 {len(errors)} 条记录因错误被跳过（见上方），这些图片仍在原处。")
    if warnings:
        print(f"[提示] 另有 {len(warnings)} 条警告已按规则处理（跳过或改名），见上方。")
    print()
    print("接下来：")
    print(f"  cd {dst.parent.as_posix()}")
    print("  python generate_dataset.py --check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
