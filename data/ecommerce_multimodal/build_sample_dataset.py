#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建 3 品类样板数据集：手表 / 背包 / 运动鞋。

做四件事
    1. 每个品类按纯度挑 top-N 个子样式（簇），每类留 2 个作为验证集
    2. 每个子样式取前 M 张图，复制到 <dst>/images/<category>/<sku>/
    3. 为每个品类生成一张评审图（10 个子样式 × 2 张代表图），供写答案时对照
    4. 生成 questions.json（已写好问法）与 answers.json（骨架，待填 visible/answer）

用法
    python build_sample_dataset.py --src <图片根> --dst sample3
"""

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from PIL import Image, ImageDraw

CATS = {
    "watch": {
        "zh": "手表",
        "skus": 10,          # 选几个子样式
        "eval": 2,           # 其中几个进验证集
        "exclude": [],       # 视觉复核后确认不属于本品类的子样式
        "questions": [
            {"id": "outdoor", "rounds": 1,
             "desc": "户外徒步跑步 / 防水+夜光+透气表带",
             "user": "<image>\n我平时户外徒步和跑步都戴，预算1000以内，要防水、能看时间和日期、"
                     "夜里有背光、表带出汗不闷，这款合适吗有什么短板？"},
            {"id": "daily", "rounds": 1,
             "desc": "日常通勤 / 耐刮+轻+显示日期星期",
             "user": "<image>\n日常通勤戴，要耐刮、戴着不重、能显示日期和星期，这款怎么样？"},
        ],
    },
    "bag_backpack": {
        "zh": "背包",
        "skus": 10,
        "eval": 2,
        # 视觉复核：这几个是斜挎包/单肩包/手提包，不是双肩背包，
        # 用“能装15.6寸电脑”的问法会变成错标签，剔除
        "exclude": ["bag_backpack_120", "bag_backpack_125", "bag_backpack_140",
                    "bag_backpack_121", "bag_backpack_109"],
        "questions": [
            {"id": "commute", "rounds": 1,
             "desc": "通勤+短途出差 / 15.6寸电脑+侧袋+背部透气",
             "user": "<image>\n通勤加短途出差用，要能装15.6寸电脑、有侧袋放水杯、"
                     "背部透气夏天不闷汗，这款符合吗？"},
            {"id": "outing", "rounds": 1,
             "desc": "周末一日徒步 / 轻量+能装外套和水+肩带不勒",
             "user": "<image>\n周末一日徒步用，要轻、能装外套和水、肩带不勒肩，这款合适吗？"},
        ],
    },
    "shoes_sneaker": {
        "zh": "运动鞋",
        "skus": 10,
        "eval": 2,
        # 视觉复核：_170 是黑色尖头女单鞋、_164 是漆皮高跟鞋、_162 看不清鞋型，
        # 拿“夜跑5公里”去问它们是错标签，剔除
        "exclude": ["shoes_sneaker_170", "shoes_sneaker_164", "shoes_sneaker_162"],
        "questions": [
            {"id": "daily_run", "rounds": 1,
             "desc": "夜跑5公里 / 缓震+透气+不磨脚",
             "user": "<image>\n每天夜跑5公里，体重70kg正常足弓，要缓震好、透气、不磨脚，这款适合吗？"},
            {"id": "student", "rounds": 1,
             "desc": "学生日常+体育课 / 百搭+耐穿+抓地",
             "user": "<image>\n学生日常穿加体育课，要百搭、耐穿、鞋底抓地，预算300以内，这款怎么样？"},
        ],
    },
}

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def load_assignments(path: Path):
    """{sku: [(path, purity), ...]}，按纯度降序"""
    by_sku = defaultdict(list)
    meta = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            by_sku[r["sku"]].append((r["path"], float(r["purity"] or 0)))
            meta[r["sku"]] = r["category"]
    for k in by_sku:
        by_sku[k].sort(key=lambda x: -x[1])
    return by_sku, meta


def sheet(picks, out_path, cols=5, thumb=170, per_cell=2):
    """picks: [(sku, size, purity, [img_path, ...]), ...]"""
    rows = (len(picks) + cols - 1) // cols
    cell_w = thumb * per_cell + 8
    cell_h = thumb + 20
    pad, head = 8, 28
    W = cols * cell_w + (cols + 1) * pad
    H = rows * cell_h + (rows + 1) * pad + head
    canvas = Image.new("RGB", (W, H), (18, 20, 24))
    d = ImageDraw.Draw(canvas)
    d.text((pad, 9), "3 品类样板 · 每格一个「视觉子样式」，按纯度挑选", fill=(240, 243, 250))
    for i, (sku, size, purity, paths) in enumerate(picks):
        r, c = divmod(i, cols)
        x0 = pad + c * (cell_w + pad)
        y0 = head + pad + r * (cell_h + pad)
        d.rectangle([x0 - 2, y0 - 2, x0 + cell_w, y0 + cell_h - 18], outline=(58, 64, 76))
        for j, p in enumerate(paths[:per_cell]):
            x = x0 + 4 + j * thumb
            try:
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    im.thumbnail((thumb - 4, thumb - 4), Image.LANCZOS)
                    canvas.paste(im, (x + (thumb - im.width) // 2, y0 + 2))
            except Exception:                                # noqa: BLE001
                d.rectangle([x, y0 + 2, x + thumb, y0 + 2 + thumb], fill=(48, 30, 30))
        d.text((x0 + 4, y0 + cell_h - 16), f"{sku}  {size}张  纯度{purity:.2f}",
               fill=(255, 214, 120))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="原始图片根目录")
    ap.add_argument("--assignments", default="assignments.csv")
    ap.add_argument("--dst", default="sample3")
    ap.add_argument("--per-sku", type=int, default=15, help="每个子样式取几张图")
    ap.add_argument("--copy", action="store_true", help="真正复制图片（默认只出评审图和骨架）")
    args = ap.parse_args()

    src, dst = Path(args.src).resolve(), Path(args.dst).resolve()
    by_sku, cat_of = load_assignments(Path(args.assignments))

    questions, answers = {}, {}
    questions["_readme"] = [
        "3 品类样板问法库。问法按【品类】组织，answers.json 里用 id 引用。",
        "问句必须以 <image>\\n 开头。rounds=1 时 user 是字符串，rounds=2 时是字符串数组。",
    ]
    answers["_readme"] = [
        "3 品类样板答案库。键 = <category>/<sku>，与 sample3/images/<category>/<sku>/ 对应。",
        "visible  图里能核对的可见特征（写 answer 时只准引用这里面的东西）",
        "caveat   图里看不到的参数，统一写“以商品页标注为准”",
        "split    train | eval（按款整款划分）",
        "qa       问法 id → 回答；单轮字符串，多轮数组",
        "先用 review_<category>.png 对照图片填写，再跑 generate_dataset.py --check 校验。",
    ]

    report = []
    for cat, cfg in CATS.items():
        excl = set(cfg.get("exclude", []))
        cand = [(sku, len(v), sum(p for _, p in v) / len(v), [src / p for p, _ in v])
                for sku, v in by_sku.items()
                if cat_of.get(sku) == cat and sku not in excl]
        cand.sort(key=lambda x: -x[2])
        picks = cand[:cfg["skus"]]
        if len(picks) < cfg["skus"]:
            print(f"[警告] {cat} 只有 {len(picks)} 个子样式可用（已排除 {len(excl)} 个）")
        if excl:
            print(f"{cat}: 已按视觉复核排除 {len(excl)} 个子样式 {sorted(excl)}")
        report.append((cat, picks))

        questions[cat] = {"name_zh": cfg["zh"], "questions": cfg["questions"]}

        for i, (sku, size, purity, paths) in enumerate(picks):
            use = paths[:args.per_sku]
            split = "eval" if i >= cfg["skus"] - cfg["eval"] else "train"
            answers[f"{cat}/{sku}"] = {
                "visible": [],
                "caveat": "",
                "split": split,
                "qa": {q["id"]: "" for q in cfg["questions"]},
                "purity": round(purity, 3),
                "cluster_size": size,
            }
            if args.copy:
                out = dst / "images" / cat / sku
                out.mkdir(parents=True, exist_ok=True)
                for j, p in enumerate(use):
                    if p.is_file():
                        shutil.copy2(p, out / f"{j:03d}{p.suffix.lower()}")
            print(f"  {cat}/{sku}: 取 {len(use)}/{size} 张，纯度 {purity:.3f} → {split}")

        sheet(picks, dst / f"review_{cat}.png")

    if args.copy:
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "questions.json").write_text(
            json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
        (dst / "answers.json").write_text(
            json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写出 {dst}/questions.json 与 {dst}/answers.json")

    print("\n" + "=" * 78)
    print(f"{'品类':<20}{'子样式':>8}{'取图上限':>10}{'实际张数':>10}")
    print("-" * 78)
    tot_sku = tot_img = 0
    for cat, picks in report:
        n_img = sum(min(len(p[3]), args.per_sku) for p in picks)
        tot_sku += len(picks)
        tot_img += n_img
        print(f"{cat:<20}{len(picks):>8}{args.per_sku:>10}{n_img:>10}")
    print("-" * 78)
    print(f"{'合计':<20}{tot_sku:>8}{'':>10}{tot_img:>10}")
    print("=" * 78)
    print(f"需要写的答案段数 = {tot_sku}（每段 5 分钟 ≈ {tot_sku * 5 / 60:.1f} 小时）")
    print(f"\n评审图：{dst}/review_<category>.png —— 对照它填 answers.json 的 visible 与 answer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
