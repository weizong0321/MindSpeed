#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电商导购多模态 SFT 数据集生成器 v2（按类别/款配对，杜绝图文错配）

目录约定
    data/ecommerce_multimodal/
    ├── images/<category>/<sku>/xxx.jpg   图片：类别 → 款
    ├── questions.json                    问法库：按【类别】
    ├── answers.json                      答案库：按【款】
    └── generate_dataset.py

用法
    python generate_dataset.py --check        # 只质检，不写文件
    python generate_dataset.py                # 生成 train.json / eval.json
    python generate_dataset.py --root <dir>   # 指定数据根目录（默认脚本所在目录）

与 v1 的区别（v1 见 generate_dataset_v1_legacy.py）
    v1: random.choice(TEMPLATE_LIBRARY) —— 每张图随机配一个类别模板，
        图文类别不匹配，答案里的参数（65W / 20000mAh / 579元）图里根本看不到，
        模型只能学“忽略图片、背模板”。
    v2: 答案由图片所在目录（类别/款）唯一确定，一图一答严格对应。
"""

import argparse
import json
import re
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

# Windows 控制台默认 GBK，中文/符号会抛 UnicodeEncodeError，这里强制 UTF-8
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

IMG_EXTS = (".jpg", ".jpeg", ".png")

# ---- 质检阈值 ---------------------------------------------------------------
MIN_IMAGES_PER_CATEGORY = 20   # 每类图片数建议下限
MIN_IMAGES_PER_SKU = 2         # 每款图片数建议下限
SHORT_SIDE_MIN = 336           # 短边低于此值会被预处理放大（配置上限 262144 像素）
SHORT_SIDE_GOOD = 512          # 建议值
AREA_MAX = 262144              # = image_max_pixels，超过会被等比压回

NUM_RE = re.compile(r"\d+(?:\.\d+)?")

ERRORS: list = []
WARNINGS: list = []
INFOS: list = []


def err(msg: str) -> None:
    ERRORS.append(msg)


def warn(msg: str) -> None:
    WARNINGS.append(msg)


def info(msg: str) -> None:
    INFOS.append(msg)


# ---- 载入 -------------------------------------------------------------------
def load_json(path: Path):
    if not path.is_file():
        err(f"缺少文件：{path}")
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        err(f"{path.name} 不是合法 JSON：{e}")
        return {}


def public_keys(d: dict):
    """去掉下划线开头的说明性键"""
    return [k for k in d if not k.startswith("_")]


# ---- 扫描图片目录 -----------------------------------------------------------
def scan_images(root: Path) -> "OrderedDict[str, OrderedDict[str, list]]":
    """返回 {category: {sku: [Path, ...]}}"""
    tree: "OrderedDict[str, OrderedDict[str, list]]" = OrderedDict()
    img_root = root / "images"
    if not img_root.is_dir():
        err(f"找不到图片根目录：{img_root}")
        return tree

    for cat_dir in sorted(p for p in img_root.iterdir() if p.is_dir()):
        skus: "OrderedDict[str, list]" = OrderedDict()
        loose = []
        for entry in sorted(cat_dir.iterdir()):
            if entry.is_dir():
                imgs = sorted(
                    f for f in entry.iterdir()
                    if f.is_file() and f.suffix.lower() in IMG_EXTS
                )
                skus[entry.name] = imgs
            elif entry.is_file() and entry.suffix.lower() in IMG_EXTS:
                loose.append(entry)
        if loose:
            warn(f"images/{cat_dir.name}/ 下有 {len(loose)} 张图未放进款目录，已忽略"
                 f"（示例：{loose[0].name}）")
        tree[cat_dir.name] = skus

    # images/ 根目录下直接平铺的图片（尚未归类）——最常见的起步状态，必须明确报出来
    flat_root = [p for p in img_root.iterdir()
                 if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if flat_root:
        warn(f"images/ 下有 {len(flat_root)} 张图片是平铺的，没有按 "
             f"images/<category>/<sku>/ 分层，本次全部忽略"
             f"（示例：{flat_root[0].name}）。请先按 DATA_SPEC.md 第 3 节归类。")
    return tree


# ---- 校验 -------------------------------------------------------------------
def validate_questions(questions: dict) -> dict:
    """返回 {category: {qid: question_dict}}"""
    index = {}
    for cat in public_keys(questions):
        spec = questions[cat]
        if "questions" not in spec or not spec["questions"]:
            err(f"questions.json: 类别 {cat} 没有 questions 列表")
            continue
        per_cat = {}
        for q in spec["questions"]:
            qid = q.get("id")
            if not qid:
                err(f"questions.json: 类别 {cat} 下有问法缺 id")
                continue
            if qid in per_cat:
                err(f"questions.json: 类别 {cat} 下问法 id 重复：{qid}")
                continue
            user = q.get("user")
            turns = [user] if isinstance(user, str) else list(user or [])
            if not turns:
                err(f"questions.json: {cat}/{qid} 的 user 为空")
                continue
            rounds = q.get("rounds", len(turns))
            if rounds != len(turns):
                err(f"questions.json: {cat}/{qid} rounds={rounds} 与 user 轮数 {len(turns)} 不一致")
            if not turns[0].lstrip().startswith("<image>"):
                warn(f"questions.json: {cat}/{qid} 第 1 轮缺 <image> 前缀，该样本不会真正喂图"
                     f"（多轮问法只有第 1 轮需要贴图）")
            per_cat[qid] = {"rounds": rounds, "user": turns, "desc": q.get("desc", "")}
        index[cat] = per_cat
    return index


def validate_and_collect(tree, q_index, answers, root: Path):
    """边校验边收集待生成样本，返回 (train, eval, stats)"""
    train, evalset = [], []
    stats = []

    # 1) questions.json 里定义了但 images/ 下还没有图片的类别（允许分批次备数据）
    missing_cats = [cat for cat in q_index if cat not in tree]
    if missing_cats:
        info(f"以下类别暂无 images/<category>/ 目录，本次跳过：{', '.join(missing_cats)}")

    for cat, skus in tree.items():
        if cat not in q_index:
            n = sum(len(v) for v in skus.values())
            err(f"images/{cat}/ 有 {n} 张图，但 questions.json 里没有该类别（请补问法）")
            continue

        n_cat_imgs = 0
        n_cat_samples = 0
        n_cat_skus_ok = 0

        for sku, imgs in skus.items():
            key = f"{cat}/{sku}"
            if not imgs:
                warn(f"images/{key}/ 是空目录，已跳过")
                continue
            n_cat_imgs += len(imgs)

            if key not in answers:
                err(f"images/{key}/ 有 {len(imgs)} 张图，但 answers.json 里没有该款")
                continue

            meta = answers[key]
            if len(imgs) < MIN_IMAGES_PER_SKU:
                warn(f"{key} 只有 {len(imgs)} 张图（建议 ≥ {MIN_IMAGES_PER_SKU}）")

            split = str(meta.get("split", "train")).lower()
            if split not in ("train", "eval"):
                err(f"{key} 的 split={split!r} 非法，只能是 train / eval")
                continue

            qa = meta.get("qa") or {}
            if not qa:
                err(f"{key} 的 qa 为空，没有可生成的问答对")
                continue

            visible = " ".join(meta.get("visible") or [])
            caveat = str(meta.get("caveat") or "")
            if not visible:
                warn(f"{key} 没有写 visible（图上可见特征），无法核对答案是否有编造")

            pairs = []
            n_err_before = len(ERRORS)
            for qid, answer in qa.items():
                if qid not in q_index[cat]:
                    err(f"{key} 的 qa 引用了 {cat} 下不存在的问法 id：{qid}")
                    continue
                q = q_index[cat][qid]
                ans_turns = [answer] if isinstance(answer, str) else list(answer)
                if len(ans_turns) != len(q["user"]):
                    err(f"{key} 的问法 {qid} 有 {len(q['user'])} 轮，"
                        f"但给了 {len(ans_turns)} 段回答，轮数必须一致")
                    continue
                pairs.append((qid, q, ans_turns))
                check_answer_numbers(key, qid, ans_turns, visible, caveat,
                                     " ".join(q["user"]))

            if not pairs:
                continue
            if len(ERRORS) > n_err_before:
                # 该款的问答对有问题，不产出样本（具体错误已在上面的检查中记录）
                continue

            n_cat_skus_ok += 1
            for img in imgs:
                rel = img.relative_to(root).as_posix()   # images/<cat>/<sku>/xxx.jpg
                for qid, q, ans_turns in pairs:
                    convs = []
                    for u, a in zip(q["user"], ans_turns):
                        convs.append({"from": "user", "value": u})
                        convs.append({"from": "assistant", "value": a})
                    sample = {"conversations": convs, "image": rel}
                    (evalset if split == "eval" else train).append(sample)
                    n_cat_samples += 1

        if n_cat_imgs < MIN_IMAGES_PER_CATEGORY:
            warn(f"类别 {cat} 只有 {n_cat_imgs} 张图（建议 ≥ {MIN_IMAGES_PER_CATEGORY}）")

        stats.append((cat, len(skus), n_cat_skus_ok, n_cat_imgs, n_cat_samples))

    # 2) answers.json 里有、但 images/ 下没图的孤儿款
    for key in public_keys(answers):
        cat, _, sku = key.partition("/")
        if not sku:
            err(f"answers.json 的键 {key!r} 不是 <category>/<sku> 格式")
            continue
        if sku not in tree.get(cat, {}):
            warn(f"answers.json 里的 {key} 在 images/{cat}/{sku}/ 下没有对应目录或图片")

    return train, evalset, stats


def check_answer_numbers(key, qid, ans_turns, visible, caveat, q_text=""):
    """答案里的数字若在 visible 里找不到证据，就要靠 caveat 声明“以商品页标注为准”。

    用户问句里出现过的数字是用户的约束条件（"15.6寸电脑"、"预算300以内"），
    答案复述它们不算编造，所以要从待核对集合里剔除，否则会产生大量假阳性。
    """
    ans_text = " ".join(ans_turns)
    unseen = sorted(set(NUM_RE.findall(ans_text))
                    - set(NUM_RE.findall(visible))
                    - set(NUM_RE.findall(q_text)),
                    key=len, reverse=True)
    if not unseen:
        return
    if not caveat:
        err(f"{key} 的问法 {qid}：答案里出现图中无法核对的数字 {unseen}，"
            f"但该款没写 caveat（图上不可见参数声明）")
    else:
        warn(f"{key} 的问法 {qid}：数字 {unseen} 不在 visible 中，"
             f"已由 caveat 声明为“以商品页标注为准”，请人工确认")


# ---- 图片尺寸检查（需要 PIL，缺失则跳过）------------------------------------
def check_image_files(tree, root: Path):
    try:
        from PIL import Image
    except ImportError:
        warn("未安装 Pillow，跳过图片尺寸检查（pip install Pillow 可启用）")
        return

    checked = 0
    for cat, skus in tree.items():
        for sku, imgs in skus.items():
            small, huge = [], []
            for p in imgs:
                try:
                    with Image.open(p) as im:
                        w, h = im.size
                except Exception as e:                       # noqa: BLE001
                    err(f"图片无法打开：{p.relative_to(root)}（{e}）")
                    continue
                if min(w, h) < SHORT_SIDE_MIN:
                    small.append(f"{p.name}({w}x{h})")
                if w * h > AREA_MAX:
                    huge.append(f"{p.name}({w}x{h})")
                checked += 1
            if small:
                warn(f"{cat}/{sku}: {len(small)} 张短边 < {SHORT_SIDE_MIN}，"
                     f"会被预处理放大导致细节丢失：{small[:3]}")
            if huge:
                warn(f"{cat}/{sku}: {len(huge)} 张面积 > {AREA_MAX}(=512x512)，"
                     f"会被等比压缩，多传的部分无效：{huge[:3]}")
    print(f"  · 已检查 {checked} 张图片的尺寸")


# ---- 输出 -------------------------------------------------------------------
def write_json(path: Path, data) -> None:
    if path.is_file():
        bak = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, bak)
        print(f"  ↳ 原 {path.name} 已备份为 {bak.name}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def report(stats, train, evalset) -> None:
    print("\n" + "=" * 74)
    print(f"{'类别':<20}{'款数':>6}{'有效款':>8}{'图片':>8}{'样本':>8}")
    print("-" * 74)
    for cat, n_sku, n_ok, n_img, n_smp in stats:
        print(f"{cat:<20}{n_sku:>6}{n_ok:>8}{n_img:>8}{n_smp:>8}")
    print("-" * 74)
    print(f"{'合计':<20}{'':>6}{'':>8}"
          f"{sum(s[3] for s in stats):>8}{sum(s[4] for s in stats):>8}")
    print("=" * 74)
    print(f"训练集 {len(train)} 条 / 验证集 {len(evalset)} 条"
          f"（验证集按款留出，与训练集无同款图片）")


def main() -> int:
    ap = argparse.ArgumentParser(description="电商导购多模态数据集生成器 v2")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent),
                    help="数据根目录（默认脚本所在目录）")
    ap.add_argument("--check", action="store_true", help="只质检，不写文件")
    ap.add_argument("--no-size-check", action="store_true", help="跳过图片尺寸检查")
    ap.add_argument("--out", default="train.json", help="训练集输出文件名")
    ap.add_argument("--out-eval", default="eval.json", help="验证集输出文件名")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    print(f"数据根目录：{root}\n")

    questions = load_json(root / "questions.json")
    answers = load_json(root / "answers.json")
    if ERRORS:
        for e in ERRORS:
            print(f"[错误] {e}")
        return 1

    q_index = validate_questions(questions)
    tree = scan_images(root)
    train, evalset, stats = validate_and_collect(tree, q_index, answers, root)

    if not args.no_size_check and not args.check:
        check_image_files(tree, root)

    if stats:
        report(stats, train, evalset)

    print()
    for i in INFOS:
        print(f"[提示] {i}")
    for w in WARNINGS:
        print(f"[警告] {w}")
    for e in ERRORS:
        print(f"[错误] {e}")

    if ERRORS:
        print(f"\n发现 {len(ERRORS)} 个错误、{len(WARNINGS)} 个警告，未写文件。")
        return 1

    if args.check:
        if not train and not evalset:
            print(f"\n[警告] --check 未发现错误，但当前没有任何可生成的样本"
                  f"（{len(WARNINGS)} 个警告），请先确认 images/ 已按 <category>/<sku>/ 分层。")
        else:
            print(f"\n[通过] --check 通过（{len(WARNINGS)} 个警告）："
                  f"预计生成 train {len(train)} 条 / eval {len(evalset)} 条。")
        return 0

    if not train and not evalset:
        print("\n[错误] 没有生成任何样本（大概率是 images/ 下还没有按 <category>/<sku>/ 放图），"
              "为避免覆盖已有 train.json，本次不写文件。")
        return 1

    write_json(root / args.out, train)
    write_json(root / args.out_eval, evalset)
    print(f"\n[完成] 已写出 {args.out}（{len(train)} 条）、{args.out_eval}（{len(evalset)} 条）")
    print(f"   警告 {len(WARNINGS)} 条，建议逐条确认后再训练。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
