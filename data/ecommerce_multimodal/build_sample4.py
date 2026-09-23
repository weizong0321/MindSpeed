#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建 sample4（Step 1 验证集）：只保留人工核验过的「同一款商品」子样式。

与 sample3 的关键差别
    sample3 按聚类纯度选子样式 → 一个子样式里塞了多款商品 → 标签只对部分图成立。
    sample4 的 14 个子样式全部用 coherence_*.png 逐行核验过。

image 级切分（不是 sku 级）
    每个子样式留最后 3 张做验证集，其余进训练集。
    这样验证集里是**同一款的没见过的图**，可以测「换个角度/背景还认不认得这款」，
    而不是只在测记忆。

用法
    python build_sample4.py --src sample3 --dst sample4
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# 人工核验为「同一款商品」的子样式（来自 coherence_*.png 逐行判定）
COHERENT = {
    "watch": ["watch_001", "watch_010", "watch_086", "watch_108",
              "watch_109", "watch_116", "watch_127", "watch_146"],
    "bag_backpack": ["bag_backpack_001", "bag_backpack_044",
                     "bag_backpack_119", "bag_backpack_155"],
    "shoes_sneaker": ["shoes_sneaker_123", "shoes_sneaker_176"],
}

EXCLUDED = {
    "watch": ["watch_037", "watch_140"],
    "bag_backpack": ["bag_backpack_007", "bag_backpack_018", "bag_backpack_052",
                     "bag_backpack_096", "bag_backpack_110", "bag_backpack_149"],
    "shoes_sneaker": ["shoes_sneaker_012", "shoes_sneaker_031", "shoes_sneaker_041",
                      "shoes_sneaker_054", "shoes_sneaker_080", "shoes_sneaker_166",
                      "shoes_sneaker_168", "shoes_sneaker_174"],
}

NUM_RE = re.compile(r"\d+(?:\.\d+)?")
EVAL_PER_SKU = 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="sample3")
    ap.add_argument("--dst", default="sample4")
    ap.add_argument("--eval-per-sku", type=int, default=EVAL_PER_SKU)
    ap.add_argument("--copy", action="store_true", default=True)
    args = ap.parse_args()

    src, dst = Path(args.src).resolve(), Path(args.dst).resolve()
    answers = json.loads((dst / "answers.json").read_text(encoding="utf-8"))
    q3 = json.loads((src / "questions.json").read_text(encoding="utf-8"))

    # 只保留 sample4 用到的品类
    questions = {"_readme": q3.get("_readme", [])}
    for cat in COHERENT:
        questions[cat] = q3[cat]
    (dst / "questions.json").write_text(
        json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")

    train, evalset, stats = [], [], []
    problems = []
    for cat, skus in COHERENT.items():
        qs = {q["id"]: q["user"] for q in questions[cat]["questions"]}
        n_cat_img = n_cat_tr = n_cat_ev = 0
        for sku in skus:
            key = f"{cat}/{sku}"
            if key not in answers:
                problems.append(f"{key} 在 answers.json 里没有")
                continue
            meta = answers[key]
            sdir = src / "images" / cat / sku
            imgs = sorted(p for p in sdir.iterdir()
                          if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            if len(imgs) <= args.eval_per_sku:
                problems.append(f"{key} 只有 {len(imgs)} 张，不够切验证集")
                continue
            ev = imgs[-args.eval_per_sku:]
            tr = imgs[:-args.eval_per_sku]

            if args.copy:
                out = dst / "images" / cat / sku
                out.mkdir(parents=True, exist_ok=True)
                for i, p in enumerate(imgs):
                    shutil.copy2(p, out / f"{i:03d}{p.suffix.lower()}")

            visible = " ".join(meta.get("visible") or [])
            caveat = str(meta.get("caveat") or "")
            for qid, qtext in qs.items():
                ans = (meta.get("qa") or {}).get(qid)
                if not ans:
                    problems.append(f"{key} 缺问法 {qid} 的答案")
                    continue
                # 质检：答案里的数字必须能在 visible 或问句里找到
                unseen = sorted(set(NUM_RE.findall(ans))
                                - set(NUM_RE.findall(visible))
                                - set(NUM_RE.findall(qtext)), key=len, reverse=True)
                if unseen and not caveat:
                    problems.append(f"{key}/{qid} 有无法核对的数字 {unseen} 且无 caveat")

                convs = [{"from": "user", "value": qtext},
                         {"from": "assistant", "value": ans}]
                rel = f"images/{cat}/{sku}"
                for p in tr:
                    idx = imgs.index(p)
                    train.append({"conversations": convs,
                                  "image": f"{rel}/{idx:03d}{p.suffix.lower()}"})
                    n_cat_tr += 1
                for p in ev:
                    idx = imgs.index(p)
                    evalset.append({"conversations": convs,
                                    "image": f"{rel}/{idx:03d}{p.suffix.lower()}"})
                    n_cat_ev += 1
            n_cat_img += len(imgs)
        stats.append((cat, len(skus), n_cat_img, n_cat_tr, n_cat_ev))

    (dst / "train.json").write_text(
        json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8")
    (dst / "eval.json").write_text(
        json.dumps(evalset, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 74)
    print(f"{'品类':<20}{'子样式':>7}{'图片':>7}{'训练样本':>9}{'验证样本':>9}")
    print("-" * 74)
    for cat, n, ni, ntr, nev in stats:
        print(f"{cat:<20}{n:>7}{ni:>7}{ntr:>9}{nev:>9}")
    print("-" * 74)
    print(f"{'合计':<20}{sum(s[1] for s in stats):>7}{sum(s[2] for s in stats):>7}"
          f"{sum(s[3] for s in stats):>9}{sum(s[4] for s in stats):>9}")
    print("=" * 74)

    print("\n被排除的子样式（核验为混合多款商品）：")
    for cat, skus in EXCLUDED.items():
        print(f"  {cat:<18}{len(skus)} 个: {', '.join(skus)}")

    if problems:
        print(f"\n[警告] {len(problems)} 个问题：")
        for p in problems[:15]:
            print("   " + p)
    else:
        print("\n[通过] 无质检问题")
    print(f"\n训练集 {len(train)} 条 / 验证集 {len(evalset)} 条")
    print("验证集是「同一款商品、没见过的图」—— 用于测泛化而不是记忆")
    return 0


if __name__ == "__main__":
    sys.exit(main())
