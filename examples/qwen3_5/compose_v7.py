#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sample7：路线乙的答案合成 —— 不断言颜色，只保留可核实的结构特征。

相对 sample6 改了什么
  1) **去掉颜色**。实测 4B/30B 说"主色"只有 0.67–0.73 正确率，
     换模型、换问法都没用（详见 EXP5_gate_report.md 第 5/7 节）。
     答案里写一个 1/4 概率是错的颜色，比不写更伤。
  2) **部件走白名单**（part_vocab.py）。VLM 会把背景写进部件：
        灰长裤 -> ['抽绳','松紧腰','侧边字母带','木质椅子']
        棕皮鞋 -> ['鞋头','鞋带','灰色袜子','金属孔板']
     按品类白名单只放行该品类真实可能有的结构词，同时把啰嗦说法归一化
        '侧边口袋' -> '口袋'，'厚实中底' -> '中底'，'越野轮胎' -> '轮胎'
  3) 保留 category 与 scene（这两个 VLM 是可靠的），所以答案仍然逐图不同。

用法
    python compose_v7.py --captions /workspace/user_data/captions_all.jsonl \\
        --data sample5 --out sample7
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(_Path(__file__).resolve().parent))
from part_vocab import filter_parts  # noqa: E402
# --- path shim (auto-added for portability) ---
import os as _os
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
MM_DATA = _Path(_os.environ.get("MM_DATA", _REPO_ROOT / "data" / "ecommerce_multimodal"))
MM_WORK = _Path(_os.environ.get("MM_WORK", _REPO_ROOT / "work"))
MM_EX = _REPO_ROOT / "examples" / "qwen3_5"
# --- end shim ---


FORBIDDEN_IN_ATTRS = ["品牌", "型号", "价格", "容量", "功率", "尺码", "成分"]

# 品类中文名（VLM 的 category 不可靠时用目录名兜底）
CAT_CN = {
    "accessories": "配饰", "apparel_pants": "长裤", "apparel_shorts": "短裤",
    "apparel_top": "上衣", "apparel_trip": "外套", "appliance_large": "家电",
    "baby_products": "婴儿用品", "bag_backpack": "双肩包", "bag_handbag": "手提包",
    "cosmetics": "化妆品", "health_supplement": "保健品",
    "household_detergent": "洗护用品", "jewelry": "珠宝首饰",
    "luggage": "行李箱", "personal_care": "个护用品", "purse_wallet": "钱包",
    "shoes_kids": "童鞋", "shoes_leather": "皮鞋", "shoes_other": "鞋",
    "shoes_sneaker": "运动鞋", "toy_vehicle": "玩具车", "watch": "手表",
}


def strip_visual_clause(text):
    """把 base 答案里的「图中为…」从句摘掉，返回 (前置, 后置)"""
    i = text.find("图中为")
    if i < 0:
        return "", text
    end = len(text)
    for ch in ("。", "；"):
        k = text.find(ch, i)
        if k >= 0:
            end = min(end, k)
    return text[:i], text[end + 1:] if end < len(text) else ""


def visual_clause(attrs, cat):
    """用该图自己的属性生成「图中为…」从句 —— 不含颜色，只含结构。"""
    if not attrs:
        return "", []
    category = (attrs.get("category") or "").strip()
    if not category or len(category) > 8:
        category = CAT_CN.get(cat, "")
    raw_parts = [p.strip() for p in (attrs.get("parts") or []) if p and p.strip()]
    parts = filter_parts(raw_parts, cat)
    parts = [p for p in parts if not any(f in p for f in FORBIDDEN_IN_ATTRS)]
    scene = (attrs.get("scene") or "").strip()
    seg = category
    if parts:
        seg += "，" + "、".join(parts[:4])
    if scene:
        seg += "，" + scene
    return seg.strip("，") if seg else "", parts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True)
    ap.add_argument("--data", default="sample5")
    ap.add_argument("--out", default="sample7")
    ap.add_argument("--min-agree", type=float, default=0.7)
    ap.add_argument("--per-sku", type=int, default=15)
    ap.add_argument("--eval-per-sku", type=int, default=3)
    ap.add_argument("--min-parts", type=int, default=2)
    args = ap.parse_args()

    data = Path(str(MM_DATA)) / args.data
    answers = json.loads((data / "answers.json").read_text(encoding="utf-8"))
    questions = json.loads((data / "questions.json").read_text(encoding="utf-8"))
    qmap = {}
    for cat, spec in questions.items():
        if cat.startswith("_"):
            continue
        for q in spec["questions"]:
            qmap[(cat, q["id"])] = q["user"]

    # ---- 1) 按子样式聚合 ----
    by_sku = defaultdict(list)
    bad_json = 0
    for line in open(args.captions, encoding="utf-8"):
        d = json.loads(line)
        if not d.get("attrs"):
            bad_json += 1
            continue
        p = d["image"].split("/")
        by_sku[p[-3] + "/" + p[-2]].append(d)
    print(f"读到 {sum(len(v) for v in by_sku.values())} 条有效标注，"
          f"{bad_json} 条解析失败，{len(by_sku)} 个子样式")

    # ---- 2) 品类一致性 = 纯度信号 ----
    keep, drop = [], []
    print("\n" + "=" * 84)
    print(f"{'子样式':<34}{'张数':>5}{'主要品类':>12}{'一致率':>8}  判定")
    print("-" * 84)
    for key in sorted(by_sku):
        items = by_sku[key]
        cats = Counter((it["attrs"].get("category") or "?") for it in items)
        top, n = cats.most_common(1)[0]
        agree = n / len(items)
        verdict = "保留" if agree >= args.min_agree else "丢弃(混款)"
        (keep if agree >= args.min_agree else drop).append((key, top, agree))
        print(f"{key:<34}{len(items):>5}{top:>12}{agree:>8.2f}  {verdict}")
    print("-" * 84)
    print(f"保留 {len(keep)} 个子样式，丢弃 {len(drop)} 个（品类一致率 < {args.min_agree}）")
    if drop:
        print("丢弃的：" + ", ".join(k for k, _, _ in drop))

    # ---- 3) 逐图合成答案 ----
    out = Path(str(MM_DATA)) / args.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    train, evalset, stats = [], [], []
    n_skip_parts = n_skip_qid = 0
    for key, _, _ in keep:
        cat, sku = key.split("/")
        base = answers.get(key) or {}
        items = sorted(by_sku[key], key=lambda d: d["image"])
        if len(items) < 8:
            continue
        per_image = []
        for it in items:
            attrs = it["attrs"]
            vc, parts = visual_clause(attrs, cat)
            if len(parts) < args.min_parts or not vc:
                n_skip_parts += 1
                continue
            qa_new = {}
            for qid, a in (base.get("qa") or {}).items():
                if (cat, qid) not in qmap:
                    n_skip_qid += 1
                    continue
                pre, post = strip_visual_clause(a)
                qa_new[qid] = f"{pre}图中为{vc}。{post}" if pre or post else a
            if not qa_new:
                continue
            per_image.append({"image": it["image"], "attrs": attrs, "qa": qa_new,
                              "parts_filtered": parts})
        if len(per_image) < 8:
            continue
        per_image = per_image[:args.per_sku]
        n_eval = min(args.eval_per_sku, len(per_image) // 4)
        rel = f"images/{cat}/{sku}"
        for i, rec in enumerate(per_image):
            is_eval = i >= len(per_image) - n_eval
            m = re.search(r"images/.*$", rec["image"])
            img_rel = m.group(0) if m else rec["image"]
            for qid, a in rec["qa"].items():
                convs = [{"from": "user", "value": qmap[(cat, qid)]},
                         {"from": "assistant", "value": a}]
                (evalset if is_eval else train).append(
                    {"conversations": convs, "image": img_rel, "qid": qid})
        stats.append((key, len(per_image), len(per_image) - n_eval, n_eval))

    (out / "train.json").write_text(json.dumps(train, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    (out / "eval.json").write_text(json.dumps(evalset, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    (out / "questions.json").write_text(
        json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")

    # 评测用真值：只保留"结构部件"（不含颜色），供新的 grounding 评测使用
    truth = {}
    for key, _, _ in keep:
        cat = key.split("/")[0]
        for it in by_sku[key]:
            m = re.search(r"images/.*$", it["image"])
            if not m:
                continue
            _, parts = visual_clause(it["attrs"], cat)
            truth[m.group(0)] = {
                "category": it["attrs"].get("category"),
                "scene": it["attrs"].get("scene"),
                "parts": parts,                      # 过滤后的结构部件
                "vlm_category_raw": it["attrs"].get("category"),
            }
    (out / "attrs_truth.json").write_text(
        json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 84)
    print(f"{args.out}：{len(stats)} 个子样式 / {sum(s[1] for s in stats)} 张图")
    print(f"  训练 {len(train)} 条 / 验证 {len(evalset)} 条")
    print(f"  因结构部件不足被跳过的图：{n_skip_parts}；问法未定义被跳过：{n_skip_qid}")
    bad = [s for s in stats if s[3] == 0]
    if bad:
        print(f"  [警告] {len(bad)} 个子样式的验证集为空")
    (out / "compose_stats.json").write_text(json.dumps(
        {"keep": [k for k, _, _ in keep], "drop": [k for k, _, _ in drop],
         "stats": stats, "train": len(train), "eval": len(evalset)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
