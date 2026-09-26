#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 VLM 逐图属性合成「逐图答案」，并顺带用"品类一致性"自动筛掉混款子样式。

设计
    base 答案（子代理按问法写好）= 结论 + 图中为… + 逐条对照 + 以商品页标注为准 + 短板
    其中只有「图中为…」这一段是"看图"才能说的。本脚本把这一段替换成
    **该张图自己的 VLM 属性**（颜色/部件/画面），其余保持不动。
    结果：格式与逐条对照仍然合规，描述部分逐图不同 → 背模板不再能达标。

纯度筛选
    同一子样式内 VLM 判出的品类若不统一（多数占比 < --min-agree），说明混了多款 → 丢弃。
    实测这比任何嵌入指标都灵（皮带子样式里混进了手表，就是靠这个发现的）。

用法
    python compose_from_attrs.py --captions captions_all.jsonl --data sample5 --out sample6
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

FORBIDDEN_IN_ATTRS = ["品牌", "型号", "价格", "容量", "功率", "尺码", "成分"]


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


def visual_clause(attrs):
    """用该图自己的属性生成「图中为…」从句"""
    if not attrs:
        return ""
    color = (attrs.get("color") or "").strip()
    cat = (attrs.get("category") or "").strip()
    parts = [p.strip() for p in (attrs.get("parts") or []) if p and p.strip()]
    scene = (attrs.get("scene") or "").strip()
    parts = [p for p in parts if not any(f in p for f in FORBIDDEN_IN_ATTRS)]
    head = f"{color}{cat}" if color and cat else (cat or color)
    seg = head
    if parts:
        seg += "，" + "、".join(parts[:4])
    if scene:
        seg += "，" + scene
    return seg.strip("，") if seg else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True)
    ap.add_argument("--data", default="sample5", help="提供 base 答案与问法的数据集")
    ap.add_argument("--out", default="sample6")
    ap.add_argument("--min-agree", type=float, default=0.7)
    ap.add_argument("--per-sku", type=int, default=15)
    ap.add_argument("--eval-per-sku", type=int, default=3)
    ap.add_argument("--min-parts", type=int, default=2)
    args = ap.parse_args()

    data = Path("/workspace/MindSpeed/data/ecommerce_multimodal") / args.data
    answers = json.loads((data / "answers.json").read_text(encoding="utf-8"))
    questions = json.loads((data / "questions.json").read_text(encoding="utf-8"))
    qmap = {}
    for cat, spec in questions.items():
        if cat.startswith("_"):
            continue
        for q in spec["questions"]:
            qmap[(cat, q["id"])] = q["user"]

    # ---- 1) 按子样式聚合 VLM 标注 ----
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
    out = Path("/workspace/MindSpeed/data/ecommerce_multimodal") / args.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    train, evalset, stats = [], [], []
    n_skip_parts = 0
    for key, _, _ in keep:
        cat, sku = key.split("/")
        base = answers.get(key) or {}
        items = sorted(by_sku[key], key=lambda d: d["image"])
        if len(items) < 8:
            continue
        # 每张图的答案
        per_image = []
        n_skip_qid = 0
        for it in items:
            attrs = it["attrs"]
            if len(attrs.get("parts") or []) < args.min_parts:
                n_skip_parts += 1
                continue
            vc = visual_clause(attrs)
            if not vc:
                n_skip_parts += 1
                continue
            qa_new = {}
            for qid, a in (base.get("qa") or {}).items():
                # 该问法在本数据集的 questions.json 里没有定义（例如被剔除的品类）→ 跳过
                if (cat, qid) not in qmap:
                    n_skip_qid += 1
                    continue
                pre, post = strip_visual_clause(a)
                qa_new[qid] = f"{pre}图中为{vc}。{post}" if pre or post else a
            if not qa_new:
                continue
            per_image.append({"image": it["image"], "attrs": attrs, "qa": qa_new})
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
    # 评测用：每张图的"真值属性"（用于算属性命中率）
    truth = {}
    for key, _, _ in keep:
        for it in by_sku[key]:
            m = re.search(r"images/.*$", it["image"])
            if m:
                truth[m.group(0)] = it["attrs"]
    (out / "attrs_truth.json").write_text(
        json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 84)
    print(f"sample6：{len(stats)} 个子样式 / {sum(s[1] for s in stats)} 张图")
    print(f"  训练 {len(train)} 条 / 验证 {len(evalset)} 条")
    print(f"  因部件不足被跳过的图：{n_skip_parts}；因问法未定义被跳过的问答：{n_skip_qid}")
    if n_skip_qid:
        print("  （问法未定义通常意味着该品类在最终选择里被剔除了）")
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
