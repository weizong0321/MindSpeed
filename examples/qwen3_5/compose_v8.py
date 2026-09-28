#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sample8：用"闭词表可靠属性"合成答案。

相对 sample7 的改动
    sample7 的视觉从句是「品类，部件A、部件B，场景」，
    而"部件"是开放式自由填写 —— 实测两个老师同一问法只有 13.3% 完全一致，
    也就是说这个目标本身欠定，模型复现不了（逐字匹配 0.138）。

    sample8 换成闭词表属性：
        shoes_leather: 系带 / 低帮 / 厚底       （都是互斥、有确定答案的选项）
        bag_backpack : 拉链 / 带侧袋
    并用 attr_render 渲染成自然短语，同时保证可以反向精确解析（用于评测）。

    只使用同时满足"可靠（两老师一致率≥0.85）"和"有信息量（基线≤0.85且取值≥2）"
    的属性；某个品类若一个都没剩下，该品类整体不参与。
"""
# --- path shim (portable paths; auto-generated) ---
import os as _os
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
MM_DATA = _Path(_os.environ.get("MM_DATA", _REPO_ROOT / "data" / "ecommerce_multimodal"))
MM_WORK = _Path(_os.environ.get("MM_WORK", _REPO_ROOT / "work"))
MM_EX = _REPO_ROOT / "examples" / "qwen3_5"
_skill_scripts = _REPO_ROOT / "skill" / "ecommerce_guide_skill" / "scripts"
if str(_skill_scripts) not in _sys.path:
    _sys.path.insert(0, str(_skill_scripts))
# --- end shim ---


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
from attr_render import render_values  # noqa: E402

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
    i = text.find("图中为")
    if i < 0:
        return "", text
    end = len(text)
    for ch in ("。", "；"):
        k = text.find(ch, i)
        if k >= 0:
            end = min(end, k)
    return text[:i], text[end + 1:] if end < len(text) else ""

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrs", default="/workspace/user_data/attr_t4b_855.json")
    ap.add_argument("--selected",
                    default="/workspace/user_data/attr_selected_855.json")
    ap.add_argument("--captions", default="/workspace/user_data/captions_all.jsonl")
    ap.add_argument("--data", default="sample5")
    ap.add_argument("--out", default="sample8")
    ap.add_argument("--min-attrs", type=int, default=2)
    ap.add_argument("--per-sku", type=int, default=15)
    ap.add_argument("--eval-per-sku", type=int, default=3)
    a = ap.parse_args()

    data = Path(str(MM_DATA)) / a.data
    answers = json.loads((data / "answers.json").read_text(encoding="utf-8"))
    questions = json.loads((data / "questions.json").read_text(encoding="utf-8"))
    qmap = {}
    for cat, spec in questions.items():
        if cat.startswith("_"):
            continue
        for q in spec["questions"]:
            qmap[(cat, q["id"])] = q["user"]

    selected = json.loads(Path(a.selected).read_text(encoding="utf-8"))

    # 场景（scene）来自原来的 captions，属性来自新的闭词表标注
    scene_of = {}
    for line in open(a.captions, encoding="utf-8"):
        d = json.loads(line)
        if not d.get("attrs"):
            continue
        m = re.search(r"images/.*$", d["image"])
        if m:
            scene_of[m.group(0)] = (d["attrs"].get("scene") or "").strip()

    attr_of = {}
    for r in json.loads(Path(a.attrs).read_text("utf-8")):
        attr_of[r["id"]] = r

    # id 形如 "<cat>__003"，而图片是 images/<cat>/<sku>/<file>.jpg
    per_image = defaultdict(dict)
    for it in json.loads(Path("/workspace/user_data/attr_list_855.json")
                         .read_text("utf-8")):
        per_image[it["image"]] = attr_of.get(it["id"], {})

    by_sku = defaultdict(list)
    for img, rec in per_image.items():
        p = img.split("/")
        by_sku[p[1] + "/" + p[2]].append(img)

    out = Path(str(MM_DATA)) / a.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    train, evalset, stats = [], [], []
    truth = {}
    n_skip = 0
    n_sku_drop = 0

    for key in sorted(by_sku):
        cat, sku = key.split("/")
        sel = selected.get(cat) or []
        if not sel:
            n_sku_drop += 1
            continue
        base = answers.get(key) or {}
        imgs = sorted(by_sku[key])
        nice = []
        for img in imgs:
            rec = per_image.get(img) or {}
            j = rec.get("attrs") or {}
            av = {k: str(j[k]).strip() for k in sel
                  if j.get(k) and "不确定" not in str(j[k])}
            phrases = render_values(cat, av, order=sel)
            if len(phrases) < a.min_attrs:
                n_skip += 1
                continue
            scene = scene_of.get(img, "")
            seg = cat and CAT_CN.get(cat, "")
            clause = (seg or "") + "，" + "、".join(phrases)
            if scene:
                clause += "，" + scene
            clause = clause.strip("，")
            qa_new = {}
            for qid, ans in (base.get("qa") or {}).items():
                if (cat, qid) not in qmap:
                    continue
                pre, post = strip_visual_clause(ans)
                qa_new[qid] = f"{pre}图中为{clause}。{post}" if pre or post else ans
            if not qa_new:
                continue
            nice.append({"image": img, "attrs": av, "qa": qa_new,
                         "clause": clause})
        if len(nice) < 8:
            n_sku_drop += 1
            continue
        nice = nice[:a.per_sku]
        n_eval = min(a.eval_per_sku, len(nice) // 4)
        for i, r in enumerate(nice):
            is_eval = i >= len(nice) - n_eval
            truth[r["image"]] = {"category": cat, "attrs": r["attrs"],
                                 "clause": r["clause"]}
            for qid, ans in r["qa"].items():
                convs = [{"from": "user", "value": qmap[(cat, qid)]},
                         {"from": "assistant", "value": ans}]
                (evalset if is_eval else train).append(
                    {"conversations": convs, "image": r["image"], "qid": qid})
        stats.append((key, len(nice), len(nice) - n_eval, n_eval))

    (out / "train.json").write_text(json.dumps(train, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    (out / "eval.json").write_text(json.dumps(evalset, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    (out / "questions.json").write_text(
        json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "attrs_truth.json").write_text(
        json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "attr_spec.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{a.out}：{len(stats)} 个子样式 / {sum(s[1] for s in stats)} 张图")
    print(f"  训练 {len(train)} 条 / 验证 {len(evalset)} 条")
    print(f"  因可用属性不足(<{a.min_attrs})跳过 {n_skip} 张")
    print(f"  因该品类无可用属性/图太少跳过 {n_sku_drop} 个子样式")
    print("DONE")
    return 0

if __name__ == "__main__":
    sys.exit(main())
