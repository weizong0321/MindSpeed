#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1 客观评测：四个可自动统计的判据。

判据
  A 组间重合度  同一问题下、**不同子样式**的输出重合度 → 越低越好（目标 < 0.30）
                （现在 sample3 是 0.45–0.91，说明模型没区分开不同商品）
  B 组内重合度  同一子样式、**不同图片**的输出重合度 → 越高越好（同款答案应当一致）
  C 跨品类串味  背包答案里出现"表盘"、鞋答案里出现"背/双肩包"等 → 目标 0
  D 归属准确率  给定输出，与哪个子样式的参考答案最像 → 命中自身子样式的比例

用法
    python eval_step1.py --model <HF目录> --tag base --out /workspace/user_data/step1_base.json
    python eval_step1.py --model <HF目录> --tag ft   --out /workspace/user_data/step1_ft.json
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import torch
from PIL import Image
from transformers import AutoProcessor

DATA = "/workspace/MindSpeed/data/ecommerce_multimodal/sample4"

# 跨品类串味词表：某品类的答案里不该出现的词
FORBIDDEN = {
    "watch": ["鞋面", "鞋底", "上脚", "系带", "低帮", "帆布", "双肩包", "拉链袋",
              "包身", "肩带", "隔层", "背包"],
    "bag_backpack": ["表盘", "表带", "表壳", "指针", "三针", "时标", "日期窗",
                     "夜光", "鞋面", "系带", "鞋底", "低帮", "帆布", "上脚"],
    "shoes_sneaker": ["表盘", "表带", "表壳", "指针", "三针", "时标",
                      "双肩包", "拉链袋", "肩带", "隔层", "包身"],
}


def bigrams(s):
    s = re.sub(r"[\s，。；：、！？（）,.!?()\"']", "", s)
    return {s[i:i + 2] for i in range(max(0, len(s) - 1))}


def sim(a, b):
    A, B = bigrams(a), bigrams(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def build_model(model_dir, device):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    arch = (cfg.architectures or [""])[0]
    import transformers
    cls = getattr(transformers, arch, None)
    if cls is None:
        from transformers import AutoModelForImageTextToText as cls   # type: ignore
    try:
        m = cls.from_pretrained(model_dir, dtype=torch.bfloat16, trust_remote_code=True)
    except TypeError:
        m = cls.from_pretrained(model_dir, torch_dtype=torch.bfloat16,
                                trust_remote_code=True)
    return m.eval().to(device)


@torch.no_grad()
def generate(model, processor, eos_ids, image_path, question, device, max_new=160):
    img = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"},
                                            {"type": "text", "text": question}]}]
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt")
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                         eos_token_id=eos_ids, pad_token_id=eos_ids[0])
    gen = out[0][inputs["input_ids"].shape[1]:]
    return processor.decode(gen, skip_special_tokens=True).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="sample4",
                    help="数据集目录名（在 /workspace/MindSpeed/data/ecommerce_multimodal/ 下）")
    ap.add_argument("--per-sku", type=int, default=3, help="每个子样式评测几张图")
    args = ap.parse_args()

    DATA = f"/workspace/MindSpeed/data/ecommerce_multimodal/{args.data}"
    print(f"[{args.tag}] 数据集 {DATA}")

    evalset = json.loads(Path(f"{DATA}/eval.json").read_text(encoding="utf-8"))
    answers = json.loads(Path(f"{DATA}/answers.json").read_text(encoding="utf-8"))
    questions = json.loads(Path(f"{DATA}/questions.json").read_text(encoding="utf-8"))

    # 问句文本 -> 问法 id（归属判断必须同问法比较，否则拿 outdoor 的答案去比 daily 的参考）
    qid_of = {}
    for cat, spec in questions.items():
        if cat.startswith("_"):
            continue
        for q in spec["questions"]:
            qid_of[q["user"]] = (cat, q["id"])

    # 每个子样式的参考答案
    refs = {}
    for key, meta in answers.items():
        if key.startswith("_"):
            continue
        for qid, txt in (meta.get("qa") or {}).items():
            refs[(key, qid)] = txt

    try:
        import torch_npu                                        # noqa: F401
        device = torch.device("npu:0")
    except Exception:                                           # noqa: BLE001
        device = torch.device("cpu")
    print(f"[{args.tag}] 设备 {device}")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tok = processor.tokenizer
    eos_ids = sorted({tok.eos_token_id,
                      tok.convert_tokens_to_ids("<|im_end|>"),
                      tok.convert_tokens_to_ids("<|endoftext|>")} - {None})
    eos_ids = [i for i in eos_ids if isinstance(i, int) and i >= 0]
    print(f"[{args.tag}] eos_ids={eos_ids}")
    model = build_model(args.model, device)

    # 采样：每个 (子样式) 取前 per-sku 条（eval.json 里每个子样式 3 图 × 2 问法 = 6 条）
    by_sku = defaultdict(list)
    for s in evalset:
        sku = "/".join(s["image"].split("/")[1:3])
        by_sku[sku].append(s)

    results = []
    t0 = time.time()
    for sku, items in sorted(by_sku.items()):
        picked = items[:args.per_sku * 2]
        for s in picked:
            q = s["conversations"][0]["value"]
            path = os.path.join(
                f"/workspace/MindSpeed/data/ecommerce_multimodal/{args.data}",
                s["image"])
            ans = generate(model, processor, eos_ids, path, q, device)
            results.append({"sku": sku, "question": q, "image": s["image"],
                            "answer": ans})
        print(f"  {sku} 完成（{len(picked)} 条，累计 {len(results)}，"
              f"{(time.time() - t0) / 60:.1f} 分钟）", flush=True)

    # ---------- 指标 ----------
    cat_of = {k: k.split("/")[0] for k in by_sku}
    # A/B：按 (问题) 分组比较
    q_groups = defaultdict(list)
    for r in results:
        q_groups[r["question"]].append(r)

    across, within = [], []
    for q, rs in q_groups.items():
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                a, b = rs[i], rs[j]
                s = sim(a["answer"], b["answer"])
                (within if a["sku"] == b["sku"] else across).append(s)

    # C：串味（自动版，覆盖任意品类数量）
    # 每个品类抽取"专属 2-gram"（只在本类及至多 1 个别的类里出现的短语），
    # 若某条输出的内容大量命中**别的品类**的专属短语，即判为串味。
    cat_grams, gram_cats = defaultdict(set), defaultdict(set)
    for (key, _q), txt in refs.items():
        c = key.split("/")[0]
        gs = bigrams(txt)
        cat_grams[c] |= gs
        for g in gs:
            gram_cats[g].add(c)
    distinctive = {c: {g for g in gs if len(gram_cats[g]) <= 2}
                   for c, gs in cat_grams.items()}

    contam = []
    for r in results:
        cat = r["sku"].split("/")[0]
        own = distinctive.get(cat, set())
        out_g = bigrams(r["answer"])
        bad = []
        for oc, gs in distinctive.items():
            if oc == cat:
                continue
            if len((gs - own) & out_g) >= 3:
                bad.append(oc)
        # 旧的硬编码词表作为补充信号
        hard = [w for w in FORBIDDEN.get(cat, []) if w in r["answer"]]
        if bad or hard:
            contam.append({"sku": r["sku"], "cats": sorted(set(bad + hard)),
                           "answer": r["answer"][:90]})

    # D：归属准确率（只在同一问法内比较）
    correct = 0
    conf = defaultdict(int)
    for r in results:
        cat, qid = qid_of.get(r["question"], (r["sku"].split("/")[0], None))
        best, best_s = None, -1.0
        for (key, q), txt in refs.items():
            if q != qid:
                continue
            sc = sim(r["answer"], txt)
            if sc > best_s:
                best_s, best = sc, key
        conf[(r["sku"], best)] += 1
        if best == r["sku"]:
            correct += 1

    n = max(1, len(results))
    avg = lambda xs: sum(xs) / len(xs) if xs else 0.0
    summary = {
        "tag": args.tag, "samples": len(results),
        "across_style_sim": round(avg(across), 3),
        "within_style_sim": round(avg(within), 3),
        "contamination_count": len(contam),
        "ownership_accuracy": round(correct / n, 3),
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    print("\n" + "=" * 70)
    print(f"[{args.tag}] Step 1 判据（{len(results)} 条生成）")
    print("-" * 70)
    print(f"  A 组间重合度（不同商品）: {summary['across_style_sim']:.3f}"
          f"   目标 < 0.30")
    print(f"  B 组内重合度（同一商品）: {summary['within_style_sim']:.3f}"
          f"   越高越好（同款应一致）")
    print(f"  C 跨品类串味条数        : {summary['contamination_count']}"
          f"   目标 0")
    print(f"  D 归属准确率            : {summary['ownership_accuracy']:.3f}"
          f"   越高越好")
    print("=" * 70)
    if contam:
        print("\n串味样例：")
        for c in contam[:6]:
            print(f"  [{c['sku']}] 命中 {c['words']} → {c['answer']}")

    Path(args.out).write_text(json.dumps(
        {"summary": summary, "results": results, "contamination": contam,
         "confusion": {f"{k[0]}←{k[1]}": v for k, v in conf.items() if k[0] != k[1]}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果 → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
