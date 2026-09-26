#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蒸馏后的评测：属性命中率（直接测"有没有看图"）+ 原有的组内/组间重合度。

为什么换指标
    之前用"输出与参考答案的 2-gram 相似度"当判据，对长文本会失真、也无法跨品类数比较。
    逐图属性给了一个**客观真值**：这张图的真实颜色/部件是 VLM 标的。
    于是可以算：
      color_hit : 生成答案里有没有出现该图的真实颜色词
      parts_hit : 该图真实部件里有几个出现在答案里
    这才是"有没有看图"的直接度量。

用法
    python eval_attrs.py --model <HF目录> --data sample6 --tag d1 \
        --per-sku 1 --out /workspace/user_data/eval_d1.json
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

BASE = "/workspace/MindSpeed/data/ecommerce_multimodal"


def ngrams(s):
    s = re.sub(r"[\s，。；：、！？（）,.!?()\"']", "", s)
    return {s[i:i + 2] for i in range(max(0, len(s) - 1))}


def sim(a, b):
    A, B = ngrams(a), ngrams(b)
    return len(A & B) / len(A | B) if A and B else 0.0


def build(model_dir, device):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    arch = (cfg.architectures or [""])[0]
    import transformers
    cls = getattr(transformers, arch, None)
    if cls is None:
        from transformers import AutoModelForImageTextToText as cls   # type: ignore
    try:
        m = cls.from_pretrained(model_dir, dtype=torch.bfloat16,
                                trust_remote_code=True)
    except TypeError:
        m = cls.from_pretrained(model_dir, torch_dtype=torch.bfloat16,
                                trust_remote_code=True)
    return m.eval().to(device)


@torch.no_grad()
def gen(model, processor, eos_ids, path, question, device, max_new=150):
    img = Image.open(path).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"},
                                        {"type": "text", "text": question}]}]
    text = processor.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True)
    inp = processor(text=[text], images=[img], return_tensors="pt")
    inp = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inp.items()}
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                         eos_token_id=eos_ids, pad_token_id=eos_ids[0])
    return processor.decode(out[0][inp["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="sample6")
    ap.add_argument("--images", default="",
                    help="图片所在数据集目录名（默认与 --data 相同；"
                         "sample6 的标签在 sample6、图片在 sample5）")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-sku", type=int, default=1, help="每个子样式评测几张图")
    args = ap.parse_args()

    d = f"{BASE}/{args.data}"
    img_root = f"{BASE}/{args.images or args.data}"
    print(f"[{args.tag}] 标签 {d}，图片 {img_root}", flush=True)
    evalset = json.loads(Path(f"{d}/eval.json").read_text(encoding="utf-8"))
    truth = json.loads(Path(f"{d}/attrs_truth.json").read_text(encoding="utf-8"))

    try:
        import torch_npu                                        # noqa: F401
        device = torch.device("npu:0")
    except Exception:                                           # noqa: BLE001
        device = torch.device("cpu")
    print(f"[{args.tag}] 数据 {args.data}，设备 {device}", flush=True)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tok = processor.tokenizer
    eos = sorted({tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"),
                  tok.convert_tokens_to_ids("<|endoftext|>")} - {None})
    eos = [i for i in eos if isinstance(i, int) and i >= 0]
    model = build(args.model, device)

    # 每个子样式取前 per-sku 张图的全部问法
    by_sku = defaultdict(list)
    for s in evalset:
        by_sku["/".join(s["image"].split("/")[1:3])].append(s)
    picked = []
    for sku, items in sorted(by_sku.items()):
        imgs = sorted({s["image"] for s in items})
        keep = set(imgs[:args.per_sku])
        picked += [s for s in items if s["image"] in keep]

    rows, t0 = [], time.time()
    for i, s in enumerate(picked, 1):
        q = s["conversations"][0]["value"]
        a = gen(model, processor, eos, os.path.join(img_root, s["image"]), q, device)
        t = truth.get(s["image"]) or {}
        color = (t.get("color") or "").strip()
        parts = [p for p in (t.get("parts") or []) if p]
        c_hit = bool(color) and color in a
        p_hit = sum(1 for p in parts if p in a) / len(parts) if parts else 0.0
        rows.append({"sku": "/".join(s["image"].split("/")[1:3]), "image": s["image"],
                     "answer": a, "truth_color": color, "truth_parts": parts,
                     "color_hit": c_hit, "parts_hit": round(p_hit, 3)})
        if i % 10 == 0 or i == len(picked):
            el = time.time() - t0
            print(f"  [{i}/{len(picked)}] 色命中 "
                  f"{sum(r['color_hit'] for r in rows) / len(rows):.2f} "
                  f"部件命中 {sum(r['parts_hit'] for r in rows) / len(rows):.2f} "
                  f"{el / i:.1f}s/条", flush=True)

    ac, wi = [], []
    qg = defaultdict(list)
    for r in rows:
        qg[r["image"].rsplit("/", 1)[0]].append(r)       # 按子样式分组即可
    for _, rs in qg.items():
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                ac.append(sim(rs[i]["answer"], rs[j]["answer"]))
    # 组内：同一子样式的不同图
    for sku, items in by_sku.items():
        rs = [r for r in rows if r["sku"] == sku]
        imgs = defaultdict(list)
        for r in rs:
            imgs[r["image"]].append(r)
        keys = sorted(imgs)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                for a in imgs[keys[i]]:
                    for b in imgs[keys[j]]:
                        wi.append(sim(a["answer"], b["answer"]))

    avg = lambda x: sum(x) / len(x) if x else 0.0
    summary = {
        "tag": args.tag, "data": args.data, "samples": len(rows),
        "color_hit": round(avg([1.0 if r["color_hit"] else 0.0 for r in rows]), 3),
        "parts_hit": round(avg([r["parts_hit"] for r in rows]), 3),
        "across_image_sim": round(avg(ac), 3),
        "within_style_sim": round(avg(wi), 3),
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    print("\n" + "=" * 72)
    print(f"[{args.tag}] {args.data} 判据（{len(rows)} 条）")
    print("-" * 72)
    print(f"  颜色命中率（图上真实主色出现在答案里）: {summary['color_hit']:.3f}")
    print(f"  部件命中率（真实部件出现在答案里的比例）: {summary['parts_hit']:.3f}")
    print(f"  同子样式不同图的重合度（越低说明逐图差异越大）: "
          f"{summary['within_style_sim']:.3f}")
    print("=" * 72)
    Path(args.out).write_text(json.dumps(
        {"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"→ {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
