#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sample3 推理对照：验证"模型到底有没有看图"。

判据设计
  单看"同分布问法能不能复现训练答案"是不够的 —— 模型完全可能忽略图片、
  只靠问法背模板（这正是旧数据集当时的失败模式）。

  所以关键是这一组对照：
      同一个问题 + 同类但不同子样式的两张图
  如果模型对两张图给出**明显不同且各自符合图上可见特征**的回答，说明它在看图；
  如果两张图回答一模一样，说明它还是只认问法、不看图。

  同时跑 BASE（原始 base）与 FT（微调后）两组，便于对比差异。

用法
    python infer_sample3.py --model <HF目录> --tag base  --out /workspace/user_data/infer_base.json
    python infer_sample3.py --model <HF目录> --tag ft    --out /workspace/user_data/infer_ft.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import torch
from PIL import Image
from transformers import AutoProcessor

DATA = "/workspace/MindSpeed/data/ecommerce_multimodal/sample3"

# (标签, 图片相对路径, 问题, 分组)
# 同一 group 内的两条 = "同类不同子样式 + 完全相同的问题"
CASES = [
    ("watch_127_同分布",   "images/watch/watch_127/000.jpg",         "日常通勤戴，要耐刮、戴着不重、能显示日期和星期，这款怎么样？", "watch_同分布"),
    ("watch_086_同分布",   "images/watch/watch_086/000.jpg",         "日常通勤戴，要耐刮、戴着不重、能显示日期和星期，这款怎么样？", "watch_同分布"),

    ("watch_127_新问题",   "images/watch/watch_127/001.jpg",         "这款适合送给喜欢户外运动的朋友吗？说说理由。",             "watch_新问题"),
    ("watch_086_新问题",   "images/watch/watch_086/001.jpg",         "这款适合送给喜欢户外运动的朋友吗？说说理由。",             "watch_新问题"),

    ("bag_001_同分布",     "images/bag_backpack/bag_backpack_001/000.jpg", "通勤加短途出差用，要能装15.6寸电脑、有侧袋放水杯、背部透气夏天不闷汗，这款符合吗？", "bag_同分布"),
    ("bag_007_同分布",     "images/bag_backpack/bag_backpack_007/000.jpg", "通勤加短途出差用，要能装15.6寸电脑、有侧袋放水杯、背部透气夏天不闷汗，这款符合吗？", "bag_同分布"),

    ("bag_001_新问题",     "images/bag_backpack/bag_backpack_001/001.jpg", "我想装相机加两件换洗衣服去露营，这个包合适吗？",         "bag_新问题"),
    ("bag_007_新问题",     "images/bag_backpack/bag_backpack_007/001.jpg", "我想装相机加两件换洗衣服去露营，这个包合适吗？",         "bag_新问题"),

    ("shoe_031_同分布",    "images/shoes_sneaker/shoes_sneaker_031/000.jpg", "每天夜跑5公里，体重70kg正常足弓，要缓震好、透气、不磨脚，这款适合吗？", "shoe_同分布"),
    ("shoe_080_同分布",    "images/shoes_sneaker/shoes_sneaker_080/000.jpg", "每天夜跑5公里，体重70kg正常足弓，要缓震好、透气、不磨脚，这款适合吗？", "shoe_同分布"),

    ("shoe_031_新问题",    "images/shoes_sneaker/shoes_sneaker_031/001.jpg", "我打算买来上下班走路通勤，这双合适吗？",                 "shoe_新问题"),
    ("shoe_080_新问题",    "images/shoes_sneaker/shoes_sneaker_080/001.jpg", "我打算买来上下班走路通勤，这双合适吗？",                 "shoe_新问题"),
]


def build_model(model_dir, device):
    """优先用 Qwen3_5 专用类，退回 AutoModelForImageTextToText"""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    arch = (cfg.architectures or [""])[0]
    print(f"  架构：{arch}")
    import transformers
    cls = getattr(transformers, arch, None)
    if cls is None:
        from transformers import AutoModelForImageTextToText as cls   # type: ignore
    try:
        model = cls.from_pretrained(model_dir, dtype=torch.bfloat16,
                                    trust_remote_code=True)
    except TypeError:
        model = cls.from_pretrained(model_dir, torch_dtype=torch.bfloat16,
                                    trust_remote_code=True)
    return model.eval().to(device)


@torch.no_grad()
def generate(model, processor, image_path, question, device, max_new_tokens=200):
    img = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": question},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt")
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return processor.decode(gen, skip_special_tokens=True).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    args = ap.parse_args()

    try:
        import torch_npu                                        # noqa: F401
        device = torch.device("npu:0")
    except Exception:                                           # noqa: BLE001
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[{args.tag}] 设备 {device}，模型 {args.model}")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    t0 = time.time()
    model = build_model(args.model, device)
    print(f"[{args.tag}] 模型加载完成 {time.time() - t0:.1f}s")

    results = []
    for tag, rel, question, group in CASES:
        path = os.path.join("/workspace/MindSpeed/data/ecommerce_multimodal/sample3",
                            rel.replace("images/", "images/", 1))
        if not os.path.isfile(path):
            print(f"  [跳过] 图片不存在 {path}")
            continue
        t1 = time.time()
        ans = generate(model, processor, path, question, device, args.max_new_tokens)
        results.append({"tag": tag, "group": group, "image": rel,
                        "question": question, "answer": ans,
                        "sec": round(time.time() - t1, 2)})
        print(f"\n===== [{args.tag}] {tag} =====\nQ: {question}\nA: {ans}")

    Path(args.out).write_text(
        json.dumps({"tag": args.tag, "model": args.model, "results": results},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[{args.tag}] 结果已写入 {args.out}")

    # 组内差异自查：同组两条回答是否雷同
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        groups[r["group"]].append(r)
    print(f"\n[{args.tag}] 组内差异（判断有没有看图）")
    for g, rs in groups.items():
        if len(rs) < 2:
            continue
        a, b = rs[0]["answer"], rs[1]["answer"]
        same = a == b
        # 简易重合度
        common = len(set(a) & set(b)) / max(1, len(set(a) | set(b)))
        print(f"  {g:16} 完全相同={same}  字符重合度={common:.2f}  "
              f"({rs[0]['tag']} vs {rs[1]['tag']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
