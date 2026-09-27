#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""路线乙的评测：颜色命中率没用了（答案里本来就不写颜色），改成测"有没有真看图"。

三个指标
  1) 纯净度：答案的「图中为…」从句里还有没有颜色词 / 背景词（目标 ≈ 0）
  2) 接地率：该图的结构部件有多少出现在答案里（目标高）
  3) **换图测试（关键）**：
       同一句问题，分别喂
         (a) 它自己的图   -> ans_correct
         (b) 同子样式的另一张图 -> ans_swap
       然后看 ans_swap 描述的是不是"换过去那张图"的部件。
       如果是，说明模型真的在按图说话；如果 ans_swap 和 ans_correct 一样，
       说明它根本没看图，只是在背模板。
     报告 grounded_correct / grounded_swap / 两答案相同率。
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
# --- path shim (auto-added for portability) ---
import os as _os
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
MM_DATA = _Path(_os.environ.get("MM_DATA", _REPO_ROOT / "data" / "ecommerce_multimodal"))
MM_WORK = _Path(_os.environ.get("MM_WORK", _REPO_ROOT / "work"))
MM_EX = _REPO_ROOT / "examples" / "qwen3_5"
# --- end shim ---


BASE = str(MM_DATA)

COLOR_WORDS = [
    "黑色", "白色", "灰色", "银色", "金色", "棕色", "红色", "粉色", "橙色",
    "黄色", "绿色", "蓝色", "紫色", "透明", "藏青", "墨绿", "酒红", "米色",
    "卡其", "驼色", "深蓝", "浅蓝", "深灰", "浅灰", "银灰",
]
BACKGROUND_WORDS = [
    "袜子", "椅子", "桌面", "地面", "地板", "床单", "床品", "沙发", "墙壁",
    "模特", "手指", "手臂", "鞋盒", "纸箱", "窗帘", "地毯",
]
SCENE_VALUES = ["白底商品图", "实拍场景", "手持展示", "带包装盒"]


def clause_of(answer):
    m = re.search(r"图中为(.*?)。", answer)
    return m.group(1) if m else ""


def core_of(clause):
    c = clause
    for s in SCENE_VALUES:
        c = c.replace(s, "")
    return c


def load_model(model_dir, device):
    from transformers import AutoConfig
    import transformers
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    arch = (cfg.architectures or [""])[0]
    cls = getattr(transformers, arch, None) or \
        transformers.AutoModelForImageTextToText
    big = sum(p.stat().st_size for p in Path(model_dir).glob("*.safetensors"))
    if big > 10e9:
        return cls.from_pretrained(model_dir, trust_remote_code=True,
                                   dtype=torch.bfloat16,
                                   low_cpu_mem_usage=True,
                                   device_map="auto").eval()
    try:
        m = cls.from_pretrained(model_dir, dtype=torch.bfloat16,
                                trust_remote_code=True)
    except TypeError:
        m = cls.from_pretrained(model_dir, torch_dtype=torch.bfloat16,
                                trust_remote_code=True)
    return m.eval().to(device)


@torch.no_grad()
def ask(model, proc, eos, path, question, device, max_new=200):
    img = Image.open(path).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"},
                                        {"type": "text", "text": question}]}]
    try:
        text = proc.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True,
                                        enable_thinking=False)
    except TypeError:
        text = proc.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True)
    inp = proc(text=[text], images=[img], return_tensors="pt")
    inp = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inp.items()}
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                         eos_token_id=eos, pad_token_id=eos[0])
    return proc.decode(out[0][inp["input_ids"].shape[1]:],
                       skip_special_tokens=True).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="sample7")
    ap.add_argument("--images", default="sample5")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-sku", type=int, default=2)
    a = ap.parse_args()

    d = Path(BASE) / a.data
    img_root = Path(BASE) / a.images
    evalset = json.loads((d / "eval.json").read_text("utf-8"))
    truth = json.loads((d / "attrs_truth.json").read_text("utf-8"))

    try:
        import torch_npu                                        # noqa: F401
        device = torch.device("npu:0")
    except Exception:                                           # noqa: BLE001
        device = torch.device("cpu")

    print(f"[{a.tag}] {a.model} -> {device}", flush=True)
    proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
    tok = proc.tokenizer
    eos = sorted({tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"),
                  tok.convert_tokens_to_ids("<|endoftext|>")} - {None})
    eos = [i for i in eos if isinstance(i, int) and i >= 0] or [tok.eos_token_id]
    model = load_model(a.model, device)

    # 按子样式分组
    by_sku = defaultdict(list)
    for s in evalset:
        by_sku["/".join(s["image"].split("/")[1:3])].append(s)

    rows, t0 = [], time.time()
    for sku, items in sorted(by_sku.items()):
        imgs = sorted({s["image"] for s in items})
        sel = imgs[:a.per_sku]
        if len(imgs) < 2:
            continue                                            # 换图测试需要两张
        for t, img in enumerate(sel):
            other = imgs[(t + 1) % len(imgs)]
            q = next(s["conversations"][0]["value"] for s in items
                     if s["image"] == img)
            a_own = ask(model, proc, eos, img_root / img, q, device)
            a_swap = ask(model, proc, eos, img_root / other, q, device)

            P_own = (truth.get(img) or {}).get("parts") or []
            P_other = (truth.get(other) or {}).get("parts") or []
            cl_own, cl_swap = clause_of(a_own), clause_of(a_swap)
            rows.append({
                "sku": sku, "image": img, "other": other,
                "q": q, "a_own": a_own, "a_swap": a_swap,
                "parts_own": P_own, "parts_other": P_other,
                # 接地：自己的部件出现在自己答案里
                "grounded_own": (sum(1 for p in P_own if p in cl_own) / len(P_own)
                                 if P_own else 0.0),
                # 换图后，描述的是不是"换过去那张图"的部件
                "grounded_swap": (sum(1 for p in P_other if p in cl_swap)
                                  / len(P_other) if P_other else 0.0),
                # 换图后，还在说自己原图的部件吗（越低越好）
                "leak_own_after_swap": (sum(1 for p in P_own if p in cl_swap)
                                        / len(P_own) if P_own else 0.0),
                "same_answer": a_own == a_swap,
                "color_in_clause": bool([w for w in COLOR_WORDS
                                         if w in core_of(cl_own)]),
                "bg_in_clause": bool([w for w in BACKGROUND_WORDS
                                      if w in core_of(cl_own)]),
                "len_own": len(a_own),
            })
            if len(rows) % 5 == 0:
                el = time.time() - t0
                print(f"  [{len(rows)}] {el / len(rows):.1f}s/图", flush=True)

    avg = lambda x: sum(x) / len(x) if x else 0.0                   # noqa: E731
    summary = {
        "tag": a.tag, "model": a.model, "data": a.data, "n": len(rows),
        "grounded_own": round(avg([r["grounded_own"] for r in rows]), 3),
        "grounded_swap": round(avg([r["grounded_swap"] for r in rows]), 3),
        "leak_own_after_swap": round(
            avg([r["leak_own_after_swap"] for r in rows]), 3),
        "same_answer_rate": round(
            avg([1.0 if r["same_answer"] else 0.0 for r in rows]), 3),
        "color_in_clause_rate": round(
            avg([1.0 if r["color_in_clause"] else 0.0 for r in rows]), 3),
        "bg_in_clause_rate": round(
            avg([1.0 if r["bg_in_clause"] else 0.0 for r in rows]), 3),
        "avg_answer_len": round(avg([r["len_own"] for r in rows]), 1),
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    print("\n" + "=" * 76)
    print(f"[{a.tag}] data={a.data}  {len(rows)} 张（每张跑了 自己的图 + 换图）")
    print(f"  ① 自己的部件出现在自己答案里   grounded_own     = "
          f"{summary['grounded_own']:.3f}")
    print(f"  ② 换图后描述的是新图的部件吗   grounded_swap    = "
          f"{summary['grounded_swap']:.3f}   ← 越高越说明真在按图说话")
    print(f"  ③ 换图后还在说自己原图的部件   leak_after_swap  = "
          f"{summary['leak_own_after_swap']:.3f}   ← 越低越好")
    print(f"  ④ 换图前后答案完全相同         same_answer_rate = "
          f"{summary['same_answer_rate']:.3f}   ← 越低越好")
    print(f"  ⑤ 从句里出现颜色词             {summary['color_in_clause_rate']:.4f}")
    print(f"  ⑥ 从句里出现背景词             {summary['bg_in_clause_rate']:.4f}")
    print(f"  ⑦ 平均答案长度 {summary['avg_answer_len']:.0f} 字，"
          f"用时 {summary['elapsed_min']} 分钟")
    print("=" * 76)

    Path(a.out).write_text(json.dumps(
        {"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"→ {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
