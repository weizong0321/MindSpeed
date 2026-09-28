#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最小可用版本：图片 + 问题 -> 经过验证的属性 + 代码渲染的答案。

流程
    1) 模型只负责读属性：生成答案，取第一句「图中为…」（生成 48 token 即停）
    2) 用 attr_render 精确反解成 {属性: 取值}，并按目录品类过滤
    3) 正文交给 render_answer.py 用代码渲染 —— 品类名词取目录名，
       问题里的诉求统一诚实免责

为什么这样比"直接用模型输出的整段话"好（78 张实测）
    品类名词写错        0.167 -> 0.000
    逻辑别扭的短板句    0.321 -> 0.000
    平均长度            92 字 -> 51 字
    属性准确率          不变（整组全对 0.756）

用法
    python extract_and_render.py --model <HF目录> --data sample8b \
        --images sample5 --limit 20
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
import time
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(_Path(__file__).resolve().parent))
from attr_render import parse_values  # noqa: E402
from render_answer import render  # noqa: E402
import attr_spec  # noqa: E402

import torch  # noqa: E402  （load_model 里要用，必须模块级导入）

BASE = str(MM_DATA)

def clause_of(a):
    m = re.search(r"图中为(.*?)。", a)
    return m.group(1).strip() if m else None

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

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default="/workspace/user_data/output/qwen3_5_0.8B_sample8b_hf")
    ap.add_argument("--data", default="sample8b")
    ap.add_argument("--images", default="sample5")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=48,
                    help="只需要第一句，48 token 够；能显著降延迟")
    ap.add_argument("--hedge", action="store_true", default=True,
                    help="属性也用\"图中看是…\"弱化（24% 出错率下更诚实）")
    ap.add_argument("--out", default="/workspace/user_data/pipeline_demo.json")
    a = ap.parse_args()

    import torch as _t  # noqa: F401  已模块级导入，这里只是保持既有结构
    from PIL import Image
    from transformers import AutoProcessor

    d = Path(BASE) / a.data
    img_root = Path(BASE) / a.images
    spec = json.loads((d / "attr_spec.json").read_text("utf-8"))
    full_spec = {}
    for cat, attrs in spec.items():
        m = {k: o for k, _, o in attr_spec.ATTRS.get(cat, [])}
        full_spec[cat] = [{"key": k, "options": m.get(k, [])} for k in attrs]

    rows = json.loads((d / "eval.json").read_text("utf-8"))
    seen, items = set(), []
    for r in rows:
        if r["image"] in seen:
            continue
        seen.add(r["image"])
        items.append(r)
    items = items[:a.limit]

    try:
        import torch_npu                                        # noqa: F401
        device = torch.device("npu:0")
    except Exception:                                           # noqa: BLE001
        device = torch.device("cpu")

    proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
    tok = proc.tokenizer
    eos = sorted({tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"),
                  tok.convert_tokens_to_ids("<|endoftext|>")} - {None})
    eos = [i for i in eos if isinstance(i, int) and i >= 0] or [tok.eos_token_id]
    t0 = time.time()
    model = load_model(a.model, device)
    print(f"模型加载 {time.time() - t0:.0f}s", flush=True)

    results, lat = [], []
    for it in items:
        cat = "/".join(it["image"].split("/")[1:2])[0:] and \
            it["image"].split("/")[1]
        q = it["conversations"][0]["value"]
        img = Image.open(img_root / it["image"]).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image"},
                                            {"type": "text", "text": q}]}]
        try:
            text = proc.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True,
                                            enable_thinking=False)
        except TypeError:
            text = proc.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)
        inp = proc(text=[text], images=[img], return_tensors="pt")
        inp = {k: (v.to(device) if hasattr(v, "to") else v)
               for k, v in inp.items()}
        t1 = time.time()
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=a.max_new,
                                 do_sample=False, eos_token_id=eos,
                                 pad_token_id=eos[0])
        dt = time.time() - t1
        lat.append(dt)
        raw = proc.decode(out[0][inp["input_ids"].shape[1]:],
                          skip_special_tokens=True).strip()
        clause = clause_of(raw)
        attrs = parse_values(cat, clause or "", full_spec.get(cat, []))
        ans = render(cat, attrs, q)
        if a.hedge and attrs:
            ans = ans.replace("图中是一双", "图中看是一双") \
                     .replace("图中是一个", "图中看是一个") \
                     .replace("图中是一件", "图中看是一件") \
                     .replace("图中是一条", "图中看是一条") \
                     .replace("图中是一台", "图中看是一台") \
                     .replace("图中是一辆", "图中看是一辆") \
                     .replace("图中是一块", "图中看是一块") \
                     .replace("图中是一瓶", "图中看是一瓶") \
                     .replace("图中是", "图中看是")
        results.append({"image": it["image"], "category": cat,
                        "question": q.split("\n", 1)[-1],
                        "raw_first_sentence": clause,
                        "attributes": attrs, "answer": ans,
                        "latency_s": round(dt, 2)})

    n = len(results)
    n_attr = sum(1 for r in results if r["attributes"])
    lat.sort()
    print(f"\n{'-'*88}")
    print(f"{n} 张图  平均延迟 {sum(lat)/n:.2f}s  中位 {lat[n//2]:.2f}s  "
          f"P90 {lat[int(n*0.9)-1] if n>1 else lat[0]:.2f}s")
    print(f"读出属性的图 {n_attr}/{n} = {n_attr/n:.3f}（其余只报品类）")
    print(f"平均答案长度 {sum(len(r['answer']) for r in results)/n:.0f} 字")
    print(f"{'-'*88}\n")
    for r in results[:8]:
        print(f"[{'/'.join(r['image'].split('/')[-3:])}]")
        print(f"  Q: {r['question']}")
        print(f"  模型只输出: {r['raw_first_sentence']}")
        print(f"  抽取属性  : {r['attributes']}   ({r['latency_s']}s)")
        print(f"  最终答案  : {r['answer']}\n")

    Path(a.out).write_text(json.dumps(results, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"→ {a.out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
