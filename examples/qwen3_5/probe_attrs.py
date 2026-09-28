#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用闭词表属性问老师，看这些属性可不可靠。

对每张图，按它的品类生成一组互斥判断题，让两个老师（4B / 30B）分别回答，
然后比较一致性。一致性高的属性说明"答案由图唯一决定"，才值得拿来做训练目标。

用法
    python probe_attrs.py --model <HF目录> --tag t4b --out attr_t4b.json
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
from attr_spec import prompt_for  # noqa: E402

import torch
from PIL import Image
from transformers import AutoProcessor

BASE = str(_REPO_ROOT / "data/ecommerce_multimodal/sample5")

def extract_json(text):
    text = text.strip()
    text = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text, flags=re.S | re.I)
    text = re.sub(r"^.*?<｜end▁of▁thinking｜>", "", text, flags=re.S)
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:                                          # noqa: BLE001
        return None

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--list", default=f"{BASE}/gate_images.json",
                    help="图片清单 JSON，元素含 image / category")
    ap.add_argument("--root", default=BASE, help="图片相对路径的根目录")
    a = ap.parse_args()

    items = json.loads(Path(a.list).read_text("utf-8"))
    root = Path(a.root)
    print(f"清单 {a.list}：{len(items)} 张", flush=True)

    # ---- 断点续跑：每张图算完立刻追加到 .jsonl，进程被杀也不丢进度 ----
    jl = Path(a.out + ".jsonl")
    done = {}
    if jl.exists():
        for line in jl.open(encoding="utf-8"):
            try:
                r = json.loads(line)
                done[(r.get("category"), r.get("id"))] = r
            except Exception:                                  # noqa: BLE001
                pass
        print(f"断点续跑：已有 {len(done)} 条，跳过", flush=True)
    jl_f = jl.open("a", encoding="utf-8")

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
    import transformers
    cls = getattr(transformers, "Qwen3_5ForConditionalGeneration", None)
    cfg = json.loads((Path(a.model) / "config.json").read_text())
    arch = (cfg.get("architectures") or [""])[0]
    cls = getattr(transformers, arch, None) or \
        transformers.AutoModelForImageTextToText
    big = sum(p.stat().st_size for p in Path(a.model).glob("*.safetensors"))
    print(f"[{a.tag}] {a.model}  权重 {big / 1e9:.1f}G  架构 {arch}", flush=True)
    if big > 10e9:
        model = cls.from_pretrained(a.model, trust_remote_code=True,
                                    dtype=torch.bfloat16,
                                    low_cpu_mem_usage=True,
                                    device_map="auto").eval()
    else:
        model = cls.from_pretrained(a.model, dtype=torch.bfloat16,
                                    trust_remote_code=True).eval().to(device)

    rows, t0 = [], time.time()
    for i, it in enumerate(items, 1):
        tid = it.get("id") or f"A{i:02d}"
        cat = it["category"]
        if (cat, tid) in done:                    # 断点续跑：这张已算过
            continue
        q, keys = prompt_for(cat)
        if q is None:
            rec = {"id": tid, "category": cat, "attrs": None,
                   "raw": "NO_SPEC", "keys": []}
            rows.append(rec)
            jl_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            jl_f.flush()
            continue
        img = Image.open(root / it["image"]).convert("RGB")
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
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=110, do_sample=False,
                                 eos_token_id=eos, pad_token_id=eos[0])
        raw = proc.decode(out[0][inp["input_ids"].shape[1]:],
                          skip_special_tokens=True).strip()
        j = extract_json(raw)
        rec = {"id": tid, "category": cat, "keys": keys,
               "attrs": j, "raw": raw[:300]}
        rows.append(rec)
        jl_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        jl_f.flush()                              # 每张都落盘
        if i % 5 == 0 or i == len(items):
            el = time.time() - t0
            print(f"  [{len(rows)}/{len(items)}] {el / max(len(rows),1):.1f}s/图",
                  flush=True)

    jl_f.close()

    # 汇总成最终的 JSON 数组（供下游脚本读取）：以 jsonl 为准，保证完整
    allrows = []
    if jl.exists():
        for line in jl.open(encoding="utf-8"):
            try:
                allrows.append(json.loads(line))
            except Exception:                                  # noqa: BLE001
                pass
    Path(a.out).write_text(json.dumps(allrows, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"→ {a.out}（{len(allrows)} 条；断点文件 {jl}）", flush=True)
    print("PROBE_ATTRS_DONE", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
