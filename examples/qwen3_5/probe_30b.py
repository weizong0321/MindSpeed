#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""等 30B 权重补齐后：测它能不能在 64GB HBM 上加载并推理。

权重是 57GB bf16，HBM 64GB —— 光权重就快满了，所以重点看：
  1) device_map="auto" 会不会把一部分甩到 CPU（甩了就等于没戏，太慢）
  2) 会不会 OOM
  3) 一张图要多少秒
"""

import json
import os
import time
import traceback

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoProcessor

try:
    import torch_npu  # noqa: F401
    HAS_NPU = True
except ImportError:
    HAS_NPU = False

MODEL = os.environ.get("MODEL_DIR", "/workspace/vlm_gate/Qwen3-VL-30B-A3B-Instruct")
OUT = "/workspace/user_data/gate_30b_probe.json"
res = {"model": MODEL, "npu": HAS_NPU, "steps": []}
T0 = time.time()


def step(name, **kw):
    rec = {"step": name, "t": round(time.time() - T0, 1), **kw}
    res["steps"].append(rec)
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2)


def npu_mem():
    try:
        import subprocess
        out = subprocess.run(["npu-smi", "info", "-t", "usages"],
                             capture_output=True, text=True, timeout=20).stdout
        return out[-400:]
    except Exception as e:                                       # noqa: BLE001
        return f"npu-smi 失败 {e}"


# 0) 权重齐全吗
import glob
# --- path shim (auto-added for portability) ---
import os as _os
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
MM_DATA = _Path(_os.environ.get("MM_DATA", _REPO_ROOT / "data" / "ecommerce_multimodal"))
MM_WORK = _Path(_os.environ.get("MM_WORK", _REPO_ROOT / "work"))
MM_EX = _REPO_ROOT / "examples" / "qwen3_5"
# --- end shim ---

shards = sorted(glob.glob(f"{MODEL}/*.safetensors"))
tot = sum(os.path.getsize(p) for p in shards)
step("weights", shards=len(shards), total_GB=round(tot / 1e9, 2))

try:
    proc = AutoProcessor.from_pretrained(MODEL)
    step("processor_ok", cls=type(proc).__name__)
except Exception as e:
    step("processor_fail", err=repr(e)[:300])

try:
    from transformers import AutoModelForImageTextToText
    step("load_start")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="auto")
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    # 看有多少权重留在 CPU
    devs = {}
    for p in model.parameters():
        d = str(p.device)
        devs[d] = devs.get(d, 0) + p.numel()
    step("load_ok", params_B=round(n / 1e9, 2), device_map=devs,
         mem=npu_mem())
except Exception as e:
    step("load_fail", err=repr(e)[:600], tb=traceback.format_exc()[-1500:])

if res["steps"][-1]["step"] == "load_ok":
    try:
        from PIL import Image
        cand = None
        for root, _d, files in os.walk(
                str(_REPO_ROOT / "data/ecommerce_multimodal/sample5/images")):
            for f in sorted(files):
                if f.lower().endswith(".jpg"):
                    cand = os.path.join(root, f)
                    break
            if cand:
                break
        step("image", path=cand)
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": cand},
            {"type": "text", "text": "只输出一行 JSON："
             '{"category": "...", "color": "...", "parts": ["...", "..."]}'}]}]
        try:
            text = proc.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True,
                                            enable_thinking=False)
        except TypeError:
            text = proc.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)
        inputs = proc(text=[text], images=[Image.open(cand).convert("RGB")],
                      return_tensors="pt").to(model.device)
        t = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
        gen = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)[0]
        step("infer_ok", sec=round(time.time() - t, 1), out=gen[:300])
    except Exception as e:
        step("infer_fail", err=repr(e)[:500], tb=traceback.format_exc()[-1000:])

print("PROBE30B_DONE", flush=True)
