#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""门槛实验：拿人工真值给"老师/学生"打分（非循环论证）。

两种模式
  caption : 用蒸馏时给老师用的同一套 prompt 问模型 → 比 color/parts
  answer  : 用评测集里的真实问题问模型（就是线上会问的问题）→ 比 color/parts

用法
  python gate_eval.py --model /path --mode answer --tag s6 \
      --out /workspace/user_data/gate_s6.json
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
GATE = "/workspace/user_data"

PROMPT = """只根据这张图里【看得见】的内容输出一行 JSON。禁止输出任何分析过程、解释、编号列表或 markdown 代码块。

示例输出（仅格式示例，内容以真实图片为准）：
{"category": "太阳镜", "color": "黑", "parts": ["大框", "深色镜片", "金属镜腿"], "scene": "实拍场景"}

字段要求：
- category: 品类，最简短的中文名词
- color: 商品本身的【主色】，只写一个颜色词（黑/白/灰/银/金/棕/红/粉/橙/黄/绿/蓝/紫/透明）。不要写背景色。
- parts: 数组，3-5 个【图上看得见】的结构特征。
- scene: 画面类型，只能是这四个之一：白底商品图 / 实拍场景 / 手持展示 / 带包装盒

严禁出现：品牌名、型号名、价格、容量、功率、尺码、材质型号、成分、防水等级。看不清就留空。

直接输出那一行 JSON："""


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


def load_model(model_dir, device):
    from transformers import AutoConfig
    import transformers
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    arch = (cfg.architectures or [""])[0]
    cls = getattr(transformers, arch, None)
    if cls is None:
        cls = transformers.AutoModelForImageTextToText
    # 大模型（>10GB，比如 30B）用 device_map="auto" 直接落到 NPU，
    # 否则会先在 CPU 全量物化再 .to()，30B 这样容易失败
    big = sum(p.stat().st_size for p in Path(model_dir).glob("*.safetensors"))
    if big > 10e9:
        print(f"    大模型 {big / 1e9:.1f}G，用 device_map=auto 加载", flush=True)
        m = cls.from_pretrained(model_dir, trust_remote_code=True,
                                dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                device_map="auto")
        return m.eval()
    kw = dict(trust_remote_code=True, dtype=torch.bfloat16)
    try:
        m = cls.from_pretrained(model_dir, **kw)
    except TypeError:
        kw.pop("dtype")
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
    ap.add_argument("--mode", choices=["caption", "answer"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    items = json.loads(Path(f"{BASE}/sample5/gate_images.json").read_text("utf-8"))
    truth = {t["id"]: t for t in json.loads(
        Path(f"{BASE}/sample5/gate_truth.json").read_text("utf-8"))["items"]}
    assert len(items) == len(truth), f"{len(items)} vs {len(truth)}"

    # 问题：同一个子样式的图用同一个问题
    # 注意：gate_images.json 里的 sku 没有品类前缀（如 shoes_kids_003），
    # 而 eval.json 的 key 是 "shoes_kids/shoes_kids_003"，所以统一用最后一段匹配
    q_by_sku, q_any = {}, "请描述这件商品，突出它的外观特征。"
    evp = Path(f"{BASE}/sample6/eval.json")
    if evp.exists():
        ev = json.loads(evp.read_text("utf-8"))
        for s in ev:
            sub = s["image"].split("/")[2]
            q_by_sku.setdefault(sub, s["conversations"][0]["value"])
    print(f"    可用问题 {len(q_by_sku)} 个子样式", flush=True)

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
    print(f"[{a.tag}] 加载 {a.model} → {device}", flush=True)
    model = load_model(a.model, device)

    rows, t0 = [], time.time()
    sel = items[:a.limit] if a.limit else items
    for i, it in enumerate(sel, 1):
        tid = f"A{i:02d}"
        t = truth[tid]
        sub = it["sku"].split("/")[-1]
        q = PROMPT if a.mode == "caption" else q_by_sku.get(sub)
        if q is None:
            print(f"    !! {sub} 没有对应问题，跳过", flush=True)
            continue
        raw = ask(model, proc, eos, os.path.join(f"{BASE}/sample5", it["image"]),
                  q, device)
        low = raw
        c_alts = [x for x in t["color_alts"] if x in low]
        p_hit = [p for p in t["parts"] if p in low]
        if a.mode == "caption":
            j = extract_json(raw) or {}
            said = f"color={j.get('color')} parts={j.get('parts')}"
        else:
            said = ""
        rows.append({
            "id": tid, "sku": it["sku"], "ambiguous": t["ambiguous"],
            "truth_color": t["truth_color"], "truth_parts": t["parts"],
            "color_hit": bool(c_alts), "color_alts_hit": c_alts,
            "parts_hit": round(len(p_hit) / len(t["parts"]), 3),
            "parts_hit_list": p_hit, "said": said, "raw": raw[:600],
        })
        if i % 5 == 0 or i == len(sel):
            el = time.time() - t0
            print(f"  [{i}/{len(sel)}] 色 {sum(r['color_hit'] for r in rows) / len(rows):.2f}"
                  f"  部件 {sum(r['parts_hit'] for r in rows) / len(rows):.2f}"
                  f"  {el / i:.1f}s/张", flush=True)

    avg = lambda x: sum(x) / len(x) if x else 0.0
    unamb = [r for r in rows if not r["ambiguous"]]
    summary = {
        "tag": a.tag, "model": a.model, "mode": a.mode, "n": len(rows),
        "color_hit": round(avg([1.0 if r["color_hit"] else 0.0 for r in rows]), 3),
        "parts_hit": round(avg([r["parts_hit"] for r in rows]), 3),
        "color_hit_unambiguous": round(
            avg([1.0 if r["color_hit"] else 0.0 for r in unamb]), 3),
        "parts_hit_unambiguous": round(avg([r["parts_hit"] for r in unamb]), 3),
        "n_unambiguous": len(unamb),
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    print("\n" + "=" * 70)
    print(f"[{a.tag}] mode={a.mode}  {a.model}")
    print(f"  颜色命中 {summary['color_hit']:.3f}   部件命中 {summary['parts_hit']:.3f}"
          f"   ({len(rows)} 张)")
    print(f"  去掉歧义图后 颜色 {summary['color_hit_unambiguous']:.3f} "
          f"部件 {summary['parts_hit_unambiguous']:.3f}  ({len(unamb)} 张)")
    print("=" * 70)
    print("\n逐张（色/部件）：")
    for r in rows:
        flag = "?" if r["ambiguous"] else " "
        print(f"  {r['id']}{flag} {r['sku']:38s} 色{'✓' if r['color_hit'] else '✗'}"
              f" {r['truth_color']:>4s} 部件{r['parts_hit']:.2f} "
              f"{'|' + r['said'] if r['said'] else ''}")

    Path(a.out).write_text(json.dumps(
        {"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"→ {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
