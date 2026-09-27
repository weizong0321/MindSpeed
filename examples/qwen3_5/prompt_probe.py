#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同一个 4B 老师，换几种问法，看颜色准确率能不能显著提高。

动机
    门槛实验显示 4B 老师颜色只有 ~70-73%，是整条流水线的天花板。
    如果只是"问法"不对（比如被背景/包装/模特衣服干扰），换个 prompt 就能救，
    不必等 30B。这里在同样的 30 张人工真值上对比 4 种问法。
"""

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

VARIANTS = {
    "v1_current": """只根据这张图里【看得见】的内容输出一行 JSON。禁止输出任何分析过程、解释、编号列表或 markdown 代码块。

示例输出（仅格式示例，内容以真实图片为准）：
{"category": "太阳镜", "color": "黑", "parts": ["大框", "深色镜片", "金属镜腿"], "scene": "实拍场景"}

字段要求：
- category: 品类，最简短的中文名词
- color: 商品本身的【主色】，只写一个颜色词（黑/白/灰/银/金/棕/红/粉/橙/黄/绿/蓝/紫/透明）。不要写背景色。
- parts: 数组，3-5 个【图上看得见】的结构特征。
- scene: 画面类型，只能是这四个之一：白底商品图 / 实拍场景 / 手持展示 / 带包装盒

严禁出现：品牌名、型号名、价格、容量、功率、尺码、材质型号、成分、防水等级。看不清就留空。

直接输出那一行 JSON：""",

    "v2_area": """看图，判断【商品主体上面积最大】的那块是什么颜色。

规则：
1. 只看商品本身，不要看背景、床单、地板、包装盒、模特穿的衣服。
2. 如果商品有多个颜色，只报面积最大的那个。
3. 颜色词只能从这个表里选一个：黑 白 灰 银 金 棕 红 粉 橙 黄 绿 蓝 紫 透明

只输出那一行 JSON，不要解释：
{"category": "品类中文名", "color": "选一个颜色词", "parts": ["看得见的结构1", "结构2", "结构3"], "scene": "白底商品图/实拍场景/手持展示/带包装盒"}

输出：""",

    "v3_two": """看图，指出商品主体的颜色。

先想清楚商品主体是哪一块（忽略背景、包装、模特衣着），然后给出占比最大的两个颜色（按面积从大到小）。
颜色词只能从：黑 白 灰 银 金 棕 红 粉 橙 黄 绿 蓝 紫 透明

只输出一行 JSON，不要解释：
{"category": "品类", "color": "第一主色", "color2": "第二主色", "parts": ["结构1", "结构2", "结构3"], "scene": "白底商品图/实拍场景/手持展示/带包装盒"}

输出：""",

    "v4_deliberate": """你会看到一张电商商品图。请判断商品主体的主色。

请按这个顺序思考，但只输出最后一行 JSON：
(1) 商品主体是什么？
(2) 背景/包装/模特衣服是什么颜色？（这些不算）
(3) 主体上面积最大的颜色是什么？

颜色词只能从：黑 白 灰 银 金 棕 红 粉 橙 黄 绿 蓝 紫 透明

最终只输出：
{"category": "品类", "color": "颜色", "parts": ["结构1", "结构2", "结构3"], "scene": "白底商品图/实拍场景/手持展示/带包装盒"}

输出：""",
}

COLOR_ALTS = {
    "银色": ["银色", "银白", "银灰", "银"], "棕色": ["棕色", "棕", "褐色", "咖啡色", "深棕"],
    "灰色": ["灰色", "灰", "深灰", "银灰", "灰白"], "黑色": ["黑色", "黑", "纯黑"],
    "白色": ["白色", "白"], "蓝色": ["蓝色", "蓝", "深蓝", "藏青"],
    "红色": ["红色", "红", "正红", "酒红"], "绿色": ["绿色", "绿", "墨绿", "深绿", "青色", "蓝绿"],
    "黄色": ["黄色", "黄"], "粉色": ["粉色", "粉", "蓝", "蓝色"], "透明": ["透明", "无色", "白色", "白"],
}


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
    ap.add_argument("--model", default="/workspace/shared_assets/models/Qwen/Qwen3.5-4B")
    ap.add_argument("--out", default="/workspace/user_data/gate_prompt_probe.json")
    a = ap.parse_args()

    items = json.loads((Path(BASE) / "sample5/gate_images.json").read_text("utf-8"))
    truth = {t["id"]: t for t in json.loads(
        (Path(BASE) / "sample5/gate_truth.json").read_text("utf-8"))["items"]}

    try:
        import torch_npu                                        # noqa: F401
        device = torch.device("npu:0")
    except Exception:                                           # noqa: BLE001
        device = torch.device("cpu")

    proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
    tok = proc.tokenizer
    eos = sorted({tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"),
                  tok.convert_tokens_to_ids("<|endoftext|>")} - {None})
    eos = [i for i in eos if isinstance(i, int) and i >= 0]
    import transformers
    cls = getattr(transformers, "Qwen3_5ForConditionalGeneration",
                  transformers.AutoModelForImageTextToText)
    model = cls.from_pretrained(a.model, dtype=torch.bfloat16,
                                trust_remote_code=True).eval().to(device)

    results = {}
    for name, prompt in VARIANTS.items():
        hits, n_parse, rows = 0, 0, []
        t0 = time.time()
        for i, it in enumerate(items, 1):
            tid = f"A{i:02d}"
            t = truth[tid]
            img = Image.open(Path(BASE) / "sample5" / it["image"]).convert("RGB")
            msgs = [{"role": "user", "content": [{"type": "image"},
                                                {"type": "text", "text": prompt}]}]
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
                out = model.generate(**inp, max_new_tokens=120, do_sample=False,
                                     eos_token_id=eos, pad_token_id=eos[0])
            raw = proc.decode(out[0][inp["input_ids"].shape[1]:],
                              skip_special_tokens=True).strip()
            j = extract_json(raw)
            if j:
                n_parse += 1
            alts = COLOR_ALTS.get(t["truth_color"], [t["truth_color"]])
            said = (j or {}).get("color", "") if j else ""
            said2 = (j or {}).get("color2", "") if j else ""
            ok = any(x in said for x in alts)
            ok2 = ok or any(x in said2 for x in alts)
            hits += ok
            rows.append({"id": tid, "truth": t["truth_color"], "said": said,
                         "said2": said2, "hit": ok, "hit_top2": ok2,
                         "ambiguous": t["ambiguous"], "raw": raw[:200]})
            if i % 10 == 0:
                print(f"  [{name}] {i}/30  命中 {hits / i:.2f}", flush=True)
        unamb = [r for r in rows if not r["ambiguous"]]
        results[name] = {
            "top1": round(hits / len(rows), 3),
            "top1_unambiguous": round(
                sum(r["hit"] for r in unamb) / len(unamb), 3),
            "top2": round(sum(r["hit_top2"] for r in rows) / len(rows), 3),
            "parsed": f"{n_parse}/{len(rows)}",
            "sec_per_img": round((time.time() - t0) / len(rows), 1),
            "rows": rows,
        }
        print(f"[{name}] top1={results[name]['top1']:.3f} "
              f"非歧义={results[name]['top1_unambiguous']:.3f} "
              f"top2={results[name]['top2']:.3f} 解析={results[name]['parsed']}",
              flush=True)

    print("\n" + "=" * 68)
    print(f"{'问法':<16s} {'top1':>7s} {'top1(非歧义)':>13s} {'top2':>7s} {'解析':>8s}")
    print("-" * 68)
    for k, v in results.items():
        print(f"{k:<16s} {v['top1']:7.3f} {v['top1_unambiguous']:13.3f} "
              f"{v['top2']:7.3f} {v['parsed']:>8s}")
    print("=" * 68)
    Path(a.out).write_text(json.dumps(results, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"→ {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
