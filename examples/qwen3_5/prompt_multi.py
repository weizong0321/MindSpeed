#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""路线甲第一步：找出最好的"多色"问法。

背景
    强制只报一个主色，对两色商品本身就是错的（黄车+绿饰条、黑表带+银壳）。
    之前 v3 问法（要两个颜色）在 30 张人工真值上覆盖率达到 0.767，而单色只有 0.667。
    这里系统比几种"列出颜色"的问法，按 top1/top2/top3 覆盖率选最好的。

同时输出每个模型说的颜色集合，方便人工核对"第二个颜色是不是真的存在"（精确率）。
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
PALETTE = "黑 白 灰 银 金 棕 红 粉 橙 黄 绿 蓝 紫 透明"

VARIANTS = {
    # 基线：只要一个主色（就是现在线上用的那套）
    "single": """只根据这张图里【看得见】的内容输出一行 JSON。禁止输出任何分析过程、解释、编号列表或 markdown 代码块。

字段要求：
- category: 品类，最简短的中文名词
- color: 商品本身的【主色】，只写一个颜色词（黑/白/灰/银/金/棕/红/粉/橙/黄/绿/蓝/紫/透明）。不要写背景色。
- parts: 数组，3-5 个【图上看得见】的结构特征。
- scene: 画面类型，只能是这四个之一：白底商品图 / 实拍场景 / 手持展示 / 带包装盒

严禁出现：品牌名、型号名、价格、容量、功率、尺码、材质型号、成分、防水等级。

直接输出那一行 JSON：""",

    # 甲-1：列出商品主体上的颜色，按面积从大到小
    "list_all": f"""看图，列出【商品主体】上的颜色，按面积从大到小排列，最多 3 个。

规则：
1. 只看商品本身，不要看背景、床单、地板、包装盒、模特穿的衣服。
2. 颜色词只能从这个表里选：{PALETTE}
3. 如果商品确实只有一种颜色，就只写一个。

只输出一行 JSON，不要解释：
{{"category": "品类", "colors": ["最大面积的颜色", "第二颜色"], "parts": ["结构1", "结构2", "结构3"], "scene": "白底商品图/实拍场景/手持展示/带包装盒"}}

输出：""",

    # 甲-2：明确区分主色和次色
    "primary_secondary": f"""看图，指出【商品主体】的配色。

第一步：想清楚商品主体是哪一块（忽略背景、包装、模特衣着）。
第二步：说出主体上面积最大的颜色（主色），以及其余看得见的颜色（次色）。

颜色词只能从：{PALETTE}

只输出一行 JSON：
{{"category": "品类", "primary": "主色", "secondary": ["次色1", "次色2"], "parts": ["结构1", "结构2", "结构3"], "scene": "白底商品图/实拍场景/手持展示/带包装盒"}}

输出：""",

    # 甲-3：要求先描述各颜色的部位，再给列表（更细的推理）
    "by_part": f"""看图，判断商品主体各部位的颜色。

请逐个部位说：例如"鞋面是黑色，鞋底是白色"。
然后汇总成颜色列表，按面积从大到小，最多 3 个。

颜色词只能从：{PALETTE}

最终只输出一行 JSON（不要输出上面的分析过程）：
{{"category": "品类", "colors": ["最大面积颜色", "第二颜色"], "parts": ["结构1", "结构2", "结构3"], "scene": "白底商品图/实拍场景/手持展示/带包装盒"}}

输出：""",
}

COLOR_ALTS = {
    "银色": ["银", "银色", "银白", "银灰"], "棕色": ["棕", "棕色", "褐色", "咖啡色", "深棕"],
    "灰色": ["灰", "灰色", "深灰", "银灰", "灰白"], "黑色": ["黑", "黑色", "纯黑"],
    "白色": ["白", "白色"], "蓝色": ["蓝", "蓝色", "深蓝", "藏青"],
    "红色": ["红", "红色", "正红", "酒红"], "绿色": ["绿", "绿色", "墨绿", "深绿", "青色", "蓝绿"],
    "黄色": ["黄", "黄色"], "粉色": ["粉", "粉色", "蓝", "蓝色"], "透明": ["透明", "无色", "白色", "白"],
}
ALL_COLORS = ["黑", "白", "灰", "银", "金", "棕", "红", "粉", "橙", "黄",
              "绿", "蓝", "紫", "透明", "米", "卡其", "驼", "青", "咖啡", "褐"]


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


def colors_of(j):
    """从各种 schema 里抽出颜色列表（有序）"""
    if not j:
        return []
    out = []
    for key in ("colors", "secondary"):
        v = j.get(key)
        if isinstance(v, list):
            out += [str(x) for x in v if x]
        elif isinstance(v, str) and v:
            out.append(v)
    for key in ("color", "primary", "color1", "color2"):
        v = j.get(key)
        if isinstance(v, str) and v:
            out.append(v)
    # 去重保序
    seen, res = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            res.append(c)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/shared_assets/models/Qwen/Qwen3.5-4B")
    ap.add_argument("--out", default="/workspace/user_data/gate_multicolor.json")
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
    cfg_cls = getattr(transformers, "Qwen3_5ForConditionalGeneration",
                      transformers.AutoModelForImageTextToText)
    model = cfg_cls.from_pretrained(a.model, dtype=torch.bfloat16,
                                    trust_remote_code=True).eval().to(device)

    results = {}
    for name, prompt in VARIANTS.items():
        rows, t0 = [], time.time()
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
                out = model.generate(**inp, max_new_tokens=140, do_sample=False,
                                     eos_token_id=eos, pad_token_id=eos[0])
            raw = proc.decode(out[0][inp["input_ids"].shape[1]:],
                              skip_special_tokens=True).strip()
            j = extract_json(raw)
            cs = colors_of(j)
            alts = COLOR_ALTS.get(t["truth_color"], [t["truth_color"]])

            def hit(k):
                return any(any(x in c for x in alts) for c in cs[:k])

            rows.append({"id": tid, "truth": t["truth_color"],
                         "ambiguous": t["ambiguous"], "colors": cs,
                         "top1": hit(1), "top2": hit(2), "top3": hit(3),
                         "n_colors": len(cs), "raw": raw[:200]})
        n = len(rows)
        unamb = [r for r in rows if not r["ambiguous"]]
        results[name] = {
            "top1": round(sum(r["top1"] for r in rows) / n, 3),
            "top2": round(sum(r["top2"] for r in rows) / n, 3),
            "top3": round(sum(r["top3"] for r in rows) / n, 3),
            "top1_unamb": round(sum(r["top1"] for r in unamb) / len(unamb), 3),
            "top2_unamb": round(sum(r["top2"] for r in unamb) / len(unamb), 3),
            "avg_colors": round(sum(r["n_colors"] for r in rows) / n, 2),
            "sec_per_img": round((time.time() - t0) / n, 1),
            "rows": rows,
        }
        v = results[name]
        print(f"[{name:18s}] top1={v['top1']:.3f} top2={v['top2']:.3f} "
              f"top3={v['top3']:.3f} | 非歧义 top1={v['top1_unamb']:.3f} "
              f"top2={v['top2_unamb']:.3f} | 平均给{v['avg_colors']}个色 "
              f"{v['sec_per_img']}s/张", flush=True)

    print("\n" + "=" * 78)
    print(f"{'问法':<20s} {'top1':>6s} {'top2':>6s} {'top3':>6s} "
          f"{'非歧义top1':>10s} {'非歧义top2':>10s} {'颜色数':>7s}")
    print("-" * 78)
    for k, v in results.items():
        print(f"{k:<20s} {v['top1']:6.3f} {v['top2']:6.3f} {v['top3']:6.3f} "
              f"{v['top1_unamb']:10.3f} {v['top2_unamb']:10.3f} "
              f"{v['avg_colors']:7.2f}")
    print("=" * 78)

    best = max(results, key=lambda k: results[k]["top2"])
    print(f"\n按 top2 覆盖率最佳: {best} ({results[best]['top2']:.3f})")

    print(f"\n=== {best} 逐图（人工核对第二个颜色是否真实存在）===")
    for r in results[best]["rows"]:
        amb = "?" if r["ambiguous"] else " "
        print(f"  {r['id']}{amb} 真值{r['truth']:>4s} → {r['colors']}")

    Path(a.out).write_text(json.dumps(results, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"\n→ {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
