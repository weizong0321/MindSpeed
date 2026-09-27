#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐图属性标注（老师版，支持 30B）。

和 caption_images.py 的区别
  1) 大模型（>10GB）走 device_map="auto"，不再先全量进 CPU 再 .to(npu)
     —— 30B 有 62GB，那样很容易失败。
  2) 可以选 prompt 版本（v1 = 原来蒸馏用的那套；v2 = 强调"主体面积最大"）。
  3) 默认 batch 更小，给 30B 留显存余量。

用法
  python caption_images2.py --model <HF目录> --root <图片根目录> \\
      --out captions_30b.jsonl --batch-size 2 --prompt-version v2
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

P_V1 = """只根据这张图里【看得见】的内容输出一行 JSON。禁止输出任何分析过程、解释、编号列表或 markdown 代码块。

示例输出（仅格式示例，内容以真实图片为准）：
{"category": "太阳镜", "color": "黑", "parts": ["大框", "深色镜片", "金属镜腿"], "scene": "实拍场景"}

字段要求：
- category: 品类，最简短的中文名词，例如 运动鞋 / 皮鞋 / 双肩包 / 手提包 / 钱包 / 行李箱 / 手表 / 项链 / 耳饰 / 太阳镜 / T恤 / 长裤 / 短裤 / 冰箱 / 洗衣机 / 保健品 / 零食 / 玩具车 / 婴儿用品 / 洗护用品 / 化妆品
- color: 商品本身的【主色】，只写一个颜色词（黑/白/灰/银/金/棕/红/粉/橙/黄/绿/蓝/紫/透明）。不要写背景色。
- parts: 数组，3-5 个【图上看得见】的结构特征。例如 "系带" "魔术贴" "低帮" "厚实中底" "圆形表盘" "金属编织表带" "正面两个拉链袋" "肩带加厚" "翻盖" "万向轮" "按压泵头"
- scene: 画面类型，只能是这四个之一：白底商品图 / 实拍场景 / 手持展示 / 带包装盒

严禁出现：品牌名、型号名、价格、容量、功率、尺码、材质型号、成分、防水等级。看不清就留空。

直接输出那一行 JSON："""

P_V2 = """看图，判断【商品主体上面积最大】的那块是什么颜色。

规则：
1. 只看商品本身，不要看背景、床单、地板、包装盒、模特穿的衣服。
2. 如果商品有多个颜色，只报面积最大的那个。
3. 颜色词只能从这个表里选一个：黑 白 灰 银 金 棕 红 粉 橙 黄 绿 蓝 紫 透明

另外给出品类和 3-5 个看得见的结构特征，以及画面类型。

只输出那一行 JSON，不要解释：
{"category": "品类中文名", "color": "选一个颜色词", "parts": ["结构1", "结构2", "结构3"], "scene": "白底商品图/实拍场景/手持展示/带包装盒"}

输出："""

PROMPTS = {"v1": P_V1, "v2": P_V2}


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
    except json.JSONDecodeError:
        s = m.group(0).replace("'", '"')
        s = re.sub(r",\s*([}\]])", r"\1", s)
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="captions.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--prompt-version", choices=sorted(PROMPTS), default="v1")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    try:
        import torch_npu                                        # noqa: F401
        device = torch.device("npu:0")
    except Exception:                                           # noqa: BLE001
        device = torch.device("cpu")

    root = Path(args.root)
    imgs = sorted(root.rglob("*.jpg"))
    imgs = imgs[args.offset:args.offset + args.limit] if args.limit \
        else imgs[args.offset:]
    print(f"设备 {device}  模型 {args.model}  prompt {args.prompt_version}",
          flush=True)
    print(f"待标注 {len(imgs)} 张", flush=True)

    out = Path(args.out)
    done = set()
    if out.is_file() and not args.no_resume:
        with open(out, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["image"])
                except Exception:                            # noqa: BLE001
                    pass
        print(f"断点续跑：已有 {len(done)} 条", flush=True)
    imgs = [p for p in imgs if str(p) not in done]
    if not imgs:
        print("没有需要处理的图片", flush=True)
        return 0

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    from transformers import AutoConfig
    import transformers
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    arch = (cfg.architectures or [""])[0]
    cls = getattr(transformers, arch, None) or \
        transformers.AutoModelForImageTextToText
    big = sum(p.stat().st_size for p in Path(args.model).glob("*.safetensors"))
    print(f"架构 {arch}，权重 {big / 1e9:.1f}G", flush=True)
    if big > 10e9:
        model = cls.from_pretrained(args.model, trust_remote_code=True,
                                    dtype=torch.bfloat16,
                                    low_cpu_mem_usage=True,
                                    device_map="auto")
    else:
        try:
            model = cls.from_pretrained(args.model, dtype=torch.bfloat16,
                                        trust_remote_code=True)
        except TypeError:
            model = cls.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                        trust_remote_code=True)
        model = model.to(device)
    model = model.eval()
    print(f"模型加载完成 {time.time() - t0:.0f}s", flush=True)

    tok = processor.tokenizer
    if hasattr(tok, "padding_side"):
        tok.padding_side = "left"
    eos_ids = sorted({tok.eos_token_id,
                      tok.convert_tokens_to_ids("<|im_end|>"),
                      tok.convert_tokens_to_ids("<|endoftext|>")} - {None})
    eos_ids = [i for i in eos_ids if isinstance(i, int) and i >= 0]

    prompt = PROMPTS[args.prompt_version]
    try:
        tmpl = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"},
                                          {"type": "text", "text": prompt}]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        tmpl = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"},
                                          {"type": "text", "text": prompt}]}],
            tokenize=False, add_generation_prompt=True)

    bs = max(1, args.batch_size)
    n_ok = n_bad = n_done = 0
    with open(out, "a", encoding="utf-8") as f, torch.no_grad():
        for start in range(0, len(imgs), bs):
            chunk = imgs[start:start + bs]
            try:
                pil = [Image.open(p).convert("RGB") for p in chunk]
                inputs = processor(text=[tmpl] * len(chunk), images=pil,
                                   return_tensors="pt", padding=True)
                inputs = {k: (v.to(device) if hasattr(v, "to") else v)
                          for k, v in inputs.items()}
                out_ids = model.generate(**inputs,
                                         max_new_tokens=args.max_new_tokens,
                                         do_sample=False, eos_token_id=eos_ids,
                                         pad_token_id=eos_ids[0])
                plen = inputs["input_ids"].shape[1]
                for j, p in enumerate(chunk):
                    raw = processor.decode(out_ids[j][plen:],
                                           skip_special_tokens=True).strip()
                    attrs = extract_json(raw)
                    n_ok += 1 if attrs else 0
                    n_bad += 0 if attrs else 1
                    f.write(json.dumps({"image": str(p), "attrs": attrs,
                                        "raw": raw},
                                       ensure_ascii=False) + "\n")
                    n_done += 1
                    if n_done <= 3:
                        print(f"    {p.parent.name}/{p.name} → {attrs}",
                              flush=True)
                f.flush()
                el = time.time() - t0
                print(f"  [{n_done}/{len(imgs)}] ok={n_ok} bad={n_bad} "
                      f"{el / max(n_done, 1):.1f}s/张 "
                      f"剩余{(len(imgs) - n_done) * el / max(n_done, 1) / 60:.1f}分钟",
                      flush=True)
            except Exception as e:                           # noqa: BLE001
                print(f"  [批失败] {chunk[0].name}..: {type(e).__name__}: {e}",
                      flush=True)
                for p in chunk:
                    f.write(json.dumps({"image": str(p), "attrs": None,
                                        "raw": f"BATCH_ERROR {type(e).__name__}"},
                                       ensure_ascii=False) + "\n")
                n_bad += len(chunk)
                n_done += len(chunk)
                f.flush()
    print(f"\n完成：成功解析 {n_ok}，失败 {n_bad} → {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
