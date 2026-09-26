#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用大模型给每张图打"可见属性"标注（蒸馏的数据来源）。

为什么只让它输出结构化属性，而不让它写整段答案
    ① 格式合规由我们的模板保证，不依赖大模型发挥
    ② 属性（颜色/部件/画面）逐图不同 → 答案逐图不同 → "背模板"这条路被堵死
    ③ 属性可客观校验（颜色对不对、部件在不在），给了一个能算的指标

输出 JSONL，每行 {"image":..., "attrs": {...}, "raw": "..."}，可断点续跑。

用法
    python caption_images.py --model <HF目录> --root <图片根> --out captions.jsonl --limit 3
"""

import argparse
import json
import os
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

PROMPT = """只根据这张图里【看得见】的内容输出一行 JSON。禁止输出任何分析过程、解释、编号列表或 markdown 代码块。

示例输出（仅格式示例，内容以真实图片为准）：
{"category": "太阳镜", "color": "黑", "parts": ["大框", "深色镜片", "金属镜腿"], "scene": "实拍场景"}

字段要求：
- category: 品类，最简短的中文名词，例如 运动鞋 / 皮鞋 / 双肩包 / 手提包 / 钱包 / 行李箱 / 手表 / 项链 / 耳饰 / 太阳镜 / T恤 / 长裤 / 短裤 / 冰箱 / 洗衣机 / 保健品 / 零食 / 玩具车 / 婴儿用品 / 洗护用品 / 化妆品
- color: 商品本身的【主色】，只写一个颜色词（黑/白/灰/银/金/棕/红/粉/橙/黄/绿/蓝/紫/透明）。不要写背景色。
- parts: 数组，3-5 个【图上看得见】的结构特征。例如 "系带" "魔术贴" "低帮" "厚实中底" "圆形表盘" "金属编织表带" "正面两个拉链袋" "肩带加厚" "翻盖" "万向轮" "按压泵头"
- scene: 画面类型，只能是这四个之一：白底商品图 / 实拍场景 / 手持展示 / 带包装盒

严禁出现：品牌名、型号名、价格、容量、功率、尺码、材质型号、成分、防水等级。看不清就留空。

直接输出那一行 JSON："""


def extract_json(text):
    """从模型输出里抠出第一个 JSON 对象（先剥掉思考块）"""
    text = text.strip()
    # 去掉  thinking...<｜end▁of▁thinking｜> 之类的思考段
    text = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text, flags=re.S | re.I)
    text = re.sub(r"^.*?<｜end▁of▁thinking｜>", "", text, flags=re.S)
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        # 常见问题：单引号 / 尾随逗号
        s = m.group(0).replace("'", '"')
        s = re.sub(r",\s*([}\]])", r"\1", s)
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--root", required=True, help="图片根目录（递归找 jpg）")
    ap.add_argument("--out", default="captions.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=8, help="一次喂几张图（提速关键）")
    args = ap.parse_args()

    try:
        import torch_npu                                        # noqa: F401
        device = torch.device("npu:0")
    except Exception:                                           # noqa: BLE001
        device = torch.device("cpu")
    print(f"设备 {device}，模型 {args.model}", flush=True)

    root = Path(args.root)
    imgs = sorted(p for p in root.rglob("*.jpg"))
    if args.limit:
        imgs = imgs[args.offset:args.offset + args.limit]
    else:
        imgs = imgs[args.offset:]
    print(f"待标注 {len(imgs)} 张", flush=True)

    out = Path(args.out)
    done = set()
    if out.is_file() and args.offset == 0:
        with open(out, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["image"])
                except Exception:                            # noqa: BLE001
                    pass
        print(f"断点续跑：已有 {len(done)} 条", flush=True)
    imgs = [p for p in imgs if str(p) not in done]

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    arch = (cfg.architectures or [""])[0]
    import transformers
    cls = getattr(transformers, arch, None)
    print(f"架构 {arch} → {cls}", flush=True)
    if cls is None:
        from transformers import AutoModelForImageTextToText as cls   # type: ignore
    try:
        model = cls.from_pretrained(args.model, dtype=torch.bfloat16,
                                    trust_remote_code=True)
    except TypeError:
        model = cls.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                    trust_remote_code=True)
    model = model.eval().to(device)
    print(f"模型加载完成 {time.time() - t0:.0f}s", flush=True)

    tok = processor.tokenizer
    if hasattr(tok, "padding_side"):
        tok.padding_side = "left"          # 解码器模型批量生成必须左填充
    eos_ids = sorted({tok.eos_token_id,
                      tok.convert_tokens_to_ids("<|im_end|>"),
                      tok.convert_tokens_to_ids("<|endoftext|>")} - {None})
    eos_ids = [i for i in eos_ids if isinstance(i, int) and i >= 0]

    # 预渲染提示词模板（所有图共用同一段指令，只有图片不同）
    try:
        tmpl = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"},
                                          {"type": "text", "text": PROMPT}]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        tmpl = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"},
                                          {"type": "text", "text": PROMPT}]}],
            tokenize=False, add_generation_prompt=True)

    bs = max(1, args.batch_size)
    n_ok = n_bad = 0
    n_done = 0
    with open(out, "a", encoding="utf-8") as f, torch.no_grad():
        for start in range(0, len(imgs), bs):
            chunk = imgs[start:start + bs]
            try:
                pil = [Image.open(p).convert("RGB") for p in chunk]
                inputs = processor(text=[tmpl] * len(chunk), images=pil,
                                   return_tensors="pt", padding=True)
                inputs = {k: (v.to(device) if hasattr(v, "to") else v)
                          for k, v in inputs.items()}
                out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                         do_sample=False, eos_token_id=eos_ids,
                                         pad_token_id=eos_ids[0])
                prompt_len = inputs["input_ids"].shape[1]
                for j, p in enumerate(chunk):
                    raw = processor.decode(out_ids[j][prompt_len:],
                                           skip_special_tokens=True).strip()
                    attrs = extract_json(raw)
                    n_ok += 1 if attrs else 0
                    n_bad += 0 if attrs else 1
                    f.write(json.dumps({"image": str(p), "attrs": attrs, "raw": raw},
                                       ensure_ascii=False) + "\n")
                    n_done += 1
                    if n_done <= 3:
                        print(f"    {p.parent.name}/{p.name} → {attrs}", flush=True)
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
