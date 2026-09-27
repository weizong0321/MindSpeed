#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""属性读取器：加载 HF 模型 → 只生成第一句属性陈述 → 反解 → 渲染正文。

为什么这样拆（本项目实测依据）
    · 颜色：4B/30B 老师都只有 0.67–0.73 正确率，换更大模型无改善 → 不断言
    · 自由文本部件：两个老师同一问法仅 13.3% 完全一致（目标欠定）→ 弃用
    · 闭词表互斥属性：两老师一致率 0.85–1.00，学生整组全对 75.6% → 采用
    所以模型只输出闭词表属性，文案由 render_answer 确定性生成。
"""

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attr_render import parse_values, render_values   # noqa: E402
from render_answer import render, CAT_CN, constraints_of  # noqa: E402
import attr_spec                                        # noqa: E402

KNOWN_CATS = list(attr_spec.ATTRS.keys())
CN2CAT = {v: k for k, v in CAT_CN.items()}


def spec_for(cat):
    """把 attr_spec.ATTRS 转成 parse_values 需要的 [{key, options}]"""
    return [{"key": k, "options": list(o)}
            for k, _desc, o in attr_spec.ATTRS.get(cat, [])]


def clause_of(text):
    m = re.search(r"图中为(.*?)。", text or "")
    return m.group(1).strip() if m else (text or "").strip()


def cat_from_path(path):
    """从 images/<cat>/<sku>/<file> 推断品类"""
    parts = Path(path).parts
    for p in parts:
        if p in KNOWN_CATS:
            return p
    return None


class AttrReader:
    def __init__(self, model_dir, device=None, max_new=48):
        import torch
        from transformers import AutoProcessor, AutoConfig
        import transformers

        self.torch = torch
        self.max_new = max_new
        if device is None:
            try:
                import torch_npu                            # noqa: F401
                device = torch.device("npu:0")
            except Exception:                               # noqa: BLE001
                device = torch.device("cpu")
        self.device = device

        self.proc = AutoProcessor.from_pretrained(model_dir,
                                                  trust_remote_code=True)
        tok = self.proc.tokenizer
        if hasattr(tok, "padding_side"):
            tok.padding_side = "left"
        # <|im_end|> 与 eos_token_id 不同，必须都给，否则生成不终止
        eos = {tok.eos_token_id,
               tok.convert_tokens_to_ids("<|im_end|>"),
               tok.convert_tokens_to_ids("<|endoftext|>")} - {None}
        self.eos = [i for i in eos if isinstance(i, int) and i >= 0]
        if not self.eos:
            self.eos = [tok.eos_token_id]

        cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        arch = (cfg.architectures or [""])[0]
        cls = getattr(transformers, arch, None) or \
            transformers.AutoModelForImageTextToText
        big = sum(p.stat().st_size
                  for p in Path(model_dir).glob("*.safetensors"))
        if big > 10e9:
            self.model = cls.from_pretrained(
                model_dir, trust_remote_code=True, dtype=torch.bfloat16,
                low_cpu_mem_usage=True, device_map="auto").eval()
        else:
            try:
                self.model = cls.from_pretrained(
                    model_dir, dtype=torch.bfloat16,
                    trust_remote_code=True)
            except TypeError:
                self.model = cls.from_pretrained(
                    model_dir, torch_dtype=torch.bfloat16,
                    trust_remote_code=True)
            self.model = self.model.eval().to(device)

    def _generate(self, image_path, question):
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": question}]}]
        try:
            text = self.proc.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            text = self.proc.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        inp = self.proc(text=[text], images=[img], return_tensors="pt")
        inp = {k: (v.to(self.device) if hasattr(v, "to") else v)
               for k, v in inp.items()}
        with self.torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=self.max_new,
                                      do_sample=False, eos_token_id=self.eos,
                                      pad_token_id=self.eos[0])
        return self.proc.decode(out[0][inp["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()

    def read(self, image_path, question="", cat=None):
        t0 = time.time()
        raw = self._generate(image_path, question)
        clause = clause_of(raw)

        cat = cat or cat_from_path(image_path)
        if cat is None:
            # 退而求其次：用模型输出的品类名词反查
            head = clause.split("，")[0].strip()
            cat = CN2CAT.get(head)
        if cat is None or cat not in attr_spec.ATTRS:
            # 无法确定品类 → 不断言属性，只给免责说明（安全降级）
            ans = ("图中这件商品的品类未能可靠识别，"
                   "更细的参数需要以商品页标注为准。")
            return {}, ans

        spec = spec_for(cat)
        attrs = parse_values(cat, clause, spec)
        ans = render(cat, attrs, question)
        # 属性也做弱化措辞：整组正确率 75.6%，不宜说成断言
        if attrs:
            ans = ans.replace("图中是一双", "图中看是一双") \
                     .replace("图中是一个", "图中看是一个") \
                     .replace("图中是一件", "图中看是一件") \
                     .replace("图中是一条", "图中看是一条") \
                     .replace("图中是一台", "图中看是一台") \
                     .replace("图中是一辆", "图中看是一辆") \
                     .replace("图中是一块", "图中看是一块") \
                     .replace("图中是一瓶", "图中看是一瓶") \
                     .replace("图中是", "图中看是")
        return attrs, ans
