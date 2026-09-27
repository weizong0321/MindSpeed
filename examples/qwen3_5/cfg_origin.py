#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比"我新建的 0.8B 配置"与"上游既有的其他尺寸配置"，看哪些是我引入的。"""

import re
import sys
from pathlib import Path
# --- path shim (auto-added for portability) ---
import os as _os
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
MM_DATA = _Path(_os.environ.get("MM_DATA", _REPO_ROOT / "data" / "ecommerce_multimodal"))
MM_WORK = _Path(_os.environ.get("MM_WORK", _REPO_ROOT / "work"))
MM_EX = _REPO_ROOT / "examples" / "qwen3_5"
# --- end shim ---


for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

D = Path(str(_REPO_ROOT / "examples/qwen3_5"))
KEYS = ["param_dtype", "reduce_dtype", "recompute", "attn_implementation",
        "use_triton_gdn", "init_model_with_meta_device", "adam_fused",
        "optimizer", "image_max_pixels", "image_min_pixels", "cutoff_len",
        "template", "enable_thinking", "micro_batch_size", "lr",
        "lr_decay_style", "clip_grad", "weight_decay", "seed",
        "enable_chunk_loss", "freeze", "router_aux_loss_coef"]

# 0.8B 是我在 d14311de 里新建的；其余尺寸是上游既有
mine = D / "qwen3_5_0.8B_config.yaml"
others = [p for p in sorted(D.glob("qwen3_5_*B_config.yaml"))
          if p.name != mine.name]


def get(p, k):
    txt = p.read_text(encoding="utf-8", errors="replace")
    m = re.search(rf"^\s*{re.escape(k)}\s*:\s*(\S.*?)(?:\s+#.*)?$", txt, re.M)
    return m.group(1).strip() if m else "-"


print(f"我的配置: {mine.name}")
print(f"上游既有: {[p.name for p in others]}\n")
print(f"{'参数':<28s}{'0.8B(我建)':>14s}   上游多数值")
print("-" * 70)
for k in KEYS:
    mv = get(mine, k)
    vals = [get(p, k) for p in others]
    vals = [v for v in vals if v != "-"]
    from collections import Counter
    top = Counter(vals).most_common(1)[0][0] if vals else "-"
    same = "（与上游相同）" if mv == top else "  ← 不同"
    print(f"{k:<28s}{mv:>14s}   {top}{same}")
