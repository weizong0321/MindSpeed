#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抽取各版本 0.8B 配置里的模型侧参数，用于写性能/改动报告。"""

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
KEYS = ["model_id", "attn_implementation", "use_triton_gdn", "freeze",
        "init_model_with_meta_device", "recompute:", "param_dtype",
        "reduce_dtype", "micro_batch_size", "train_iters", "lr",
        "optimizer", "adam_fused", "cutoff_len", "image_max_pixels",
        "image_min_pixels", "template", "enable_thinking",
        "save_interval", "use_deter_comp", "overwrite_cache"]

files = sorted(D.glob("qwen3_5_0.8B*config.yaml"))
print(f"共 {len(files)} 个配置\n")
for f in files:
    txt = f.read_text(encoding="utf-8", errors="replace")
    print(f"=== {f.name} ===")
    for k in KEYS:
        m = re.search(rf"^\s*{re.escape(k)}\s*(.*)$", txt, re.M)
        if m:
            v = m.group(1).strip()
            print(f"    {k:<28s} {v}")
    # 数据集指向
    m = re.search(r"^\s*dataset:\s*&?\w*\s*(\S+)", txt, re.M)
    if m:
        print(f"    {'dataset':<28s} {m.group(1)}")
    print()
