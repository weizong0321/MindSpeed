#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只抽取会变的模型侧参数，做横向对比表。"""

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
rows = []
for f in sorted(D.glob("qwen3_5_0.8B_*config.yaml")):
    txt = f.read_text(encoding="utf-8", errors="replace")

    def g(k, d="-"):
        m = re.search(rf"^\s*{re.escape(k)}\s*:\s*(\S+)", txt, re.M)
        return m.group(1) if m else d

    ds = g("dataset")
    n = re.search(r"train_iters\s*:\s*(\d+)", txt)
    rows.append({
        "cfg": f.name.replace("qwen3_5_0.8B_", "").replace("_config.yaml", ""),
        "iters": int(n.group(1)) if n else 0,
        "mbs": g("micro_batch_size"),
        "recompute": g("recompute"),
        "freeze_visual": "yes" if "model.visual" in txt else "no",
        "meta_dev": g("init_model_with_meta_device"),
        "gdn": g("use_triton_gdn"),
        "attn": g("attn_implementation"),
        "dataset": ds.split("/")[-2] + "/" + ds.split("/")[-1] if "/" in ds else ds,
    })

hdr = (f"{'配置':<22s}{'iters':>7s}{'mbs':>5s}{'recompute':>11s}"
       f"{'freeze视觉':>11s}{'meta初始化':>11s}{'gdn':>7s}{'attn':>8s}  数据")
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(f"{r['cfg']:<22s}{r['iters']:>7d}{r['mbs']:>5s}{r['recompute']:>11s}"
          f"{r['freeze_visual']:>11s}{r['meta_dev']:>11s}{r['gdn']:>7s}"
          f"{r['attn']:>8s}  {r['dataset']}")
