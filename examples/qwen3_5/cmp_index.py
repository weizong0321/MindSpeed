#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比 sample6_hf（能加载）和 sample7_hf（加载失败）的 index / 分片一致性。"""

import json
import sys
from pathlib import Path

from safetensors import safe_open

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

CASES = [
    ("base_hf", "/workspace/user_data/base_hf/Qwen3.5-0.8B-base"),
    ("sample6_hf", "/workspace/user_data/output/qwen3_5_0.8B_sample6_hf"),
    ("sample7_hf", "/workspace/user_data/output/qwen3_5_0.8B_sample7_hf"),
]

for tag, d in CASES:
    D = Path(d)
    if not D.exists():
        print(f"{tag}: 不存在")
        continue
    print(f"\n===== {tag} =====")
    print(f"  文件: {sorted(p.name for p in D.iterdir() if p.is_file())}")
    idx_p = D / "model.safetensors.index.json"
    if not idx_p.exists():
        print("  没有 index")
        continue
    wmap = json.loads(idx_p.read_text())["weight_map"]
    files = sorted(set(wmap.values()))
    print(f"  index: {len(wmap)} keys, 引用 {files}")
    for f in files:
        fp = D / f
        if not fp.exists():
            print(f"    !! 缺失 {f}")
            continue
        with safe_open(str(fp), framework="pt") as fh:
            ks = set(fh.keys())
        mtp = [k for k in ks if k.startswith("mtp.")]
        print(f"    {f}: {len(ks)} keys（其中 mtp.* {len(mtp)} 个）")
    mtp_idx = [k for k in wmap if k.startswith("mtp.")]
    print(f"  index 里的 mtp.* key: {len(mtp_idx)} 个")
    cfg = json.loads((D / "config.json").read_text())
    for k in ("num_nextn_predict_layers", "num_mtp_layers", "mtp_layers"):
        if k in cfg:
            print(f"  config.{k} = {cfg[k]}")
