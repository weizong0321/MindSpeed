#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 overlay 里缺的分片用软链接补齐（之前 fetch_vlm.py 被中途杀掉，
   下载完成后没走到"链接其余分片"那一步）。"""

import json
import os
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

SRC = Path("/workspace/shared_assets/models/Qwen/Qwen3-VL-30B-A3B-Instruct")
DST = Path("/workspace/vlm_gate/Qwen3-VL-30B-A3B-Instruct")

idx = json.loads((SRC / "model.safetensors.index.json").read_text())
need = sorted(set(idx["weight_map"].values()))

made = bad = 0
for f in need:
    t = DST / f
    try:
        if t.exists() and t.stat().st_size > 0:
            print(f"  已有 {f} ({t.stat().st_size})")
            continue
    except OSError:
        pass
    s = SRC / f
    try:
        ok = s.exists() and s.stat().st_size > 0
    except OSError as e:
        ok = False
        print(f"  !! 源文件坏 {f}: {e}")
    if not ok:
        print(f"  !! 源缺失 {f}")
        bad += 1
        continue
    if t.is_symlink() or t.exists():
        t.unlink()
    os.symlink(s, t)
    print(f"  + 链接 {f} -> {s.stat().st_size}")
    made += 1

print(f"\n新建 {made} 个链接，源缺失 {bad} 个")
# 最终校验
tot = 0
missing = []
for f in need:
    t = DST / f
    try:
        tot += t.stat().st_size
    except OSError:
        missing.append(f)
print(f"13 个分片合计 {tot / 1e9:.2f} GB，缺失 {missing}")
