#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修 sample7_hf 的 safetensors index：分片文件名对不上。

现象
    merge_dcp_to_hf.py 把权重写成   model-00001-of-00001.safetensors
    但它直接拷了 base 模型的 index，里面写的是
                                    model.safetensors-00001-of-00001.safetensors
    于是 from_pretrained 找不到文件。

做法
    读实际分片里有哪些 key，和 index 的 key 集合核对；一致就把 weight_map
    的值统一改成真实文件名，并修正 metadata.total_size。
"""

import json
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from safetensors import safe_open

D = Path(sys.argv[1] if len(sys.argv) > 1
         else "/workspace/user_data/output/qwen3_5_0.8B_sample7_hf")

idx_p = D / "model.safetensors.index.json"
idx = json.loads(idx_p.read_text())
wmap = idx["weight_map"]

shards = sorted(p.name for p in D.glob("*.safetensors"))
print(f"目录 {D}")
print(f"实际分片: {shards}")
print(f"index 引用的文件名: {sorted(set(wmap.values()))}")

# 收集实际分片里的 key
real = {}
for s in shards:
    with safe_open(str(D / s), framework="pt") as f:
        for k in f.keys():
            real[k] = s

idx_keys = set(wmap)
real_keys = set(real)
print(f"\nindex key 数 {len(idx_keys)}，分片 key 数 {len(real_keys)}")
only_idx = sorted(idx_keys - real_keys)
only_real = sorted(real_keys - idx_keys)
if only_idx:
    print(f"!! 只在 index 里（会缺失）: {len(only_idx)} 个，例 {only_idx[:5]}")
if only_real:
    print(f"!! 只在分片里（未被引用）: {len(only_real)} 个，例 {only_real[:5]}")

if only_idx or only_real:
    print("\nkey 集合不一致，不敢乱改 index。请人工检查。")
    sys.exit(1)

# 一致：把 weight_map 指向真实文件名
changed = 0
for k, v in list(wmap.items()):
    if real[k] != v:
        wmap[k] = real[k]
        changed += 1
print(f"\n修正 {changed} 条 weight_map 记录")

total = 0
for s in shards:
    total += (D / s).stat().st_size
idx.setdefault("metadata", {})["total_size"] = total

idx_p.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"已写回 {idx_p}")

# 复核
chk = json.loads(idx_p.read_text())
files = set(chk["weight_map"].values())
missing = [f for f in files if not (D / f).exists()]
print(f"复核：index 引用 {len(files)} 个文件，缺失 {missing}")
print("INDEX_FIX_OK" if not missing else "INDEX_FIX_FAIL")
