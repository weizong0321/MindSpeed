#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 Qwen3-VL-30B-A3B-Instruct 的融合专家权重转置，修掉和本地 transformers 的布局冲突。

问题
    本地 transformers 的 MoE 专家用 nn.functional.linear(x, W)，
    所以 W 必须是 [out, in]：
        gate_up_proj  [num_experts, 2*intermediate, hidden]
        down_proj     [num_experts, hidden, intermediate]
    而官方 checkpoint 存的是反过来：
        gate_up_proj  [num_experts, hidden, 2*intermediate]
        down_proj     [num_experts, intermediate, hidden]
    只是最后两维转置，其余张量全部匹配。

做法
    逐分片读 → 只对这两类 key 做 transpose(-1,-2) → 另存到新目录。
    key 名和 index 不变（index 只记 key→文件，不记形状），所以 index.json 直接复制。

注意
    不要用 ignore_mismatched_sizes=True —— 那会把 48 层的专家层随机初始化成垃圾。
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from safetensors.torch import load_file, save_file

SRC = Path(os.environ.get("SRC", "/workspace/vlm_gate/Qwen3-VL-30B-A3B-Instruct"))
DST = Path(os.environ.get("DST", "/workspace/vlm_gate_fixed/Qwen3-VL-30B-A3B-Instruct"))

FIX_SUFFIXES = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")


def human(n):
    return f"{n / 1e9:.2f}G"


def main() -> int:
    DST.mkdir(parents=True, exist_ok=True)
    idx = json.loads((SRC / "model.safetensors.index.json").read_text())
    wmap = idx["weight_map"]
    shards = sorted(set(wmap.values()))
    print(f"源 {SRC}")
    print(f"目标 {DST}")
    print(f"{len(shards)} 个分片，{len(wmap)} 个权重\n", flush=True)

    st = os.statvfs(DST.parent)
    print(f"可用空间 {human(st.f_bavail * st.f_frsize)}\n", flush=True)

    # 1) 小文件直接复制（config/tokenizer/preprocessor 等）
    for p in sorted(SRC.iterdir()):
        if not p.is_file() or p.name in shards:
            continue
        try:
            shutil.copy2(p, DST / p.name)
        except OSError as e:
            print(f"  复制 {p.name} 失败: {e}")
    print("小文件已复制\n", flush=True)

    # 2) 逐分片转置
    t_all = time.time()
    total_fixed = 0
    for i, sh in enumerate(shards, 1):
        src = SRC / sh
        dst = DST / sh
        t0 = time.time()
        if dst.exists() and dst.stat().st_size > 0:
            print(f"[{i}/{len(shards)}] {sh} 已存在，跳过", flush=True)
            continue
        sd = load_file(str(src))
        fixed = []
        for k in list(sd):
            if k.endswith(FIX_SUFFIXES):
                old = tuple(sd[k].shape)
                sd[k] = sd[k].transpose(-1, -2).contiguous()
                fixed.append(f"{k.split('.')[-1]} {old} -> {tuple(sd[k].shape)}")
        size = sum(t.numel() * t.element_size() for t in sd.values())
        save_file(sd, str(dst), metadata={"format": "pt"})
        del sd
        total_fixed += len(fixed)
        print(f"[{i}/{len(shards)}] {sh}  {human(size)}  "
              f"转置 {len(fixed)} 个张量  {time.time() - t0:.0f}s", flush=True)
        if fixed:
            print(f"      例: {fixed[0]}", flush=True)

    # 3) index 直接复制（形状不在 index 里，key→文件映射没变）
    shutil.copy2(SRC / "model.safetensors.index.json",
                 DST / "model.safetensors.index.json")

    print(f"\n完成。共转置 {total_fixed} 个张量，用时 "
          f"{(time.time() - t_all) / 60:.1f} 分钟", flush=True)

    # 4) 校验
    missing = [f for f in set(wmap.values())
               if not (DST / f).exists() or (DST / f).stat().st_size == 0]
    tot = sum((DST / f).stat().st_size for f in set(wmap.values())
              if (DST / f).exists())
    print(f"分片齐全: {not missing}  缺失 {missing}", flush=True)
    print(f"合计 {human(tot)}", flush=True)
    print("CONVERT_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
