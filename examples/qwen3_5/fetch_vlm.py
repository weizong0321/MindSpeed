#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 hf-mirror.com 补齐缺失权重分片，并用软链接拼出一个完整的模型目录。

shared_assets 是只读挂载，所以做法是：
    /workspace/user_data/vlm/<name>/   ← 小文件(配置/tokenizer)直接复制
                                       ← 大分片对 shared 里的做软链接
                                       ← 缺失的分片从 hf-mirror 下载
"""

import argparse
import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

MIRROR = "https://hf-mirror.com"
SHARED = Path("/workspace/shared_assets/models/Qwen")
# 放本地 overlay（300G，稳），别放 glusterfs —— 那卷一直在报 Errno 107
DEST_ROOT = Path("/workspace/vlm_gate")


def human(n):
    return f"{n / 1e9:.2f}G" if n > 1e9 else f"{n / 1e6:.1f}M"


def get(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    return urllib.request.urlopen(req, timeout=timeout)


def remote_size(url):
    """HEAD 拿 Content-Length（跟随时）。"""
    req = urllib.request.Request(url, method="HEAD",
                                 headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length") or 0)


def build(name, repo):
    src = SHARED / name
    dst = DEST_ROOT / name
    dst.mkdir(parents=True, exist_ok=True)

    idx = json.loads((src / "model.safetensors.index.json").read_text())
    need = sorted(set(idx["weight_map"].values()))
    have = set()
    for p in src.iterdir():
        try:
            if p.is_file() and p.stat().st_size > 0:
                have.add(p.name)
        except OSError:
            pass
    missing = [f for f in need if f not in have]
    print(f"[{name}] 需要 {len(need)} 个分片，shared 里有 {len(need) - len(missing)}，"
          f"缺 {len(missing)}: {missing}", flush=True)

    # 1) 小文件直接链接/复制
    for p in sorted(src.iterdir()):
        t = dst / p.name
        if p.name in need:
            continue                                    # 分片单独处理
        try:
            if t.exists() or t.is_symlink():
                continue
            if not p.is_file():
                continue
        except OSError:
            continue                                    # 坏条目，跳过，后面按缺失下载
        try:
            os.symlink(p, t)
        except OSError:
            shutil.copy2(p, t)
    print(f"[{name}] 小文件已就位", flush=True)

    # 2) 分片：能链就链，缺的就下
    for f in need:
        t = dst / f
        try:
            if t.exists() and t.stat().st_size > 0:
                continue
        except OSError:
            pass
        s = src / f
        try:
            s_ok = s.exists() and s.stat().st_size > 0
        except OSError:
            s_ok = False                     # glusterfs 掉线条目 → 当缺失处理
        if s_ok:
            try:
                os.symlink(s, t)
                continue
            except OSError as e:
                print(f"  symlink {f} 失败 {e}，改为复制", flush=True)
                shutil.copy2(s, t)
                continue
        url = f"{MIRROR}/{repo}/resolve/main/{f}"
        tmp = dst / (f + ".part")
        for attempt in range(1, 6):
            try:
                want = remote_size(url)
                t0 = time.time()
                print(f"  下载 {f} ({human(want)}) 第 {attempt} 次 ...", flush=True)
                got, last = 0, 0.0
                with get(url) as r, open(tmp, "wb") as out:
                    while True:
                        b = r.read(1 << 22)
                        if not b:
                            break
                        out.write(b)
                        got += len(b)
                        if time.time() - last > 20:
                            sp = got / max(1e-9, time.time() - t0)
                            print(f"    {human(got)}/{human(want)}  "
                                  f"{sp / 1e6:.1f} MB/s", flush=True)
                            last = time.time()
                if want and got != want:
                    print(f"  !! 大小不符 want={want} got={got}，重试", flush=True)
                    continue
                tmp.rename(t)
                print(f"  ✓ {f} 完成 {human(got)} 用时 {time.time() - t0:.0f}s",
                      flush=True)
                break
            except Exception as e:                              # noqa: BLE001
                print(f"  下载失败 {type(e).__name__}: {str(e)[:150]}，"
                      f"{5}s 后重试", flush=True)
                time.sleep(5)

    # 3) 校验
    still = [f for f in need if not (dst / f).exists()]
    print(f"[{name}] {'完整 ✓' if not still else '仍缺 ' + str(still)}  → {dst}",
          flush=True)
    return not still


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-VL-30B-A3B-Instruct")
    ap.add_argument("--repo", default="Qwen/Qwen3-VL-30B-A3B-Instruct")
    a = ap.parse_args()
    st = os.statvfs(DEST_ROOT.parent)
    print(f"user_data 可用空间 {human(st.f_bavail * st.f_frsize)}", flush=True)
    ok = build(a.model, a.repo)
    print("FETCH_DONE", "OK" if ok else "INCOMPLETE", flush=True)
