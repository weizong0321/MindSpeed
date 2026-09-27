#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""比较两个老师在闭词表属性上的一致性。

输出每个属性的：
    覆盖率  两个老师都没弃权的样本比例
    弃权率  写了"不确定"的比例
    一致率  两边都给出答案时，答案相同的比例   ← 关键指标
    合法率  答案是否落在允许的选项里（模型有没有乱编）
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(_Path(__file__).resolve().parent))
from attr_spec import ATTRS, UNSURE  # noqa: E402
# --- path shim (auto-added for portability) ---
import os as _os
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
MM_DATA = _Path(_os.environ.get("MM_DATA", _REPO_ROOT / "data" / "ecommerce_multimodal"))
MM_WORK = _Path(_os.environ.get("MM_WORK", _REPO_ROOT / "work"))
MM_EX = _REPO_ROOT / "examples" / "qwen3_5"
# --- end shim ---


U = MM_WORK
def load(tag):
    p = U / f"attr_{tag}.json"
    if not p.exists():
        return None
    # 关键：按 (品类, id) 建索引，不能用 id 单独索引。
    # sample_attr_images.py 里 id 用的是 cat[:6]，导致
    # apparel_pants / apparel_shorts / apparel_top 都变成 "appare_XX"，
    # 四个 shoes_* 都变成 "shoes_XX"，只按 id 做 key 会把它们互相覆盖掉。
    out = {}
    for r in json.loads(p.read_text("utf-8")):
        out[(r.get("category"), r.get("id"))] = r
    return out


def main():
    tags = sys.argv[1:3] if len(sys.argv) >= 3 else ["t4b", "t30b"]
    A = load(tags[0])
    B = load(tags[1])
    if not A or not B:
        print(f"缺少结果文件 attr_{tags[0]}.json / attr_{tags[1]}.json")
        return 1
    print(f"比较 {tags[0]} vs {tags[1]}\n")

    per = defaultdict(lambda: {"n": 0, "both": 0, "agree": 0,
                               "unsure_a": 0, "unsure_b": 0, "illegal": 0})
    for key in sorted(A, key=lambda k: (str(k[0]), str(k[1]))):
        ra, rb = A[key], B.get(key)
        if not rb or not ra.get("keys") or not rb.get("keys"):
            continue
        tid = ra.get("id")
        cat = ra["category"]
        spec = dict((k, opts) for k, _, opts in ATTRS.get(cat, []))
        ja, jb = ra.get("attrs") or {}, rb.get("attrs") or {}
        for k in ra["keys"]:
            if k not in spec:
                continue
            opts = spec[k] + [UNSURE]
            va, vb = ja.get(k), jb.get(k)
            d = per[f"{cat}.{k}"]
            d["n"] += 1
            # 合法性
            bad = False
            for v in (va, vb):
                if v is None:
                    continue
                if not any(str(v).strip().startswith(o) or o.startswith(str(v).strip())
                           for o in opts):
                    bad = True
            if bad:
                d["illegal"] += 1
            ua = (va is None) or (UNSURE in str(va))
            ub = (vb is None) or (UNSURE in str(vb))
            d["unsure_a"] += 1 if ua else 0
            d["unsure_b"] += 1 if ub else 0
            if not ua and not ub:
                d["both"] += 1
                if str(va).strip() == str(vb).strip():
                    d["agree"] += 1

    print(f"\n{'属性':<34s}{'样本':>5s}{'双答':>6s}{'覆盖':>7s}{'一致':>6s}"
          f"{'一致率':>8s}  判定")
    print("-" * 88)
    tot_both = tot_agree = 0
    kept = []
    for key in sorted(per, key=lambda k: -per[k]["agree"] / max(per[k]["both"], 1)):
        d = per[key]
        rate = d["agree"] / d["both"] if d["both"] else 0
        cov = d["both"] / d["n"] if d["n"] else 0
        tot_both += d["both"]
        tot_agree += d["agree"]
        # 判定要求同时满足"覆盖够"和"一致率高"：
        # 只看一致率会被弃权骗过去（n=6 里只双答 1 次也会显示 1.000）
        if cov >= 0.6 and rate >= 0.85:
            verdict = "保留"
            kept.append((key, rate, cov))
        elif cov >= 0.6 and rate >= 0.7:
            verdict = "边缘"
        elif cov < 0.6:
            verdict = "弃权太多"
        else:
            verdict = "剔除"
        print(f"{key:<34s}{d['n']:>5d}{d['both']:>6d}{cov:>7.2f}{d['agree']:>6d}"
              f"{rate:>8.3f}  {verdict}")
    print("-" * 88)
    print(f"{'合计':<34s}{'':>5s}{tot_both:>6d}{'':>7s}{tot_agree:>6d}"
          f"{tot_agree / max(tot_both, 1):>8.3f}")

    print(f"\n=== 可用属性（覆盖>=0.6 且 一致率>=0.85）：{len(kept)} 个 ===")
    for k, r, c in kept:
        print(f"  {k:<34s} 一致率 {r:.3f}  覆盖 {c:.2f}")
    (U / "attr_keep.json").write_text(json.dumps(
        [{"attr": k, "agreement": round(r, 3), "coverage": round(c, 3)}
         for k, r, c in kept], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {U / 'attr_keep.json'}")

    print("\n判读：")
    print("  一致率 >= 0.85 且弃权少 → 该属性可靠，可以拿来做训练目标")
    print("  一致率 0.6~0.85        → 边缘，要么改选项定义，要么弃用")
    print("  一致率 < 0.6           → 该属性依然不可靠，剔除")
    print(f"\n参照：旧目标 parts 在同样两个老师之间只有 0.133 完全一致。")

    # 按品类汇总
    print("\n=== 按品类 ===")
    bycat = defaultdict(lambda: [0, 0])
    for key, d in per.items():
        c = key.split(".")[0]
        bycat[c][0] += d["both"]
        bycat[c][1] += d["agree"]
    for c in sorted(bycat):
        b, g = bycat[c]
        print(f"  {c:22s} {g}/{b} = {g / max(b, 1):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
