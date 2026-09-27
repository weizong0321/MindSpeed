#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从"可靠"的属性里再筛掉"没信息量"的。

关键区分
    一致率 1.000 可能有两种完全不同的原因：
      (a) 该属性确实由图唯一决定，两个老师都看对了  → 有价值
      (b) 这 6 张图本来取值就全一样（多数类基线 1.00）→ 一致率是假的
    例：shoes_sneaker.toe 6 张全是"圆头"、watch.dial_shape 6 张全是"圆形"。
    这类属性两个老师当然一致，但它不携带任何逐图信息。

筛选标准（三条都要满足）
    可靠：一致率 >= 0.85
    覆盖：双答率 >= 0.6（弃权太多就不算数）
    有信息量：多数类基线 <= 0.85 且 取值数 >= 2
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(_Path(__file__).resolve().parent))
from attr_spec import ATTRS  # noqa: E402
# --- path shim (auto-added for portability) ---
import os as _os
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
MM_DATA = _Path(_os.environ.get("MM_DATA", _REPO_ROOT / "data" / "ecommerce_multimodal"))
MM_WORK = _Path(_os.environ.get("MM_WORK", _REPO_ROOT / "work"))
MM_EX = _REPO_ROOT / "examples" / "qwen3_5"
# --- end shim ---


U = MM_WORK
keep = {k["attr"]: k for k in
        json.loads((U / "attr_keep.json").read_text("utf-8"))}
rows = json.loads((U / "attr_t4b_big.json").read_text("utf-8"))

vals = defaultdict(Counter)
for r in rows:
    cat = r["category"]
    j = r.get("attrs") or {}
    for k, v in j.items():
        full = f"{cat}.{k}"
        if full in keep and v is not None:
            vals[full][str(v).strip()] += 1

selected = {}
rejected = []
for full, info in keep.items():
    cat, attr = full.split(".", 1)
    c = vals.get(full, Counter())
    tot = sum(c.values())
    top, tn = c.most_common(1)[0] if c else ("", 0)
    base = tn / tot if tot else 1.0
    nval = len(c)
    if base <= 0.85 and nval >= 2:
        selected.setdefault(cat, []).append(attr)
    else:
        rejected.append((full, info["agreement"], info["coverage"], base, nval))

print(f"保留 {sum(len(v) for v in selected.values())} 个"
      f"（有信息量且可靠），剔除 {len(rejected)} 个\n")
print(f"{'被剔除的属性':<34s}{'一致率':>8s}{'覆盖':>7s}{'多数类':>8s}{'取值数':>7s}")
print("-" * 66)
for full, ag, cov, base, nval in sorted(rejected, key=lambda x: -x[3]):
    why = "取值单一" if nval == 1 else ("基线过高" if base > 0.85 else "覆盖不足")
    print(f"{full:<34s}{ag:>8.3f}{cov:>7.2f}{base:>8.2f}{nval:>7d}  {why}")

print(f"\n=== 保留的属性（按品类）===")
for cat in sorted(selected):
    spec = {k: (d, o) for k, d, o in ATTRS.get(cat, [])}
    parts = []
    for a in selected[cat]:
        c = vals[f"{cat}.{a}"]
        base = max(c.values()) / sum(c.values())
        parts.append(f"{a}(基线{base:.2f})")
    print(f"  {cat:22s} {', '.join(parts)}")

# 生成只含保留属性的规格，供标注器使用
spec_out = {}
for cat, attrs in selected.items():
    spec_out[cat] = [{"key": k, "desc": d, "options": o}
                     for k, d, o in ATTRS.get(cat, []) if k in attrs]
(U / "attr_final_spec.json").write_text(
    json.dumps(spec_out, ensure_ascii=False, indent=2), encoding="utf-8")
(U / "attr_selected.json").write_text(
    json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n→ {U / 'attr_final_spec.json'}")
print(f"共 {sum(len(v) for v in selected.values())} 个属性 / {len(selected)} 个品类")
