#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立校验子代理产出的 sample5/answers.json，并剔除它标为可疑的条目。

校验项
  1. 覆盖全部 57 个子样式，键与 selected_styles.json 完全一致
  2. 每个都有 visible(3-5) / caveat / qa 且 qa 的两个问法都有答案
  3. 数字质检：答案里的数字必须能在 visible 或问句里找到（防编造参数）
  4. 长度检查：每段 60–260 字

输出
  selected_styles_final.json   剔除可疑条目后的最终清单
"""

import json
import re
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# 子代理标出的可疑条目（它已写进 answers.json 的 _readme）
SUSPECT = [
    "bag_backpack/bag_backpack_007",
    "bag_handbag/bag_handbag_007",
    "purse_wallet/purse_wallet_008",
    "shoes_leather/shoes_leather_006",
    "shoes_leather/shoes_leather_007",
    "shoes_leather/shoes_leather_008",
    "watch/watch_004",
    "luggage/luggage_007",
    "cosmetics/cosmetics_006",
    "health_supplement/health_supplement_007",
]

NUM = re.compile(r"\d+(?:\.\d+)?")


def main() -> int:
    root = Path(".")
    sel = json.loads((root / "selected_styles.json").read_text(encoding="utf-8"))
    styles = {(s["category"], s["sku"]) for s in sel["styles"]}
    ans = json.loads((root / "sample5/answers.json").read_text(encoding="utf-8"))
    qs = json.loads((root / "sample5/questions.json").read_text(encoding="utf-8"))

    keys = {k for k in ans if not k.startswith("_")}
    print("=" * 72)
    print("1. 覆盖性")
    print("-" * 72)
    print(f"  答案条目 {len(keys)} / 期望 {len(styles)}")
    missing = {f"{c}/{s}" for c, s in styles} - keys
    extra = keys - {f"{c}/{s}" for c, s in styles}
    print(f"  缺失 {len(missing)}: {sorted(missing)[:5]}")
    print(f"  多余 {len(extra)}: {sorted(extra)[:5]}")

    # 问法映射
    qof = {}
    for cat, spec in qs.items():
        if cat.startswith("_"):
            continue
        for q in spec["questions"]:
            qof[(cat, q["id"])] = q["user"]

    print("\n2. 字段完整性 + 长度 + 数字质检")
    print("-" * 72)
    problems, lens = [], []
    for k in sorted(keys):
        cat = k.split("/")[0]
        m = ans[k]
        vis = m.get("visible") or []
        if not (3 <= len(vis) <= 5):
            problems.append(f"{k}: visible 有 {len(vis)} 条（期望 3-5）")
        if not m.get("caveat"):
            problems.append(f"{k}: 缺 caveat")
        qa = m.get("qa") or {}
        for qid in [q["id"] for q in qs[cat]["questions"]]:
            a = qa.get(qid)
            if not a:
                problems.append(f"{k}: 缺问法 {qid} 的答案")
                continue
            lens.append(len(a))
            if not (60 <= len(a) <= 260):
                problems.append(f"{k}/{qid}: 长度 {len(a)}（期望 60-260）")
            unseen = sorted(set(NUM.findall(a))
                            - set(NUM.findall(" ".join(vis)))
                            - set(NUM.findall(qof.get((cat, qid), ""))),
                            key=len, reverse=True)
            if unseen:
                problems.append(f"{k}/{qid}: 数字 {unseen} 无法在 visible/问句中找到证据")

    print(f"  答案段数 {len(lens)}，长度 中位 {sorted(lens)[len(lens) // 2]}"
          f"，min {min(lens)}，max {max(lens)}")
    if problems:
        print(f"  [问题] {len(problems)} 条：")
        for p in problems[:12]:
            print("    " + p)
    else:
        print("  无问题")

    print("\n3. 剔除可疑条目")
    print("-" * 72)
    keep = [s for s in sel["styles"] if f"{s['category']}/{s['sku']}" not in SUSPECT]
    dropped = [s for s in sel["styles"] if f"{s['category']}/{s['sku']}" in SUSPECT]
    print(f"  保留 {len(keep)} 个 / 剔除 {len(dropped)} 个")
    for s in dropped:
        print(f"    - {s['category']}/{s['sku']} ({s['n']} 张)")
    out = root / "selected_styles_final.json"
    out.write_text(json.dumps({"styles": keep, "dropped_suspect": SUSPECT},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    n_img = sum(s["n"] for s in keep)
    print(f"\n  最终：{len(keep)} 个子样式 / {n_img} 张图（上限 {sum(s['n'] for s in keep)}）")
    print(f"  → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
