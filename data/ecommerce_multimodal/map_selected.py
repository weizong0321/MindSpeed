#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把用户判读的 slot 列表映射到子样式，按品类汇总。"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

PURE_SLOTS = """1.5 1.10 2.1 2.2 2.3 2.9 3.1 4.3 4.4 4.6 4.7 4.9 4.10 5.5 5.6 6.4 6.6
7.6 8.3 8.7 9.8 9.9 10.1 10.7 10.8 11.2 12.4 12.5 12.6 12.8 13.1 13.2 13.3 13.4
13.9 14.3 14.4 14.5 15.4 16.5 17.2 17.6 17.9 17.10 18.4 18.5 18.10 19.1 19.2 19.5
19.9 20.1 20.3 20.4 20.6 20.9 20.10""".split()

wanted = set()
for s in PURE_SLOTS:
    sheet, slot = s.split(".")
    wanted.add((int(sheet), int(slot)))

rows = list(csv.DictReader(open("cand_sheets/candidates.csv", encoding="utf-8-sig")))
sel = [r for r in rows if (int(r["sheet"]), int(r["slot"])) in wanted]
missing = wanted - {(int(r["sheet"]), int(r["slot"])) for r in rows}
print(f"判读 {len(wanted)} 个 → 匹配到 {len(sel)} 个")
if missing:
    print(f"[警告] 未匹配：{sorted(missing)}")

cat_names = json.loads(Path("cluster_labels.json").read_text(
    encoding="utf-8")).get("categories", {})
by_cat = defaultdict(list)
for r in sel:
    by_cat[r["category"]].append((r["sku"], int(r["n"])))

print(f"\n涉及 {len(by_cat)} 个品类，共 {len(sel)} 个子样式，"
      f"{sum(int(r['n']) for r in sel)} 张图：")
print("-" * 66)
for c in sorted(by_cat, key=lambda k: -sum(n for _, n in by_cat[k])):
    items = sorted(by_cat[c], key=lambda x: -x[1])
    tot = sum(n for _, n in items)
    print(f"{cat_names.get(c, c) + ' (' + c + ')':<28}{len(items)}个/{tot}张  "
          f"{', '.join(f'{k}({n})' for k, n in items[:5])}"
          f"{' ...' if len(items) > 5 else ''}")
print("-" * 66)

# 输出清单，供后续构建数据集
out = Path("selected_styles.json")
out.write_text(json.dumps(
    {"slots": PURE_SLOTS,
     "styles": [{"sku": r["sku"], "category": r["category"], "n": int(r["n"])}
                for r in sorted(sel, key=lambda r: r["category"])]},
    ensure_ascii=False, indent=2), encoding="utf-8")
print(f"→ {out}")

# 每类挑选张数最多的前 2 个，作为「写答案时需要看图」的清单
print("\n需要看图写答案的子样式（全部 %d 个）" % len(sel))
need = [(r["category"], r["sku"], int(r["n"])) for r in
        sorted(sel, key=lambda r: (r["category"], -int(r["n"])))]
print(f"  张数分布：最小 {min(n for _, _, n in need)}，"
      f"中位 {sorted(n for _, _, n in need)[len(need) // 2]}，"
      f"最大 {max(n for _, _, n in need)}")
