#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成"可以直接写进精度报告"的数字：点估计 + 置信区间 + 样本量。

为什么必须带区间
    门槛实验只有 30 张，属性评测每项只有 4-12 个样本。
    只报点估计（比如"平衡准确率 1.000"）会严重误导 ——
    n=10 的 1.000，95% 置信下界大约只有 0.7。
    所以：
      · 命中率类 → Wilson 区间（小样本比正态近似稳）
      · 平衡准确率/宏平均 → 对图片做 bootstrap（属性之间相关，不能套二项公式）
"""

import json
import math
import random
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

U = Path("/workspace/user_data")
S5 = Path("/workspace/MindSpeed/data/ecommerce_multimodal/sample5")

COLOR_ALTS = {
    "银色": ["银", "银色", "银白", "银灰"], "棕色": ["棕", "棕色", "褐色", "咖啡色"],
    "灰色": ["灰", "灰色", "深灰", "银灰", "灰白"], "黑色": ["黑", "黑色", "纯黑"],
    "白色": ["白", "白色"], "蓝色": ["蓝", "蓝色", "深蓝", "藏青"],
    "红色": ["红", "红色", "正红", "酒红"],
    "绿色": ["绿", "绿色", "墨绿", "深绿", "青色", "蓝绿"],
    "黄色": ["黄", "黄色"], "粉色": ["粉", "粉色", "蓝", "蓝色"],
    "透明": ["透明", "无色", "白色", "白"],
}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


print("=" * 92)
print("A. 非循环颜色准确率（30 张人工标注真值）")
print("=" * 92)
truth = {t["id"]: t for t in json.loads(
    (S5 / "gate_truth.json").read_text("utf-8"))["items"]}
rows = {}
for tag in ("base", "s6", "t4b_cap", "t30b_cap"):
    p = U / f"gate_{tag}.json"
    if p.exists():
        rows[tag] = {r["id"]: r for r in json.loads(p.read_text("utf-8"))["rows"]}

print(f"\n{'模型':<10s}{'n':>4s}{'共答对':>8s}{'命中率':>9s}{'95% 置信区间':>20s}")
print("-" * 92)
for tag, rs in rows.items():
    k = n = 0
    for i, t in truth.items():
        r = rs.get(i)
        if not r:
            continue
        n += 1
        alts = COLOR_ALTS.get(t["truth_color"], [t["truth_color"]])
        if any(x in r["raw"] for x in alts):
            k += 1
    lo, hi = wilson(k, n)
    print(f"{tag:<10s}{n:>4d}{k:>8d}{k/n:>9.3f}      [{lo:.2f}, {hi:.2f}]")

print("\n注意：这 30 张是我按品类分层挑的，不是从 14 万张里随机抽的，")
print("      所以严格说只能代表'这 30 张'；要外推必须随机抽样。")

# ---- B. 属性 ----
print("\n" + "=" * 92)
print("B. 闭词表属性（sample8b，78 张评测图）")
print("=" * 92)
p = U / "eval8_s8b.json"
if p.exists():
    d = json.loads(p.read_text("utf-8"))
    s, rs, pa = d["summary"], d["rows"], d["per_attr"]

    print(f"\n{'属性':<30s}{'n':>4s}{'平衡准确':>9s}{'基线':>7s}"
          f"{'bootstrap 95% 区间':>24s}")
    print("-" * 92)

    def bal_of(rows_sub, key):
        # per_attr 的 key 是 "品类.属性"，但 gold_own 只有属性名，所以要拆开
        cat, attr = key.split(".", 1)
        cls, cc = {}, {}
        for r in rows_sub:
            if r.get("cat") != cat:
                continue
            g = r["gold_own"].get(attr)
            if g is None:
                continue
            cls[g] = cls.get(g, 0) + 1
            if r["pred_own"].get(attr) == g:
                cc[g] = cc.get(g, 0) + 1
        if not cls:
            return None
        return sum(cc.get(c, 0) / cls[c] for c in cls) / len(cls)

    random.seed(0)
    keys = sorted(pa)
    for k in keys:
        n = pa[k]["n"]
        point = bal_of(rs, k) or 0
        base = max(pa[k]["cls"].values()) / n if n else 0
        boots = []
        for _ in range(2000):
            sub = [rs[random.randrange(len(rs))] for _ in range(len(rs))]
            v = bal_of(sub, k)
            if v is not None:
                boots.append(v)
        boots.sort()
        if boots:
            lo = boots[int(0.025 * len(boots))]
            hi = boots[int(0.975 * len(boots)) - 1]
            ci = f"[{lo:.2f}, {hi:.2f}]"
        else:
            ci = "样本不足"
        print(f"{k:<30s}{n:>4d}{point:>9.3f}{base:>7.2f}      {ci:>20s}")

    # 宏平均
    boots = []
    for _ in range(2000):
        sub = [rs[random.randrange(len(rs))] for _ in range(len(rs))]
        vals = [bal_of(sub, k) for k in keys]
        vals = [v for v in vals if v is not None]
        if vals:
            boots.append(sum(vals) / len(vals))
    boots.sort()
    print("-" * 92)
    print(f"{'宏平均平衡准确率':<30s}{'':>4s}{s['macro_balanced']:>9.3f}"
          f"{s['macro_baseline']:>7.3f}      "
          f"[{boots[int(0.025*len(boots))]:.2f}, "
          f"{boots[int(0.975*len(boots))-1]:.2f}]")
    print(f"{'宏平均原始准确率':<30s}{'':>4s}{s['macro_acc']:>9.3f}")

    # 换图测试
    kk, nn = 0, 0
    for r in rs:
        if r["gold_own"] != r["gold_other"]:
            nn += 1
            m_own = sum(1 for k in r["gold_own"]
                        if r["pred_own"].get(k) == r["gold_own"][k])
            m_oth = sum(1 for k in r["gold_own"]
                        if r["pred_own"].get(k) == r["gold_other"].get(k))
            if m_own > m_oth:
                kk += 1
    lo, hi = wilson(kk, nn)
    print(f"\n换图测试（喂自己的图更像自己）: {kk}/{nn} = {kk/nn:.3f}  "
          f"95% CI [{lo:.2f}, {hi:.2f}]  （0.5 = 瞎猜）")

    # 逐张命中率（整个属性组全对）
    allright = sum(1 for r in rs if r["gold_own"]
                   and all(r["pred_own"].get(k) == v
                           for k, v in r["gold_own"].items()))
    lo, hi = wilson(allright, len(rs))
    print(f"整组属性全对: {allright}/{len(rs)} = {allright/len(rs):.3f}  "
          f"95% CI [{lo:.2f}, {hi:.2f}]")

    # 对照基线：完全不看图，每个属性都答"训练集里最常见的那个值"
    # 注意不能用评测集自己的众数（那是作弊），要用训练集的众数
    tr_path = Path("/workspace/MindSpeed/data/ecommerce_multimodal/"
                   "sample8b/attrs_truth.json")
    if tr_path.exists():
        allt = json.loads(tr_path.read_text("utf-8"))
        eval_imgs = {r["image"] for r in rs}
        cnt = {}
        for img, t in allt.items():
            if img in eval_imgs:
                continue
            for k, v in (t.get("attrs") or {}).items():
                cnt.setdefault(f"{t['category']}.{k}", {})
                cnt[f"{t['category']}.{k}"][v] = \
                    cnt[f"{t['category']}.{k}"].get(v, 0) + 1
        maj = {k: max(v, key=v.get) for k, v in cnt.items()}
        hit = 0
        for r in rs:
            ok = True
            for k, v in (r["gold_own"] or {}).items():
                if maj.get(f"{r['cat']}.{k}") != v:
                    ok = False
                    break
            hit += 1 if ok else 0
        if rs:
            lo2, hi2 = wilson(hit, len(rs))
            print(f"对照·永远答训练集众数（完全不看图）: {hit}/{len(rs)} = "
                  f"{hit/len(rs):.3f}  95% CI [{lo2:.2f}, {hi2:.2f}]")
            print(f"  → 整组全对的净增益: "
                  f"{(allright - hit)/len(rs):+.3f}")
