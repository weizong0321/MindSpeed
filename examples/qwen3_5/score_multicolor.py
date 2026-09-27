#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诚实地评估"多色问法"：不能只看覆盖率。

问题
    by_part 问法 top2 覆盖率 0.767，看着比单色的 0.700 好。
    但它平均一次说 2.77 个颜色。多报几个颜色，覆盖率当然会涨 —— 这是指标通胀，
    不是标签变准了。必须同时看 precision：
        商品真实是有颜色集合 P（人工判定），模型说集合 M。
        覆盖率 recall  = |M ∩ P| / |P|          （真色有没有漏）
        准确率 precision = |M ∩ P| / |M|        （说的颜色有几个是真的）
    只有 precision 不塌，多色方案才成立。
"""

import json
import sys
from pathlib import Path
# --- path shim (auto-added for portability) ---
import os as _os
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
MM_DATA = _Path(_os.environ.get("MM_DATA", _REPO_ROOT / "data" / "ecommerce_multimodal"))
MM_WORK = _Path(_os.environ.get("MM_WORK", _REPO_ROOT / "work"))
MM_EX = _REPO_ROOT / "examples" / "qwen3_5"
# --- end shim ---


for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

BASE = Path(str(_REPO_ROOT / "data/ecommerce_multimodal/sample5"))

# 把模型输出的各种说法归一到"色系"单字，才能和真值集合比较
NORM = [
    ("透明", "透明"), ("无色", "透明"),
    ("藏青", "蓝"), ("深蓝", "蓝"), ("蓝", "蓝"), ("青", "绿"), ("蓝绿", "绿"),
    ("墨绿", "绿"), ("深绿", "绿"), ("绿", "绿"),
    ("银白", "银"), ("银灰", "银"), ("银", "银"),
    ("灰白", "灰"), ("深灰", "灰"), ("灰", "灰"),
    ("黑色", "黑"), ("纯黑", "黑"), ("黑", "黑"),
    ("白色", "白"), ("米白", "白"), ("白", "白"),
    ("正红", "红"), ("酒红", "红"), ("红", "红"),
    ("粉", "粉"), ("橙", "橙"), ("黄", "黄"),
    ("金", "金"), ("棕", "棕"), ("咖啡", "棕"), ("褐", "棕"),
    ("紫", "紫"), ("米", "米"), ("卡其", "卡其"), ("驼", "驼"),
]
PALETTE = {"黑", "白", "灰", "银", "金", "棕", "红", "粉", "橙",
           "黄", "绿", "蓝", "紫", "透明"}


def norm(word):
    for pat, canon in NORM:
        if pat in word:
            return canon
    return None


def main():
    truth = {t["id"]: t for t in json.loads(
        (BASE / "gate_truth.json").read_text("utf-8"))["items"]}
    prod = json.loads((BASE / "gate_product_colors.json").read_text("utf-8"))["colors"]
    data = json.loads(Path("/workspace/user_data/gate_multicolor.json")
                      .read_text("utf-8"))

    print("诚实的多色评估（真值 = 商品真实颜色集合，人工判定）")
    print(f"{'问法':<20s} {'主色覆盖':>8s} {'recall':>7s} {'precision':>10s} "
          f"{'F1':>6s} {'平均色数':>8s} {'纯假色图数':>10s}")
    print("-" * 78)
    for name, v in data.items():
        cov = rec = prec = 0.0
        n = len(v["rows"])
        fake_only = 0
        for r in v["rows"]:
            P = set(prod[r["id"]])
            M = {norm(c) for c in r["colors"]}
            M.discard(None)
            main = norm(r["truth"]) or r["truth"]
            cov += 1 if main in M else 0
            inter = M & P
            rec += len(inter) / len(P) if P else 0
            prec += len(inter) / len(M) if M else 0
            if M and not inter:
                fake_only += 1
        cov, rec, prec = cov / n, rec / n, prec / n
        f1 = 2 * rec * prec / (rec + prec) if rec + prec else 0
        v["_cov"] = round(cov, 3)
        v["_recall"] = round(rec, 3)
        v["_precision"] = round(prec, 3)
        v["_f1"] = round(f1, 3)
        v["_fake_only"] = fake_only
        print(f"{name:<20s} {cov:8.3f} {rec:7.3f} {prec:10.3f} {f1:6.3f} "
              f"{v['avg_colors']:8.2f} {fake_only:10d}")

    print("\n注：'纯假色图数' = 模型说的颜色在商品上一个都不存在的图数。")
    print("    单色问法 precision 天然接近 1（只说一个色），多色问法会塌。")

    print("\n=== 逐图：by_part 说了什么 vs 商品真实有什么 ===")
    rows = data["by_part"]["rows"]
    for r in rows:
        P = set(prod[r["id"]])
        M = {norm(c) for c in r["colors"]}
        M.discard(None)
        ok = M & P
        bad = M - P
        mark = "" if not bad else "  <-- 假色: " + ",".join(sorted(bad))
        print(f"  {r['id']} 真值主色{r['truth']:>4s} 真实集合{sorted(P)} "
              f"模型{sorted(M)} 命中{sorted(ok)}{mark}")

    Path("/workspace/user_data/gate_multicolor_scored.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n→ /workspace/user_data/gate_multicolor_scored.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
