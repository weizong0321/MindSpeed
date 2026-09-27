#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型到底在不在看图？用检索测试判断，而不是逐字匹配。

动机
    8.7 节里 s7 的"从句逐字复现"只有 0.138。但这可能不是"没看图"，而是
    标签本身在同义词之间摇摆：同一张图，VLM 这次写"图案"，下次写"纹理"，
    模型输出"纹理"被判错，其实它说对了。

做法（不依赖字符串完全相等）
    对每张评测图 i，它所属子样式里还有别的图 j。
    模型喂 i 时输出的属性集合记作 M。
    比较 sim(M, truth(i)) 和 sim(M, truth(j))：
      - 如果模型真在看图，M 应该更像 i 的属性，而不是 j 的。
      - 如果它在套品类模板，M 对 i 和 j 的相似度会差不多。
    判据 = 在 truth(i) != truth(j) 的样本上，"更像自己"的比例。
    套模板的模型会接近 0.5（瞎猜），真看图的模型应该明显更高。
"""

import json
import re
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

U = MM_WORK
B = Path(str(MM_DATA))


def clause_of(answer):
    m = re.search(r"图中为(.*?)。", answer)
    return m.group(1).strip() if m else None


def parse_clause(c):
    """把从句拆成 (颜色词集合, 品类, 部件集合, 场景)"""
    if not c:
        return set(), "", set(), ""
    segs = [s for s in c.split("，") if s]
    scene = segs[-1] if segs else ""
    body = segs[:-1]
    head = body[0] if body else ""
    parts = set()
    for seg in body[1:]:
        for p in re.split(r"[、,]", seg):
            if p.strip():
                parts.add(p.strip())
    return head, parts, scene


def sim(a_parts, b_parts):
    """部件集合的 Jaccard"""
    if not a_parts and not b_parts:
        return 1.0
    if not a_parts or not b_parts:
        return 0.0
    return len(a_parts & b_parts) / len(a_parts | b_parts)


def score(model_parts, truth_parts):
    """部分重合度也认：用 F1 而不是严格集合相等"""
    if not truth_parts:
        return 0.0
    hit = len(model_parts & truth_parts)
    if hit == 0:
        return 0.0
    prec = hit / len(model_parts) if model_parts else 0.0
    rec = hit / len(truth_parts)
    return 2 * prec * rec / (prec + rec)


def run(tag, rows, truth, label):
    disc = tie = 0
    s_own = s_oth = 0.0
    n = 0
    for r in rows:
        c = clause_of(r["a_own"])
        _, mparts, _ = parse_clause(c)
        ti = (truth.get(r["image"]) or {}).get("parts") or []
        tj = (truth.get(r["other"]) or {}).get("parts") or []
        if set(ti) == set(tj):
            tie += 1
            continue
        si, sj = score(mparts, set(ti)), score(mparts, set(tj))
        s_own += si
        s_oth += sj
        n += 1
        if si > sj:
            disc += 1
    if n == 0:
        print(f"{label}: 没有可区分的样本")
        return
    print(f"\n===== {label}（{tag}）=====")
    print(f"  可区分样本 {n} 张（另 {tie} 张两张图属性相同，无法区分，已排除）")
    print(f"  更像自己的图: {disc}/{n} = {disc / n:.3f}   ← 0.5 = 瞎猜，越高质量越好")
    print(f"  平均相似度：对自己 {s_own / n:.3f} vs 对换的图 {s_oth / n:.3f}")


def maybe_json(p):
    p = Path(p)
    if p.exists():
        return json.loads(p.read_text("utf-8"))
    return {}


def main():
    t7 = maybe_json(B / "sample7/attrs_truth.json")
    t6 = maybe_json(B / "sample6/attrs_truth.json")
    t5 = maybe_json(B / "sample5/attrs_truth.json")

    for tag, f, truth, label in (
            ("s7", "eval7_s7.json", t7, "sample7 模型（乙）"),
            ("s6", "eval7_s6.json", t6, "sample6 模型（带颜色）")):
        p = U / f
        if not p.exists():
            continue
        rows = json.loads(p.read_text("utf-8"))["rows"]
        run(tag, rows, truth, label)

    print("\n参照：如果模型只会套品类模板，这个数应该在 0.5 附近。")
    print("      明显高于 0.5 → 它确实在读这一张图，逐字匹配低只是同义词问题。")

    # 额外：看看有没有可用的 sample5 真值（含颜色），做颜色维度的检索
    p = U / "eval7_s6.json"
    if p.exists() and t5:
        rows = json.loads(p.read_text("utf-8"))["rows"]
        hitc = tot = 0
        for r in rows:
            c = clause_of(r["a_own"])
            head, _, _ = parse_clause(c)
            ti = (t5.get(r["image"]) or {}).get("color")
            if not ti:
                continue
            tot += 1
            if ti and ti in head:
                hitc += 1
        if tot:
            print(f"\ns6 报的颜色与 sample5 真值一致: {hitc}/{tot} = "
                  f"{hitc / tot:.3f}（说明颜色确实跟着图变）")


if __name__ == "__main__":
    main()
