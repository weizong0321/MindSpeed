#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""「属性抽取 + 代码渲染」流水线的可用性对比。

做法
    模型输出 = 整个答案，但第一句「图中为…」就是属性陈述。
    所以流水线只要：取第一句 -> 反解属性 -> 丢掉模型写的正文 -> 用代码重新渲染。
    **不需要重新推理**（复用 eval8_s8b.json 里已保存的输出）。

对比
    before：模型直接输出的整段答案
    after ：只保留属性，正文由 render_answer 生成
    逐项数缺陷：品类名词写错 / 逻辑别扭句 / 长度 / 属性准确率
"""
# --- path shim (portable paths; auto-generated) ---
import os as _os
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
MM_DATA = _Path(_os.environ.get("MM_DATA", _REPO_ROOT / "data" / "ecommerce_multimodal"))
MM_WORK = _Path(_os.environ.get("MM_WORK", _REPO_ROOT / "work"))
MM_EX = _REPO_ROOT / "examples" / "qwen3_5"
_skill_scripts = _REPO_ROOT / "skill" / "ecommerce_guide_skill" / "scripts"
if str(_skill_scripts) not in _sys.path:
    _sys.path.insert(0, str(_skill_scripts))
# --- end shim ---


import json
import re
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(_Path(__file__).resolve().parent))
from attr_render import parse_values  # noqa: E402
from render_answer import render, CAT_CN  # noqa: E402
import attr_spec  # noqa: E402

U = MM_WORK
B = Path(str(_REPO_ROOT / "data/ecommerce_multimodal/sample8b"))

d = json.loads((U / "eval8_s8b.json").read_text("utf-8"))
rows = d["rows"]
selected = json.loads((B / "attr_spec.json").read_text("utf-8"))

full_spec = {}
for cat, attrs in selected.items():
    m = {k: o for k, _, o in attr_spec.ATTRS.get(cat, [])}
    full_spec[cat] = [{"key": k, "options": m.get(k, [])} for k in attrs]

# 问题文本
qmap = {}
for r in json.loads((B / "eval.json").read_text("utf-8")):
    qmap.setdefault(r["image"], r["conversations"][0]["value"])

def clause_of(a):
    m = re.search(r"图中为(.*?)。", a)
    return m.group(1).strip() if m else None

def head_of(clause):
    return (clause or "").split("，")[0]

# ---- 逐条跑流水线 ----
out = []
for r in rows:
    cat = r["cat"]                      # 目录名，权威
    clause = r["clause_own"] or ""
    # 1) 反解模型读出的属性（用模型自己的输出）
    pred = parse_values(cat, clause, full_spec[cat])
    # 若反解为空（模型格式跑偏），退回用评测时已解析的结果
    if not pred:
        pred = r["pred_own"]
    q = qmap.get(r["image"], "")
    ans_new = render(cat, pred, q)
    out.append({"image": r["image"], "cat": cat, "question": q,
                "raw_before": r["a_own"], "attrs": pred,
                "answer_after": ans_new,
                "gold": r["gold_own"]})

# ---- 缺陷对比 ----
def audit(key):
    n = len(out)
    cat_wrong = bad_short = 0
    lens = []
    for o in out:
        a = o[key]
        lens.append(len(a))
        if key == "raw_before":
            h = head_of(clause_of(a))
            if CAT_CN.get(o["cat"]) and h and h != CAT_CN[o["cat"]]:
                cat_wrong += 1
        m = re.search(r"短板是([^。]*)。", a)
        if m and ("看不到" in m.group(1) or "图中没有" in m.group(1)):
            bad_short += 1
    return cat_wrong, bad_short, sum(lens) / n, n

print("=" * 96)
print(f"流水线对比（{len(out)} 张图）")
print("=" * 96)
for key, label in (("raw_before", "before：模型直接输出"),
                   ("answer_after", "after ：属性+代码渲染")):
    cw, bs, ln, n = audit(key)
    print(f"\n【{label}】")
    print(f"  品类名词写错        {cw}/{n} = {cw/n:.4f}")
    print(f"  逻辑别扭的短板句    {bs}/{n} = {bs/n:.4f}")
    print(f"  平均长度            {ln:.0f} 字")

# 属性准确率（流水线用的就是模型读出的属性，所以和之前一致）
tot = hit = 0
for o in out:
    for k, v in (o["gold"] or {}).items():
        tot += 1
        if o["attrs"].get(k) == v:
            hit += 1
print(f"\n  单属性准确率        {hit}/{tot} = {hit/tot:.3f}（与评测一致，未变）")

# 弃权：反解不出任何属性 → 只能只报品类
empty = [o for o in out if not o["attrs"]]
print(f"  弃权（读不出任何属性）{len(empty)}/{len(out)} = "
      f"{len(empty)/len(out):.4f}  ← 渲染没法补救模型没读出来的东西")
# 这些弃权里，有多少是模型把品类认错了（用了别的品类的属性词）
cat_conf = 0
for o in empty:
    head = head_of(clause_of(o["raw_before"]))
    if head and head in set(CAT_CN.values()) and head != CAT_CN.get(o["cat"]):
        cat_conf += 1
print(f"     其中是模型把品类认错（说了别的品类的词）: {cat_conf}/{len(empty)}")
if empty[:3]:
    print("     例：")
    for o in empty[:3]:
        print(f"       {o['image']}")
        print(f"         真值品类 {CAT_CN.get(o['cat'])}，从句: {clause_of(o['raw_before'])}")

# 有属性可说的图中，属性是否全对
withattr = [o for o in out if o["attrs"]]
allok = sum(1 for o in withattr if o["gold"]
            and all(o["attrs"].get(k) == v for k, v in o["gold"].items()))
print(f"\n  有属性输出的图 {len(withattr)} 张，其中整组全对 "
      f"{allok}/{len(withattr)} = {allok/max(len(withattr),1):.3f}")
print(f"  折算到全部 {len(out)} 张：{allok}/{len(out)} = {allok/len(out):.3f}")

print("\n" + "=" * 96)
print("逐条对照（前 10 条）")
print("=" * 96)
for o in out[:10]:
    print(f"\n[{'/'.join(o['image'].split('/')[-3:])}]  真值 {o['gold']}  读出 {o['attrs']}")
    print(f"  before: {o['raw_before']}")
    print(f"  after : {o['answer_after']}")

(U / "pipeline_outputs.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n→ {U / 'pipeline_outputs.json'}")
