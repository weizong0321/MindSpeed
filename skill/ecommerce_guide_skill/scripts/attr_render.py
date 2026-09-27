#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""属性值 ⇄ 自然语言短语 的双向映射。

为什么要双向
    答案里要读起来自然（"低帮、系带、带侧袋"），
    但评测要能精确还原成 (属性, 取值) 才能算平衡准确率。
    所以渲染是有损自然语言的唯一入口，必须可逆。

渲染规则
    - 取值本身就是形容词的（低帮/系带/圆头/拉链…）→ 直接用
    - 有/无 类取值会读起来别扭（"有、无"）→ 渲染成 带侧袋 / 无侧袋
      这也让短语自带属性名，反解时不会歧义
"""

# (品类, 属性, 取值) -> 短语
SPECIAL = {
    ("bag_backpack", "side_pocket"): {"有": "带侧袋", "无": "无侧袋"},
    ("bag_handbag", "side_pocket"): {"有": "带侧袋", "无": "无侧袋"},
    ("luggage", "side_pocket"): {"有": "带侧袋", "无": "无侧袋"},
    ("purse_wallet", "strap"): {"有": "带背带", "无": "无背带"},
    ("accessories", "metal_parts"): {"有": "带金属件", "无": "无金属件"},
    ("accessories", "foldable"): {"能": "可折叠", "不能": "不可折叠"},
    ("baby_products", "handle"): {"有": "带手柄", "无": "无手柄"},
    ("baby_products", "nipple"): {"有": "带奶嘴", "无": "无奶嘴"},
    ("health_supplement", "label"): {"有": "带标签", "无": "无标签"},
    ("household_detergent", "pump"): {"有": "带泵头", "无": "无泵头"},
    ("household_detergent", "tear"): {"有": "带撕口", "无": "无撕口"},
    ("personal_care", "brush"): {"有": "带刷头", "无": "无刷头"},
    ("toy_vehicle", "remote"): {"能": "可遥控", "不能": "不可遥控"},
    ("appliance_large", "display"): {"有": "带显示屏", "无": "无显示屏"},
    ("apparel_top", "quilted"): {"有": "带绗缝", "无": "无绗缝"},
    ("cosmetics", "cap"): {"有": "带盖子", "无": "无盖子"},
    ("jewelry", "pendant"): {"有": "带吊坠", "无": "无吊坠"},
    ("apparel_pants", "pockets"): {"有": "带口袋", "无": "无口袋"},
    ("apparel_shorts", "pockets"): {"有": "带口袋", "无": "无口袋"},
    ("shoes_other", "tip"): {"圆头": "圆头", "尖头": "尖头"},
}

UNSURE = "不确定"


def render_one(cat, attr, value):
    v = str(value).strip()
    if v == UNSURE or not v:
        return None
    sp = SPECIAL.get((cat, attr))
    if sp and v in sp:
        return sp[v]
    return v


def render_values(cat, attr_values, order=None, max_items=4):
    """attr_values: {attr: value} -> 短语列表（按 order 排序）"""
    keys = order or sorted(attr_values)
    out = []
    for k in keys:
        if k not in attr_values:
            continue
        p = render_one(cat, k, attr_values[k])
        if p and p not in out:
            out.append(p)
    return out[:max_items]


def build_reverse(cat, spec):
    """(短语 -> (属性, 取值))，用于从答案里反解属性"""
    rev = {}
    for item in spec:
        k, opts = item["key"], item["options"]
        for o in opts:
            p = render_one(cat, k, o)
            if p:
                rev[p] = (k, o)
    return rev


def parse_values(cat, clause, spec):
    """从句子里反解出 {属性: 取值}；匹配不到的短语忽略"""
    rev = build_reverse(cat, spec)
    out = {}
    for p, (k, v) in rev.items():
        if k in out:
            continue
        if p in clause:
            out[k] = v
    return out


if __name__ == "__main__":
    import json
    import sys
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    spec = json.loads(
        open("/workspace/user_data/attr_final_spec.json", encoding="utf-8").read())
    demos = [
        ("shoes_leather", {"closure": "系带", "height": "低帮", "sole": "厚底"}),
        ("bag_backpack", {"closure": "拉链", "side_pocket": "有"}),
        ("purse_wallet", {"closure": "翻盖", "strap": "无"}),
    ]
    for cat, av in demos:
        phr = render_values(cat, av)
        clause = f"{cat}，" + "、".join(phr)
        back = parse_values(cat, clause, spec.get(cat, []))
        print(f"{cat:16s} {av} -> {phr}")
        print(f"{'':16s} 反解 -> {back}   一致={back == av}")
