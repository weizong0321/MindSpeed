#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""确定性答案渲染器：模型只负责读属性，文案由代码生成。

为什么这样拆
    答案模板是早期按子样式手写的 78 份，实测：
      · 41% 含逻辑别扭的句子（"短板是图中看不到…"）
      · 属性值被塞进不匹配的槽位 → "镜框偏太阳镜的位置"
      · 6.4% 的答案品类名词写错 → 真值"配饰"却写"钱包"
    这些都**在训练数据里就存在**，模型只是忠实复现。所以把正文交给代码渲染，
    可以把这类缺陷一次性清零：
      · 品类名词取目录名（权威），不会写错
      · 句式固定且通顺，不会出现病句/槽位错配
      · 对问题里的诉求统一诚实免责，不编造图上判断不了的东西

输出结构
    第 1 句：图中是什么 + 读出来的客观属性（只写经过验证的属性）
    第 2 句：复述问题的诉求 + 说明图上判断不了 + 以商品页为准
"""

import re

CAT_CN = {
    "accessories": "配饰", "apparel_pants": "长裤", "apparel_shorts": "短裤",
    "apparel_top": "上衣", "apparel_trip": "外套", "appliance_large": "电器",
    "baby_products": "婴儿用品", "bag_backpack": "双肩包", "bag_handbag": "手提包",
    "cosmetics": "化妆品", "health_supplement": "保健品",
    "household_detergent": "洗护用品", "jewelry": "首饰", "luggage": "行李箱",
    "personal_care": "个护用品", "purse_wallet": "钱包", "shoes_kids": "童鞋",
    "shoes_leather": "皮鞋", "shoes_other": "鞋", "shoes_sneaker": "运动鞋",
    "toy_vehicle": "玩具车", "watch": "手表",
}

# 品类名词的量词/搭配，让第一句读起来自然
HEAD = {
    "accessories": "图中是一件配饰",
    "apparel_pants": "图中是一条长裤",
    "apparel_shorts": "图中是一条短裤",
    "apparel_top": "图中是一件上衣",
    "apparel_trip": "图中是一件外套",
    "appliance_large": "图中是一台电器",
    "baby_products": "图中是一件婴儿用品",
    "bag_backpack": "图中是一个双肩包",
    "bag_handbag": "图中是一个手提包",
    "cosmetics": "图中是一件化妆品",
    "health_supplement": "图中是一瓶保健品",
    "household_detergent": "图中是一件洗护用品",
    "jewelry": "图中是一件首饰",
    "luggage": "图中是一个行李箱",
    "personal_care": "图中是一件个护用品",
    "purse_wallet": "图中是一个钱包",
    "shoes_kids": "图中是一双童鞋",
    "shoes_leather": "图中是一双皮鞋",
    "shoes_other": "图中是一双鞋",
    "shoes_sneaker": "图中是一双运动鞋",
    "toy_vehicle": "图中是一辆玩具车",
    "watch": "图中是一块手表",
}

DISCLAIMER = "这些从图上判断不了，需要以商品页标注为准。"


def constraints_of(question):
    """从问题文本里抽出诉求（这些是问题自己给的，复述它们不算视觉断言）"""
    t = question.split("\n", 1)[-1]
    i = t.find("要")
    if i < 0:
        return []
    j = max(t.rfind("，这款"), t.rfind("，这"), t.rfind("，合适"))
    seg = t[i + 1:j] if j > i else t[i + 1:]
    seg = seg.rstrip("？。，")
    return [x.strip() for x in re.split(r"[、,]", seg) if x.strip()]


def render(cat, attrs, question="", attr_order=None):
    """cat: 目录名（权威品类）；attrs: {属性: 取值}；question: 原始问题"""
    head = HEAD.get(cat, f"图中是{CAT_CN.get(cat, '商品')}")
    # 复用 attr_render：它把 有/无 渲染成"带侧袋/无背带"，可读且可反解
    try:
        from attr_render import render_values
        order = attr_order or _spec_order(cat)
        vals = render_values(cat, attrs, order=order)
    except Exception:                                          # noqa: BLE001
        vals = [str(attrs[k]).strip() for k in (attr_order or sorted(attrs))
                if k in attrs and str(attrs[k]).strip()
                and "不确定" not in str(attrs[k])]
    seg1 = head + ("，" + "、".join(vals) if vals else "") + "。"

    cs = constraints_of(question)
    if cs:
        seg2 = "、".join(cs) + "这些从图上判断不了，需要以商品页标注为准。"
    else:
        seg2 = "更细的参数从图上判断不了，需要以商品页标注为准。"
    return seg1 + seg2


def _spec_order(cat):
    """按 attr_spec 里定义的顺序排，读起来更自然（材质在前、有无在后）"""
    try:
        import attr_spec
        return [k for k, _, _ in attr_spec.ATTRS.get(cat, [])]
    except Exception:                                          # noqa: BLE001
        return None


if __name__ == "__main__":
    import sys
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    demos = [
        ("shoes_leather", {"closure": "系带", "height": "低帮", "sole": "厚底"},
         "<image>\n日常通勤穿，要版型不紧绷、口袋能放手机、洗后不掉色，这款怎么样？"),
        ("watch", {"strap_material": "金属", "dial_shape": "圆形"}, "<image>\n送人用，要看起来有质感、尺码通用、包装体面，这款合适吗？"),
        ("purse_wallet", {"closure": "翻盖", "strap": "无"}, ""),
        ("accessories", {"type": "皮带"}, "<image>\n日常用，要耐用好搭、不挑衣服、价格实惠，这款怎么样？"),
    ]
    for cat, av, q in demos:
        print(f"[{cat}] {av}")
        print(f"  {render(cat, av, q)}")
        print(f"  诉求解析: {constraints_of(q)}")
