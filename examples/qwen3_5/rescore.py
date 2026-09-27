#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修正判据后重新打分。

两个必须修的问题
  1) 颜色真值只写了"银色""棕色"，但模型输出"银""棕"——子串匹配判成"错"。
     实际那是我的判据 bug，不是模型的错。这里补上单字说法。
  2) 部件用严格子串匹配太苛刻：真值"裤腰"，模型说"松紧腰"，语义一样却判 0。
     这里给每个部件加同义说法，同时保留严格值以便对比。
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

COLOR_ALTS = {
    "银色": ["银色", "银白", "银灰", "银"],
    "棕色": ["棕色", "棕", "褐色", "咖啡色", "深棕"],
    "灰色": ["灰色", "灰", "深灰", "银灰", "灰白"],
    "黑色": ["黑色", "黑", "纯黑"],
    "白色": ["白色", "白"],
    "蓝色": ["蓝色", "蓝", "深蓝", "藏青"],
    "红色": ["红色", "红", "正红", "酒红"],
    "绿色": ["绿色", "绿", "墨绿", "深绿", "青色", "蓝绿"],
    "黄色": ["黄色", "黄"],
    "粉色": ["粉色", "粉", "蓝", "蓝色"],
    "透明": ["透明", "无色", "白色", "白"],
}

# 部件同义说法（真值词 → 可接受的说法）
PART_SYN = {
    "裤腰": ["裤腰", "松紧腰", "腰头", "腰部", "抽绳"],
    "裤腿": ["裤腿", "裤管", "腿部", "裤身"],
    "口袋": ["口袋", "兜", "袋"],
    "裤脚": ["裤脚", "裤口", "下摆"],
    "鞋面": ["鞋面", "鞋身", "鞋帮", "网面", "针织"],
    "鞋底": ["鞋底", "中底", "大底", "厚底", "白底"],
    "鞋带": ["鞋带", "系带", "魔术贴", "鞋带孔"],
    "鞋头": ["鞋头", "尖头", "圆头"],
    "鞋跟": ["鞋跟", "后跟"],
    "镜框": ["镜框", "镜架", "框架", "金属框"],
    "镜片": ["镜片", "镜面", "深色镜片"],
    "镜腿": ["镜腿", "镜脚"],
    "鼻托": ["鼻托"],
    "表盘": ["表盘", "表圈", "刻度", "小表盘"],
    "表带": ["表带", "皮带", "金属表带", "表链"],
    "表冠": ["表冠", "表把"],
    "链条": ["链条", "链子", "细链"],
    "吊坠": ["吊坠", "坠子", "坠饰"],
    "项链": ["项链", "颈链"],
    "脖子": ["脖子", "颈部", "颈"],
    "礼盒": ["礼盒", "包装盒", "盒", "内衬"],
    "内衬": ["内衬", "衬里"],
    "瓶盖": ["瓶盖", "盖子", "瓶帽", "盖"],
    "瓶身": ["瓶身", "瓶体", "管身", "圆柱瓶身"],
    "标签": ["标签", "标识", "文字", "字印", "说明"],
    "条码": ["条码", "条形码"],
    "文字": ["文字", "字印", "标识", "标签", "说明"],
    "奶嘴": ["奶嘴", "奶瓶嘴", "吸嘴"],
    "手柄": ["手柄", "把手", "提手", "双肩带", "手柄"],
    "刻度": ["刻度", "容量", "毫升"],
    "拉链": ["拉链", "拉锁"],
    "提手": ["提手", "把手", "手提"],
    "双肩带": ["双肩带", "肩带", "背带", "双肩"],
    "前袋": ["前袋", "侧袋", "口袋", "袋"],
    "管身": ["管身", "瓶身", "软管", "挤压管"],
    "翻盖": ["翻盖", "搭扣", "翻折"],
    "金属扣": ["金属扣", "扣头", "搭扣", "卡扣", "金属"],
    "扣头": ["扣头", "金属扣", "搭扣", "卡扣"],
    "打孔": ["打孔", "孔", "打眼"],
    "皮带": ["皮带", "带身", "皮革带"],
    "车轮": ["车轮", "轮胎", "轮子", "越野轮胎"],
    "车身": ["车身", "车体", "车壳"],
    "车斗": ["车斗", "车厢", "后斗", "车箱"],
    "轮胎": ["轮胎", "车轮", "轮子"],
    "包装袋": ["包装袋", "袋装", "包装", "袋"],
    "撕口": ["撕口", "开口", "易撕"],
    "出风口": ["出风口", "风口", "风栅"],
    "面板": ["面板", "控制面板", "操作面板"],
    "机身": ["机身", "机体", "外壳"],
    "机盖": ["机盖", "顶盖", "盖板", "透明顶盖"],
    "控制面板": ["控制面板", "旋钮", "面板", "按键"],
    "底座": ["底座", "底部", "支脚"],
    "显示屏": ["显示屏", "显示", "数字显示"],
    "膏体": ["膏体", "口红", "膏"],
    "盖子": ["盖子", "盖", "外盖"],
    "金色": ["金色", "金", "玫瑰金"],
    "手": ["手", "手持", "手指"],
    "三条纹": ["三条纹", "条纹", "logo", "标识"],
    "系带": ["系带", "抽绳", "绳"],
    "腰头": ["腰头", "裤腰", "松紧腰", "腰部"],
    "抽绳": ["抽绳", "系带", "绳"],
    "袖口": ["袖口", "袖"],
    "立领": ["立领", "领口", "领"],
    "绗缝": ["绗缝", "格纹", "压线"],
    "内衬": ["内衬", "衬里", "里衬"],
    "画面": ["画面", "图案", "印刷"],
}


def hit_any(words, text):
    return [w for w in words if w in text]


def score(rows, truth):
    c_ok = c_strict = 0
    p_exact = p_syn = 0.0
    n = 0
    for r in rows:
        t = truth.get(r["id"])
        if not t:
            continue
        n += 1
        txt = r["raw"]
        alts = COLOR_ALTS.get(t["truth_color"], [t["truth_color"]])
        c_ok += bool(hit_any(alts, txt))
        c_strict += t["truth_color"] in txt
        parts = t["parts"]
        p_exact += sum(1 for p in parts if p in txt) / len(parts)
        p_syn += sum(1 for p in parts
                     if hit_any(PART_SYN.get(p, [p]), txt)) / len(parts)
    if not n:
        return None
    return {"n": n, "color": c_ok / n, "color_strict": c_strict / n,
            "parts": p_exact / n, "parts_syn": p_syn / n}


def main():
    U = MM_WORK
    truth = {t["id"]: t for t in json.loads(Path(
        str(_REPO_ROOT / "data/ecommerce_multimodal/sample5/gate_truth.json")
    ).read_text("utf-8"))["items"]}
    unamb = {k for k, v in truth.items() if not v["ambiguous"]}

    runs = {}
    for t in ("base", "s6", "t4b_cap", "t4b_ans", "t30b_cap"):
        p = U / f"gate_{t}.json"
        if p.exists():
            runs[t] = json.loads(p.read_text())["rows"]
    if not runs:
        print("没有结果文件")
        return 1

    print("判据修正后的绝对准确率（真值=人工标注，非循环）")
    print(f"{'模型':<10s} {'n':>3s} {'颜色':>7s} {'颜色(严)':>8s} "
          f"{'部件':>7s} {'部件(同义)':>10s}")
    print("-" * 52)
    for t, rows in runs.items():
        s = score(rows, truth)
        print(f"{t:<10s} {s['n']:3d} {s['color']:7.3f} {s['color_strict']:8.3f} "
              f"{s['parts']:7.3f} {s['parts_syn']:10.3f}")

    common = set.intersection(*[{r["id"] for r in rows} for rows in runs.values()])
    print(f"\n同一子集（{len(common)} 张，所有模型都有结果）")
    print(f"{'模型':<10s} {'颜色':>7s} {'部件(同义)':>10s}")
    print("-" * 30)
    for t, rows in runs.items():
        s = score([r for r in rows if r["id"] in common], truth)
        print(f"{t:<10s} {s['color']:7.3f} {s['parts_syn']:10.3f}")

    print(f"\n只看非歧义图（{len(unamb)} 张）")
    print(f"{'模型':<10s} {'颜色':>7s} {'部件(同义)':>10s}")
    print("-" * 30)
    for t, rows in runs.items():
        s = score([r for r in rows if r["id"] in unamb], truth)
        print(f"{t:<10s} {s['color']:7.3f} {s['parts_syn']:10.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
