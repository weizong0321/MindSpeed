#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建 sample5（Step 2）：57 个人工核验过的纯子样式，覆盖 21 个品类。

与 sample4 的唯一变量
    sample4：14 个子样式 / 3 个品类 / 每子样式 ≤15 张
    sample5：57 个子样式 / 21 个品类 / 每子样式 ≤15 张
    → 子样式数量 +4 倍，其余保持一致，用来验证「数量增加能否把 D 拉上去、C 压到 0」

用法
    python build_sample5.py --src <图片根> --dst sample5 --per-sku 15 --eval-per-sku 3
"""

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# 每个品类两套问法（鞋子/箱包/服装等相近品类复用同一套措辞）
Q = {
    "watch": [
        ("outdoor", "户外徒步跑步 / 防水+夜光+透气表带",
         "<image>\n我平时户外徒步和跑步都戴，预算1000以内，要防水、能看时间和日期、夜里有背光、表带出汗不闷，这款合适吗有什么短板？"),
        ("daily", "日常通勤 / 耐刮+轻+显示日期星期",
         "<image>\n日常通勤戴，要耐刮、戴着不重、能显示日期和星期，这款怎么样？"),
    ],
    "bag_backpack": [
        ("commute", "通勤+出差 / 电脑+侧袋+背部透气",
         "<image>\n通勤加短途出差用，要能装15.6寸电脑、有侧袋放水杯、背部透气夏天不闷汗，这款符合吗？"),
        ("outing", "一日徒步 / 轻+装外套水+肩带不勒",
         "<image>\n周末一日徒步用，要轻、能装外套和水、肩带不勒肩，这款合适吗？"),
    ],
    "bag_handbag": [
        ("daily_carry", "通勤背 / 手机钥匙伞+轻+好搭",
         "<image>\n日常通勤背，要能放手机、钥匙和一把折叠伞，不重、好搭配，这款怎么样？"),
        ("shopping", "逛街 / 斜挎+五金不掉色+内里好找",
         "<image>\n出门逛街用，要能斜挎、五金不掉色、内里好找东西，这款合适吗？"),
    ],
    "purse_wallet": [
        ("daily", "日常 / 卡位+现金+塞裤兜+耐刮",
         "<image>\n日常用，要能放6张卡和几张现金、能塞进裤兜、皮面耐刮，这款怎么样？"),
        ("gift", "送人 / 质感+做工+不显廉价",
         "<image>\n送人用，要看起来有质感、做工整齐、不显廉价，这款合适吗？"),
    ],
    "luggage": [
        ("business", "出差一周 / 登机+万向轮+拉杆",
         "<image>\n出差一周用，要能登机、万向轮好推、拉杆不晃，这款符合吗？"),
        ("travel", "旅行 / 耐摔+锁+扩容",
         "<image>\n旅行用，要耐摔、有锁、扩容方便，这款合适吗？"),
    ],
    "shoes_sneaker": [
        ("daily_run", "夜跑5公里 / 缓震+透气+不磨脚",
         "<image>\n每天夜跑5公里，体重70kg正常足弓，要缓震好、透气、不磨脚，这款适合吗？"),
        ("student", "学生日常+体育课 / 百搭+耐穿+抓地",
         "<image>\n学生日常穿加体育课，要百搭、耐穿、鞋底抓地，预算300以内，这款怎么样？"),
    ],
    "shoes_leather": [
        ("commute", "上班通勤 / 不磨脚+防滑+配西裤",
         "<image>\n上班通勤穿，要皮鞋不磨脚、鞋底防滑、能配西裤，这款怎么样？"),
        ("formal", "正式场合 / 皮面光泽+鞋型挺+久站不累",
         "<image>\n正式场合穿，要皮面有光泽、鞋型挺括、久站不累，这款合适吗？"),
    ],
    "shoes_other": [
        ("summer", "夏天 / 透气+好走+好搭",
         "<image>\n夏天穿，要透气不闷脚、走路不累、好搭配裙子或短裤，这款怎么样？"),
        ("daily", "日常 / 鞋跟稳+不夹脚+久走不磨",
         "<image>\n日常穿，要鞋跟稳、不夹脚、走路多也不磨，这款合适吗？"),
    ],
    "shoes_kids": [
        ("daily", "给孩子 / 鞋底软+好穿脱+不压脚背",
         "<image>\n给孩子买，要鞋底软、好穿脱、脚背不压，这款怎么样？"),
        ("school", "上学 / 耐磨+防滑+配校服",
         "<image>\n孩子上学穿，要耐磨、防滑、能配校服，这款合适吗？"),
    ],
    "apparel_pants": [
        ("commute", "通勤 / 版型+口袋+不掉色",
         "<image>\n日常通勤穿，要版型不紧绷、口袋能放手机、洗后不掉色，这款怎么样？"),
        ("sport", "运动 / 弹力+透气+蹲下不勒",
         "<image>\n运动时穿，要弹力好、透气不闷、蹲下不勒，这款合适吗？"),
    ],
    "apparel_shorts": [
        ("summer", "夏天 / 透气+长度+好搭",
         "<image>\n夏天穿，要透气、长度合适、能配T恤，这款怎么样？"),
        ("sport", "运动 / 弹力+口袋+不粘腿",
         "<image>\n运动穿，要弹力好、有口袋、出汗不粘腿，这款合适吗？"),
    ],
    "apparel_top": [
        ("winter", "秋冬 / 保暖不臃肿+内外两穿+不起球",
         "<image>\n秋冬穿，要保暖不臃肿、内搭外穿都行、洗后不起球，这款怎么样？"),
        ("daily", "日常 / 版型正+领口不变形+好搭",
         "<image>\n日常穿，要版型正、领口不变形、好搭配，这款合适吗？"),
    ],
    "cosmetics": [
        ("dry_skin", "干皮 / 保湿不刺激+不搓泥",
         "<image>\n我皮肤偏干，要保湿不刺激、上脸不搓泥、能日常用，这款怎么样？"),
        ("gift", "送人 / 包装体面+口碑+适合多数肤质",
         "<image>\n送人用，要包装体面、口碑好、适合大多数肤质，这款合适吗？"),
    ],
    "personal_care": [
        ("daily", "日常 / 温和+好冲+气味不冲",
         "<image>\n日常用，要温和不刺激、好冲洗、气味不冲，这款怎么样？"),
        ("family", "全家 / 性价比+容量+放心",
         "<image>\n全家用，要性价比高、容量够、用着放心，这款合适吗？"),
    ],
    "appliance_large": [
        ("replace", "换新 / 省电+噪音小+售后",
         "<image>\n家里换新，要省电、噪音小、售后有保障，这款怎么样？"),
        ("small_space", "小空间 / 占地小+好装+能效高",
         "<image>\n空间有限，要占地小、安装方便、能效等级高，这款合适吗？"),
    ],
    "health_supplement": [
        ("elder", "给长辈 / 成分清楚+好吞+正规厂家",
         "<image>\n给长辈买，要成分清楚、好吞服、正规厂家，这款怎么样？"),
        ("daily", "日常补充 / 吃法简单+无异味+能长期",
         "<image>\n日常补充，要吃法简单、无异味、能长期吃，这款合适吗？"),
    ],
    "baby_products": [
        ("newborn", "新生儿 / 材质安全+好清洗+无刺激味",
         "<image>\n给新生儿用，要材质安全、好清洗、无刺激气味，这款怎么样？"),
        ("daily_care", "带娃 / 好消毒+耐用+好带",
         "<image>\n日常带娃用，要方便消毒、耐用、出门好带，这款合适吗？"),
    ],
    "toy_vehicle": [
        ("kid5", "5岁孩子 / 耐摔+无尖角+可玩性",
         "<image>\n给5岁孩子买，要耐摔、无尖角、可玩性高，这款怎么样？"),
        ("gift", "送人 / 包装完整+配件齐+耐玩",
         "<image>\n送人用，要包装完整、配件齐全、能玩久一点，这款合适吗？"),
    ],
    "jewelry": [
        ("daily", "日常戴 / 不过敏+不勾衣+好搭",
         "<image>\n日常戴，要不过敏、不勾衣服、好搭配，这款怎么样？"),
        ("gift", "送人 / 包装+款式经典+做工",
         "<image>\n送人用，要包装体面、款式经典、做工细致，这款合适吗？"),
    ],
    "household_detergent": [
        ("daily", "日常 / 去污+气味+不伤手",
         "<image>\n家里日常用，要去污力够、气味不刺鼻、不伤手，这款怎么样？"),
        ("stock", "囤货 / 性价比+好冲+适合机洗",
         "<image>\n囤货用，要性价比高、好冲洗、适合机洗，这款合适吗？"),
    ],
    "accessories": [
        ("daily", "日常 / 耐用+好搭+实惠",
         "<image>\n日常用，要耐用好搭、不挑衣服、价格实惠，这款怎么样？"),
        ("gift", "送人 / 质感+通用尺码+包装",
         "<image>\n送人用，要看起来有质感、尺码通用、包装体面，这款合适吗？"),
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assign", default="assignments.csv")
    ap.add_argument("--selected", default="selected_styles.json")
    ap.add_argument("--dst", default="sample5")
    ap.add_argument("--per-sku", type=int, default=15)
    ap.add_argument("--eval-per-sku", type=int, default=3)
    args = ap.parse_args()

    sel = json.loads(Path(args.selected).read_text(encoding="utf-8"))["styles"]
    by_sku = defaultdict(list)
    with open(args.assign, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            by_sku[r["sku"]].append(Path(r["path"]))

    dst = Path(args.dst).resolve()
    # answers.json 由人工/模型看过检查表后写出；没有就先只生成 user 轮
    ans_path = dst / "answers.json"
    answers = json.loads(ans_path.read_text(encoding="utf-8")) if ans_path.is_file() else {}
    print(f"answers.json：{'已加载' if answers else '尚未生成（本次只写 user 轮）'}")
    questions = {"_readme": [
        "sample5 问法库：57 个纯子样式 / 21 个品类，每类两套问法。",
        "问句必须以 <image>\\n 开头；rounds=1 时 user 是字符串。",
    ]}
    train, evalset, stats, problems = [], [], [], []
    for cat in sorted({s["category"] for s in sel}):
        if cat not in Q:
            problems.append(f"品类 {cat} 没有问法定义")
            continue
        questions[cat] = {
            "name_zh": cat,
            "questions": [{"id": qid, "desc": d, "rounds": 1, "user": u}
                          for qid, d, u in Q[cat]],
        }

    for s in sel:
        cat, sku = s["category"], s["sku"]
        paths = sorted(by_sku.get(sku) or [])
        if len(paths) < 8:
            problems.append(f"{sku} 只有 {len(paths)} 张，跳过")
            continue
        paths = paths[:args.per_sku]
        if len(paths) < 8:
            problems.append(f"{sku} 取图后只剩 {len(paths)} 张，跳过")
            continue
        ev = paths[-args.eval_per_sku:]
        tr = paths[:-args.eval_per_sku]
        out = dst / "images" / cat / sku
        out.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(paths):
            if p.is_file():
                shutil.copy2(p, out / f"{i:03d}{p.suffix.lower()}")
        rel = f"images/{cat}/{sku}"
        meta = answers.get(f"{cat}/{sku}") or {}
        for qid, _, u in Q[cat]:
            a = (meta.get("qa") or {}).get(qid)
            if not a:
                problems.append(f"{cat}/{sku} 缺问法 {qid} 的答案")
                continue
            convs = [{"from": "user", "value": u},
                     {"from": "assistant", "value": a}]
            for p in tr:
                idx = paths.index(p)
                train.append({"conversations": convs, "image": f"{rel}/{idx:03d}.jpg",
                              "qid": qid})
            for p in ev:
                idx = paths.index(p)
                evalset.append({"conversations": convs, "image": f"{rel}/{idx:03d}.jpg",
                                "qid": qid})
        stats.append((cat, sku, len(paths)))

    (dst / "questions.json").write_text(
        json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
    (dst / "train.json").write_text(
        json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8")
    (dst / "eval.json").write_text(
        json.dumps(evalset, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print(f"sample5：{len(stats)} 个子样式 / {sum(s[2] for s in stats)} 张图")
    print(f"  训练样本 {len(train)} 条 / 验证样本 {len(evalset)} 条")
    print(f"  品类数 {len({s[0] for s in stats})}")
    print("=" * 70)
    print(f"\n{'品类':<22}{'子样式数':>8}{'图片':>7}")
    print("-" * 40)
    by_cat = defaultdict(lambda: [0, 0])
    for cat, _, n in stats:
        by_cat[cat][0] += 1
        by_cat[cat][1] += n
    for cat, (c, n) in sorted(by_cat.items(), key=lambda x: -x[1][1]):
        print(f"{cat:<22}{c:>8}{n:>7}")
    print("-" * 40)
    print(f"{'合计':<22}{len(stats):>8}{sum(s[2] for s in stats):>7}")
    if problems:
        print(f"\n[问题] {len(problems)} 条：")
        for p in problems[:10]:
            print("   " + p)
    print(f"\n下一步：answers.json 需要为 {len(stats)} 个子样式各写 2 段答案")
    return 0


if __name__ == "__main__":
    sys.exit(main())
