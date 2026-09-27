#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""电商智能导购助手 —— 交互式入口（可真正推理）。

设计原则（基于本项目的实测结论）
    模型只负责**读图上客观可判别的结构属性**，正文由确定性模板渲染。
    原因：实测颜色/自由文本描述的标注跨标注者一致率只有 0.13–0.70，
    而闭词表属性达到 0.85–1.00，且 0.8B 学生能把其中一部分学对
    （整组属性全对 75.6%，对照"完全不看图"51.3%）。
    因此本 Skill 不对颜色、材质、尺码、耐用性等图上无法核实的内容做断言。

用法
    python run_chat.py                       # 交互模式
    python run_chat.py --model <HF目录>      # 指定权重
    交互中输入：
        纯文字：  这个包适合通勤吗
        图文：    /path/to/a.jpg|这个包适合通勤吗
        exit      退出
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_and_render_model import AttrReader  # noqa: E402


def describe(reader, image, question, cat=None):
    """返回 (抽取到的属性, 最终答案)"""
    return reader.read(image, question, cat=cat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default="/workspace/user_data/output/qwen3_5_0.8B_sample8b_hf")
    ap.add_argument("--category", default=None,
                    help="强制品类（目录名，如 bag_backpack）；不传则从图片路径推断")
    a = ap.parse_args()

    print("=" * 60)
    print("  电商智能导购助手（多模态）")
    print("=" * 60)
    print("  纯文字提问：直接输入需求")
    print("  图文提问  ：图片路径|你的需求")
    print("  退出      ：exit")
    print("-" * 60)
    print("  提示：图中可核实的客观结构属性会被读出；")
    print("        颜色/材质/尺码/耐用性等图上判断不了，会明确说明。")
    print("=" * 60)

    t0 = time.time()
    reader = AttrReader(a.model)
    print(f"✅ 模型加载完成（{time.time() - t0:.0f}s）\n")

    while True:
        try:
            raw = input("顾客：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n导购AI：对话已结束")
            break
        if not raw:
            continue
        if raw.lower() in ("exit", "quit", "退出"):
            print("导购AI：感谢使用，再见")
            break

        if "|" in raw:
            img, query = raw.split("|", 1)
            img, query = img.strip(), query.strip()
            if not os.path.exists(img):
                print(f"导购AI：找不到图片 {img}")
                continue
            try:
                attrs, answer = describe(reader, img, query, a.category)
            except Exception as e:                              # noqa: BLE001
                print(f"导购AI：处理失败 {type(e).__name__}: {e}")
                continue
            print(f"导购AI：{answer}")
            if attrs:
                print(f"        （读出的属性：{json.dumps(attrs, ensure_ascii=False)}）")
            else:
                print("        （图中未能读出可信属性，仅给出品类与免责说明）")
        else:
            print("导购AI：请输入商品图片路径以便看图回答，格式："
                  "图片路径|你的需求")


if __name__ == "__main__":
    main()
