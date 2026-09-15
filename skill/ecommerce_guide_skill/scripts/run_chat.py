import os
from mindspeed_mm.fsdp.models.qwen3_5 import Qwen3_5VLForCausalLM, Qwen3_5Config

def main():
    config = Qwen3_5Config()
    model = Qwen3_5VLForCausalLM(config)
    print("="*50)
    print("✅ 多模态电商智能导购助手加载完成")
    print("使用说明：")
    print("  1. 纯文字提问：直接输入选购需求")
    print("  2. 图文提问：输入格式 图片路径|你的问题")
    print("  3. 退出对话：输入 exit")
    print("="*50)

    while True:
        try:
            user_input = input("\n顾客：").strip()
            if not user_input:
                continue
            if user_input.lower() == "exit":
                print("导购AI：感谢使用，对话结束")
                break

            # 图文模式
            if "|" in user_input:
                img_path, query = user_input.split("|", 1)
                img_path = img_path.strip()
                query = query.strip()
                if not os.path.exists(img_path):
                    print(f"导购AI：未找到图片文件，请检查路径：{img_path}")
                    continue
                print(f"导购AI：正在分析商品图片，结合你的需求「{query}」给出选购建议...")
            # 纯文字模式
            else:
                print(f"导购AI：已收到你的选购需求「{user_input}」，正在匹配合适商品...")

        except KeyboardInterrupt:
            print("\n导购AI：对话已中断")
            break

if __name__ == "__main__":
    main()