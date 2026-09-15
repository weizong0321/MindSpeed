import os
import torch
import argparse
from tqdm import tqdm

def convert_hf_to_dcp(input_dir, output_dir):
    """将HuggingFace Qwen3.5-VL权重转换为MindSpeed-MM DCP格式"""
    os.makedirs(output_dir, exist_ok=True)
    hf_state_dict = {}

    # 加载所有权重文件
    for f in os.listdir(input_dir):
        if f.endswith(".bin") or f.endswith(".safetensors"):
            file_path = os.path.join(input_dir, f)
            if f.endswith(".safetensors"):
                from safetensors.torch import load_file
                state = load_file(file_path)
            else:
                state = torch.load(file_path, map_location="cpu")
            hf_state_dict.update(state)

    # 权重映射转换（文本主干 + 视觉模块）
    dcp_state_dict = {}
    mapping = {
        "model.embed_tokens.weight": "embed_tokens.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "lm_head.weight",
        "visual.encoder.patch_embed.weight": "vision_encoder.patch_embed.weight",
        "visual.encoder.post_layernorm.weight": "vision_encoder.post_layernorm.weight",
        "visual.encoder.post_layernorm.bias": "vision_encoder.post_layernorm.bias",
        "visual.mlp.0.weight": "vision_proj.linear_1.weight",
        "visual.mlp.0.bias": "vision_proj.linear_1.bias",
        "visual.mlp.2.weight": "vision_proj.linear_2.weight",
        "visual.mlp.2.bias": "vision_proj.linear_2.bias",
    }

    for hf_key, dcp_key in tqdm(mapping.items(), desc="转换权重"):
        if hf_key in hf_state_dict:
            dcp_state_dict[dcp_key] = hf_state_dict[hf_key]

    # 保存DCP格式权重
    torch.save(dcp_state_dict, os.path.join(output_dir, "model_00001.pt"))
    print(f"✅ 权重转换完成，保存至：{output_dir}")
    print(f"   共转换 {len(dcp_state_dict)} 个参数张量")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="HF权重目录")
    parser.add_argument("--output_dir", required=True, help="DCP权重输出目录")
    args = parser.parse_args()
    convert_hf_to_dcp(args.input_dir, args.output_dir)