from .qwen3_5_config import Qwen3_5Config
from .modeling_qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
)

# 多模态主模型：训练/推理实际使用的类，已通过 @model_register.register("qwen3_5") 注册，
# 配置里的 model.model_id = "qwen3_5" 即对应此类。
# 保留旧名 Qwen3_5VLForCausalLM 作为别名，兼容既有调用（如 skill/ecommerce_guide_skill）。
Qwen3_5VLForCausalLM = Qwen3_5ForConditionalGeneration

__all__ = [
    "Qwen3_5Config",
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5VLForCausalLM",
]
