from transformers import PretrainedConfig

class Qwen3_5Config(PretrainedConfig):
    model_type = "qwen3_5"
    def __init__(
        self,
        # 文本主干参数（Qwen3.5-0.8B）
        vocab_size=151936,
        hidden_size=1024,
        intermediate_size=2816,
        num_hidden_layers=16,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        # 视觉模块参数（对齐Qwen3-VL官方标准）
        vision_hidden_size=1152,
        vision_intermediate_size=4304,
        vision_num_attention_heads=16,
        vision_num_hidden_layers=24,
        image_token_index=151859,
        image_size=384,
        patch_size=14,
        projector_hidden_act="gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        # 文本参数
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        # 视觉参数
        self.vision_hidden_size = vision_hidden_size
        self.vision_intermediate_size = vision_intermediate_size
        self.vision_num_attention_heads = vision_num_attention_heads
        self.vision_num_hidden_layers = vision_num_hidden_layers
        self.image_token_index = image_token_index
        self.image_size = image_size
        self.patch_size = patch_size
        self.projector_hidden_act = projector_hidden_act