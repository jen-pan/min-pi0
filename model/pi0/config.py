from dataclasses import dataclass

@dataclass
class BaseModelConfig:
    hidden_size: int
    intermediate_size: int
    use_final_norm: bool
    cache: bool

@dataclass
class MixtureConfig:
    image_text: BaseModelConfig = BaseModelConfig(
        hidden_size=2048,
        intermediate_size=16384,
        use_final_norm=False,
        cache=True
    )
    proprio: BaseModelConfig = BaseModelConfig(
        hidden_size=1024,
        intermediate_size=4096,
        use_final_norm=True,  # this gets zeroed out in the final layer
        cache=True
    )
    action: BaseModelConfig = BaseModelConfig(
        hidden_size=1024,
        intermediate_size=4096,
        use_final_norm=True,
        cache=False
    )

@dataclass
class MoEConfig:
    mixture: MixtureConfig
    num_hidden_layers: int = 18
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    head_dim: int = 256
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_bias: bool = False
    attention_dropout: float = 0.0
    pad_token_id: int = 0
    
@dataclass 
class Pi0Config:
    mixture: MixtureConfig
    pretrained_model_path: str = "/Users/jenny/Documents/min-pi0/paligemma-3b-pt-224"  # path to pretrained PaLiGemma model
    image_token_index: int = 257152
    vocab_size: int = 257216
    pad_token_id: int = 0
    flow_sampling: str = "beta"
    num_inference_steps: int = 10
    final_action_clip_value: float = 1.0  
    num_proprio_tokens: int = 1
    num_action_tokens: int = 4
    action_dim: int = 7 
    proprio_dim: int = 7  
    num_image_tokens: int = 256
    max_seq_len: int = 286  #TODO: just added 30 for text
    tokenizer_padding: str = "max_length"
    max_image_text_tokens: int = 286
    flow_sig_min: float = 0.001

@dataclass
class VisionConfig:
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int = 224
    patch_size: int = 14
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    num_image_tokens: int = 256

@dataclass
class VisionProjectorConfig:
    hidden_size: int = 1152
    projection_dim: int = 2048