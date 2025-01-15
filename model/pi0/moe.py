from typing import Dict, Tuple, Optional
import flax.linen as nn
import jax.numpy as jnp
from omegaconf import OmegaConf
from model.kv_cache import MultiModalKVCache
from model.pi0.modules import ModalStack

def moe_layers(
    modality_stacks: Dict[str, ModalStack],
    attention_mask: jnp.ndarray,
    input_idx: Dict[str, jnp.ndarray],
    input_embeddings: Dict[str, jnp.ndarray],
    layer_idx: int,
    kv_cache: Optional[MultiModalKVCache] = None,
    use_cache: bool = True,
    is_final_layer: bool = False,
) -> Dict[str, jnp.ndarray]:
    """Forward pass through layers."""
    modality_names = list(input_embeddings.keys())
    
    outputs = {}
    for name in modality_names:
        layer_cache = kv_cache.get_layer_cache(name, layer_idx) if use_cache and kv_cache else None
        
        hidden_states, new_kv = modality_stacks[name].layers[layer_idx](
            input_embeddings[name],
            attention_mask,
            input_idx[name],
            layer_cache
        )
        outputs[name] = hidden_states
        
        # update cache if needed
        if use_cache and kv_cache and new_kv is not None:
            key_states, value_states = new_kv
            kv_cache.update(name, layer_idx, key_states, value_states)
    
    # zero out image_text and proprio if final layer
    if is_final_layer:
        for name in ["image_text", "proprio"]:
            if name in outputs:
                outputs[name] = jnp.zeros_like(outputs[name])
    
    return outputs

class MoE(nn.Module):
    """Multi-modal model that coordinates interactions between different modality stacks."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.cache_names = [name for name in config.multimodal if config.multimodal[name].cache]   

        # init for each modality
        self.modality_stacks = {}
        for modality_name, modality_config in config.multimodal.items():
            modality_config = OmegaConf.merge(config, modality_config)
            self.modality_stacks[modality_name] = ModalStack(modality_config)
            
        # share weights between proprio and action stacks
        self.modality_stacks["proprio"] = self.modality_stacks["action"]

    def forward(
        self,
        attention_mask: jnp.ndarray,
        input_idx: Dict[str, jnp.ndarray],
        input_embeddings: Dict[str, jnp.ndarray],
        kv_cache: Optional[MultiModalKVCache] = None,
        use_cache: bool = True,
        return_cache: bool = False,
    ) -> Tuple[Dict[str, jnp.ndarray], Optional[MultiModalKVCache]]:
        """
        Forward pass through all modality stacks.
        
        Args:
            attention_mask: Attention mask for all modalities
            input_idx: Position IDs for each modality
            input_embeddings: Input embeddings for each modality
            kv_cache: Optional KV cache manager for all modalities
            use_cache: Whether to use and update KV caches
            return_cache: Whether to return updated KV caches
            
        Returns:
            Tuple of (outputs, kv_cache) where outputs contains the final hidden states
            for each modality and kv_cache contains the updated caches if return_cache is True.
        """
        inputs = list(input_embeddings.keys()) # "image_text", "proprio", "action" and each of their embeddings in a dict

        # scale inputs
        for name in inputs:
            hidden_size = input_embeddings[name].shape[-1]
            normalizer = jnp.sqrt(hidden_size)
            input_embeddings[name] *= normalizer

        hidden_states = input_embeddings
        for layer_idx in range(self.num_hidden_layers):
            is_final_layer = layer_idx == self.num_hidden_layers - 1
            hidden_states = moe_layers(
                self.modality_stacks,
                attention_mask,
                input_idx,
                hidden_states,
                layer_idx=layer_idx,
                kv_cache=kv_cache,
                use_cache=use_cache,
                is_final_layer=is_final_layer,
            )

        outputs = {
            name: self.modality_stacks[name].final_norm(hidden_states[name])
            for name in inputs
        }

        if return_cache:
            return outputs, kv_cache
        return outputs, None

