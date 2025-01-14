"""
Mixture of experts implementation for pi0 model.
"""
from typing import Tuple, Optional, List
import flax.linen as nn
import jax.numpy as jnp
from model.paligemma.modeling_gemma import (
    GemmaMLP,
    GemmaRMSNorm,
    GemmaRotaryEmbedding,
)
from model.utils import apply_rotary_pos_emb, repeat_kv

class SinPosEmb(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        half_features = self.features // 2
        emb = jnp.log(10000) / (half_features - 1)
        emb = jnp.exp(jnp.arange(half_features) * -emb)
        emb = x * emb
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)
        return emb
    
class ActionEmbedding(nn.Module):
    """Encodes action chunks using a 3-layer MLP with flow matching timestep information."""
    width: int

    @nn.compact
    def __call__(
        self,
        action: jnp.ndarray,  # [batch_size, seq_len, action_dim]
        time_emb: jnp.ndarray,  # [batch_size, time_emb_dim]
    ) -> jnp.ndarray:
        emb = nn.Dense(self.width)(action)
        time_emb_expanded = jnp.expand_dims(time_emb, axis=1).repeat(action.shape[1], axis=1)
        emb = jnp.concatenate([time_emb_expanded, emb], axis=-1)
        emb = nn.swish(nn.Dense(self.width)(emb))
        emb = nn.Dense(self.width)(emb)
        return emb

class MultiModalAttention(nn.Module):
    """Self-attention module that handles multiple modalities"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        assert config.hidden_size % self.num_heads == 0

        self.k_proj = nn.Dense(config.hidden_size, self.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Dense(config.hidden_size, self.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Dense(self.num_heads * self.head_dim, config.hidden_size)
        
        self.rope_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        position_ids: jnp.ndarray,
        kv_cache: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    ) -> Tuple[jnp.ndarray, Optional[Tuple[jnp.ndarray, jnp.ndarray]]]:
        bsz, seq_len = hidden_states.shape[:2]
        
        # get positional embeddings
        cos, sin = self.rope_emb(hidden_states, position_ids)
        
        # project and reshape Q, K, V
        query_states = self._project_and_reshape(self.q_proj, hidden_states, bsz, seq_len, self.num_heads)
        key_states = self._project_and_reshape(self.k_proj, hidden_states, bsz, seq_len, self.num_key_value_heads)
        value_states = self._project_and_reshape(self.v_proj, hidden_states, bsz, seq_len, self.num_key_value_heads)
        
        # apply rotary embeddings
        query_states = apply_rotary_pos_emb(query_states, cos, sin)
        key_states = apply_rotary_pos_emb(key_states, cos, sin)
        
        # use cached KV states if provided
        if kv_cache is not None:
            cached_key, cached_value = kv_cache
            key_states = jnp.concatenate([cached_key, key_states], axis=2)
            value_states = jnp.concatenate([cached_value, value_states], axis=2)
            
        # store current KV states for caching
        current_kv = (key_states, value_states)
            
        # repeat KV states for multi-query attention
        key_states, value_states = self._repeat_kv(key_states, value_states)
        
        attn_weights = jnp.matmul(query_states, key_states.transpose(0, 1, 3, 2))
        attn_weights = attn_weights / jnp.sqrt(self.head_dim)
        attn_weights = jnp.clip(attn_weights, -self.config.attn_clamp, self.config.attn_clamp)
        attn_weights = nn.softmax(attn_weights + attention_mask, axis=-1)
        attn_output = jnp.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(bsz, seq_len, -1)
        
        return self.o_proj(attn_output), current_kv
    
    def _project_and_reshape(self, proj_layer, x, bsz, seq_len, num_heads):
        x = proj_layer(x)
        return x.reshape(bsz, seq_len, num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def _repeat_kv(self, key_states, value_states):
        """Repeat key/value states to match number of query heads."""
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        return key_states, value_states

class ModalBlock(nn.Module):
    """A transformer block that processes a single modality."""
    
    def __init__(self, config):
        super().__init__()
        self.input_ln = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = MultiModalAttention(config)
        self.post_attn_ln = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = GemmaMLP(config)
        
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        position_ids: jnp.ndarray,
        kv_cache: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    ) -> Tuple[jnp.ndarray, Optional[Tuple[jnp.ndarray, jnp.ndarray]]]:
        normed_states = self.input_ln(hidden_states)
        attention_output, new_kv = self.attention(
            normed_states,
            attention_mask,
            position_ids,
            kv_cache
        )
        hidden_states = hidden_states + attention_output
        
        normed_states = self.post_attn_ln(hidden_states)
        mlp_output = self.mlp(normed_states)
        hidden_states = hidden_states + mlp_output
        
        return hidden_states, new_kv

class ModalStack(nn.Module):
    """A stack of transformer layers that processes a specific modality (vision/text, or proprio/action)."""
    
    def __init__(self, config):
        super().__init__()
        self.layers = [ModalBlock(config) for _ in range(config.num_hidden_layers)]
        self.final_norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        position_ids: jnp.ndarray,
        kv_caches: Optional[List[Tuple[jnp.ndarray, jnp.ndarray]]] = None,
    ) -> Tuple[jnp.ndarray, Optional[List[Tuple[jnp.ndarray, jnp.ndarray]]]]:
        new_kv_caches = [] if kv_caches is not None else None
        
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            hidden_states, new_kv = layer(hidden_states, attention_mask, position_ids, cache)
            if new_kv_caches is not None:
                new_kv_caches.append(new_kv)
                
        return self.final_norm(hidden_states), new_kv_caches



