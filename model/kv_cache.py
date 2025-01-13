from typing import Tuple, Optional
from dataclasses import dataclass
import jax.numpy as jnp
from jax import Array

@dataclass(frozen=True)
class KVCache:
    """Immutable key-value cache for attention layers.
    
    Attributes:
        key_cache: Tuple of key arrays for each layer
        value_cache: Tuple of value arrays for each layer
    """
    key_cache: Tuple[Array, ...] = ()
    value_cache: Tuple[Array, ...] = ()

    @property
    def num_items(self) -> int:
        """Returns number of cached sequence items."""
        if not self.key_cache:
            return 0
        # [bs, num_heads_kv, seq_len, head_dim]
        return self.key_cache[0].shape[-2]

    def has_item(self, layer_idx: int) -> bool:
        """Checks if cache exists for given layer."""
        return len(self.key_cache) > layer_idx

    def get(self, layer_idx: int) -> Tuple[Array, Array] | None:
        """Gets cached key-value pair for given layer if it exists."""
        if not self.has_item(layer_idx):
            return None
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update(
        self,
        key_states: Array,
        value_states: Array,
        layer_idx: int,
    ) -> 'KVCache':
        """Returns a new KVCache with updated states for the given layer.
        
        Args:
            key_states: New key states to add [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            value_states: New value states to add [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            layer_idx: Index of the layer to update
            
        Returns:
            A new KVCache instance with updated states
        """
        new_key_cache = list(self.key_cache)
        new_value_cache = list(self.value_cache)

        while len(new_key_cache) <= layer_idx:
            new_key_cache.append(key_states)
            new_value_cache.append(value_states)

        if layer_idx < len(self.key_cache):
            new_key_cache[layer_idx] = jnp.concatenate(
                [new_key_cache[layer_idx], key_states], axis=-2
            )
            new_value_cache[layer_idx] = jnp.concatenate(
                [new_value_cache[layer_idx], value_states], axis=-2
            )

        return KVCache(
            key_cache=tuple(new_key_cache),
            value_cache=tuple(new_value_cache)
        )

    @classmethod
    def create_empty(cls) -> 'KVCache':
        """Creates an empty KV cache."""
        return cls()
