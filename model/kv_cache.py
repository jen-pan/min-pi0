"""
KV cache management for multi-modal transformer models.
"""
from typing import Dict, NamedTuple, Optional, Tuple
from dataclasses import dataclass
import jax.numpy as jnp

@dataclass(frozen=True)
class KVCache:
    """Immutable key-value cache for attention layers.
    
    Attributes:
        key_cache: Tuple of key arrays for each layer
        value_cache: Tuple of value arrays for each layer
    """
    key_cache: Tuple[jnp.ndarray, ...] = ()
    value_cache: Tuple[jnp.ndarray, ...] = ()

    @property
    def num_items(self) -> int:
        """Returns number of cached sequence items."""
        if not self.key_cache:
            return 0
        # [bs, num_heads_kv, seq_len, head_dim]
        return self.key_cache[0].shape[2]
    
    def has_item(self, layer_idx: int) -> bool:
        """Checks if cache exists for given layer."""
        return len(self.key_cache) > layer_idx

    def get(self, layer_idx: int) -> Optional[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Gets cached key-value pair for given layer if it exists."""
        if not self.has_item(layer_idx):
            return None
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    def update(
        self,
        key_states: jnp.ndarray,
        value_states: jnp.ndarray,
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
                [new_key_cache[layer_idx], key_states], axis=2
            )
            new_value_cache[layer_idx] = jnp.concatenate(
                [new_value_cache[layer_idx], value_states], axis=2
            )

        return KVCache(
            key_cache=tuple(new_key_cache),
            value_cache=tuple(new_value_cache)
        )

    @classmethod
    def create_empty(cls) -> 'KVCache':
        """Creates an empty KV cache."""
        return cls()

class MultiModalKVCacheState(NamedTuple):
    """Immutable state container for multi-modal KV caches."""
    caches: Dict[str, KVCache]

class MultiModalKVCache:
    """Manages KV caches for all modalities in a multi-modal transformer.
    """
    
    @staticmethod
    def create_empty(modality_names: Tuple[str, ...]) -> MultiModalKVCacheState:
        """Creates empty caches for specified modalities.
        
        Args:
            modality_names: Tuple of modality names to create caches for
            
        Returns:
            New MultiModalKVCacheState with empty caches
        """
        return MultiModalKVCacheState(caches={
            name: KVCache.create_empty() 
            for name in modality_names
        })
    
    @staticmethod
    def update(
        state: MultiModalKVCacheState,
        modality: str,
        layer_idx: int,
        key_states: jnp.ndarray,
        value_states: jnp.ndarray
    ) -> MultiModalKVCacheState:
        """Returns new state with updated cache for a specific modality and layer.
        
        Args:
            state: Current cache state
            modality: Name of modality to update
            layer_idx: Index of layer to update
            key_states: New key states to add
            value_states: New value states to add
            
        Returns:
            New MultiModalKVCacheState with updated cache
        """
        if modality not in state.caches:
            return state
            
        new_caches = dict(state.caches)
        new_caches[modality] = new_caches[modality].update(
            key_states, value_states, layer_idx
        )
        return MultiModalKVCacheState(caches=new_caches)
    
    @staticmethod
    def get_layer_cache(
        state: MultiModalKVCacheState,
        modality: str,
        layer_idx: int
    ) -> Optional[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Gets cached KV states for a specific modality and layer.
        
        Args:
            state: Current cache state
            modality: Name of modality to get cache for
            layer_idx: Index of layer to get cache for
            
        Returns:
            Tuple of (key_states, value_states) if cache exists, None otherwise
        """
        if modality not in state.caches:
            return None
        return state.caches[modality].get(layer_idx)
    
    @staticmethod
    def has_cache(
        state: MultiModalKVCacheState,
        modality: str,
        layer_idx: int
    ) -> bool:
        """Checks if cache exists for a specific modality and layer."""
        return (
            modality in state.caches and 
            state.caches[modality].has_item(layer_idx)
        )
