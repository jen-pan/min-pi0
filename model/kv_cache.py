"""
KV cache management for multi-modal transformer models.
"""
from typing import Dict, List, Optional, Tuple
import jax.numpy as jnp

class KVCache:
    """Manages key-value caches for a single modality."""
    
    def __init__(self):
        self.cached_kvs: List[Optional[Tuple[jnp.ndarray, jnp.ndarray]]] = []
        
    def update(self, key_states: jnp.ndarray, value_states: jnp.ndarray, layer_idx: int):
        """Update cache for a specific layer."""
        while len(self.cached_kvs) <= layer_idx:
            self.cached_kvs.append(None)
        self.cached_kvs[layer_idx] = (key_states, value_states)
    
    def get(self, layer_idx: int) -> Optional[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Get cached KV states for a layer if they exist."""
        if layer_idx < len(self.cached_kvs):
            return self.cached_kvs[layer_idx]
        return None
    
    def has_item(self, layer_idx: int) -> bool:
        """Check if cache exists for a layer."""
        return layer_idx < len(self.cached_kvs) and self.cached_kvs[layer_idx] is not None
    
    @classmethod
    def create_empty(cls) -> 'KVCache':
        """Create an empty cache."""
        return cls()

class MultiModalKVCache:
    """Manages KV caches for all modalities in a multi-modal transformer."""
    
    def __init__(self, modality_names: List[str]):
        """Initialize caches for specified modalities."""
        self.caches: Dict[str, KVCache] = {
            name: KVCache.create_empty() 
            for name in modality_names
        }
    
    def update(
        self,
        modality: str,
        layer_idx: int,
        key_states: jnp.ndarray,
        value_states: jnp.ndarray
    ):
        """Update cache for a specific modality and layer."""
        if modality in self.caches:
            self.caches[modality].update(key_states, value_states, layer_idx)
    
    def get_layer_cache(
        self,
        modality: str,
        layer_idx: int
    ) -> Optional[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Get cached KV states for a specific modality and layer."""
        if modality in self.caches:
            return self.caches[modality].get(layer_idx)
        return None
    
    def has_cache(self, modality: str, layer_idx: int) -> bool:
        """Check if cache exists for a specific modality and layer."""
        return (
            modality in self.caches and 
            self.caches[modality].has_item(layer_idx)
        )
    
    def get_modality_cache(self, modality: str) -> Optional[KVCache]:
        """Get entire cache for a modality."""
        return self.caches.get(modality)
    
    def get_all_caches(self) -> Dict[str, KVCache]:
        """Get all caches."""
        return self.caches
    
    @classmethod
    def create_empty(cls, modality_names: List[str]) -> 'MultiModalKVCache':
        """Create empty caches for specified modalities."""
        return cls(modality_names)
