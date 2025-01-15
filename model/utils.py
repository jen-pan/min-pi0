import jax.numpy as jnp


def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    # Build the [-x2, x1, -x4, x3, ...] tensor for the sin part of the positional encoding.
    x1 = x[..., :x.shape[-1] // 2]  # Takes the first half of the last dimension
    x2 = x[..., x.shape[-1] // 2:]  # Takes the second half of the last dimension
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray, unsqueeze_dim: int = 1) -> jnp.ndarray:
    # Add the head dimension using expand_dims instead of unsqueeze
    cos = jnp.expand_dims(cos, axis=unsqueeze_dim)
    sin = jnp.expand_dims(sin, axis=unsqueeze_dim)
    # Apply the formula (34) of the Rotary Positional Encoding paper.
    x = (x * cos) + (rotate_half(x) * sin)
    return x


def repeat_kv(hidden_states: jnp.ndarray, n_rep: int) -> jnp.ndarray:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = jnp.reshape(hidden_states, 
        (batch, num_key_value_heads, 1, slen, head_dim))
    hidden_states = jnp.repeat(hidden_states, n_rep, axis=2)
    return jnp.reshape(hidden_states,
        (batch, num_key_value_heads * n_rep, slen, head_dim))
