from typing import Optional, Tuple
import glob
import os
from omegaconf import OmegaConf
from safetensors import safe_open
import jax
import jax.numpy as jnp
import flax.linen as nn
import logging

from model.pi0.moe import MoE
from model.paligemma.modeling_siglip import SiglipVisionModel, PaliGemmaMultiModalProjector
from model.pi0.modules import ActionEmbedding, SinPosEmb

class Pi0(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.image_token_index = config.image_token_index

        self.max_image_text_tokens = config.max_image_text_tokens
        self.num_proprio_tokens = config.num_proprio_tokens
        self.num_action_tokens = config.num_action_tokens
        self.total_input_tokens = self.max_image_text_tokens + self.num_proprio_tokens + self.num_action_tokens
        self.num_inference_steps = config.num_inference_steps  
        
        self.image_text_hidden_size = config.multimodal.image_text.hidden_size
        self.proprio_hidden_size = config.multimodal.proprio.hidden_size
        self.action_hidden_size = config.multimodal.action.hidden_size
        self.action_dim = config.action_dim
        self.proprio_dim = config.proprio_dim
        self.action_clip_value = config.action_clip_value
        self.flow_sig_min = config.flow_sig_min

        # vision
        vision_config = OmegaConf.to_container(config.siglip_model, resolve=True)
        self.siglip = SiglipVisionModel(*vision_config)
        projector_config = OmegaConf.to_container(config.multimodal_projector, resolve=True)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(*projector_config)

        # language
        self.text_embedding = nn.Embed(config.vocab_size, self.image_text_hidden_size, self.pad_token_id)

        # robot-specific
        self.action_embedding = ActionEmbedding(self.action_hidden_size)
        self.time_embedding = SinPosEmb(self.action_hidden_size)
        self.proprio_embedding = nn.Dense(self.proprio_hidden_size)
        self.action_decoder = nn.Dense(self.action_dim)

        moe_config = OmegaConf.to_container(config.moe, resolve=True)
        self.moe = MoE(*moe_config)
        self.moe.modality_stacks["proprio"] = self.moe.modality_stacks["action"]

    def load_pretrained(self): 
        """Loads pretrained weights from PaLiGemma for vision, projector, and language model components into
        our model architecture."""
        
        tensors = {}
        safetensors_path = os.path.join(self.config.pretrained_model_path, "*.safetensors")
        for filepath in glob.glob(safetensors_path):
            f = None
            try:
                from safetensors.flax import load_file
                tensors.update(load_file(filepath))
            except Exception as e:
                logging.error(f"Failed to load safetensors file {filepath}: {e}")
                raise
            finally:
                if f is not None:
                    f.close()

        components = {
            'embed_tokens': {
                'model': self.text_embedding,
                'prefix': 'language_model.model.embed_tokens.',
            },
            'vision_tower': {
                'model': self.siglip,
                'prefix': 'vision_tower.',
            },
            'multi_modal_projector': {
                'model': self.multi_modal_projector,
                'prefix': 'multi_modal_projector.',
            }
        }

        for key, config in components.items():
            state_dict = config['model'].state_dict()
            for k, v in tensors.items():
                if key in k:
                    new_key = k.replace(config['prefix'], '')
                    state_dict[new_key] = v
            config['model'].load_state_dict(state_dict, strict=True)

        moe_state_dict = self.moe.state_dict()
        moe_state_dict = {
            k: v for k, v in moe_state_dict.items() 
            if 'lora_' not in k
        } # preserves existing lora weights and prevents them from being overwritten by the pretrained weights
        
        for k, v in tensors.items():
            if 'language_model.model' in k:
                new_key = k.replace('language_model.model.', 'multimodal.image_text.') #TODO: fix config
                moe_state_dict[new_key] = v
                
        self.moe.load_state_dict(moe_state_dict, strict=False)

    def freeze_static_weights(self):
        """text embedding and part of last layer of vlm, including lora"""
        self.text_embedding.weight.requires_grad = False
        # TODO: can freeze last layer post-attn params for image_text, minimal memory savings?

    @property
    def action_expert_parameters(self):
        # action and proprio share weights, so don't add modality_stacks['action']
        return (list(self.action_embedding.parameters()) + list(self.proprio_embedding.parameters())
            + list(self.moe.modality_stacks["proprio"].parameters()) + list(self.action_decoder.parameters()))  

    @property
    def trainable_vlm_parameters(self):
        # don't train text embedding, do train vision encoder (openVLA showed it was better to not freeze)
        return (list(self.siglip.parameters()) + list(self.multi_modal_projector.parameters())
            + [param for _, param in self.moe.modality_stacks["image_text"].named_parameters()]
        ) 

    ############################################################
    # Setup inputs
    ############################################################
    def get_input_idx(self, bsz: int):
        """position ids for each blocks that get passed into rope embedding"""
        image_text_idx = jnp.tile(jnp.arange(1, self.max_image_text_tokens + 1)[None, :],(bsz,))
        proprio_idx = jnp.tile(jnp.arange(1, self.num_proprio_tokens + 1)[None, :],(bsz,))
        action_idx = jnp.tile(
            jnp.arange(
                self.num_proprio_tokens + 1, # start where proprio ends because they share weights
                self.num_proprio_tokens + self.num_action_tokens + 1,
            )[None, :],
            (bsz,)
        )
        return image_text_idx, proprio_idx, action_idx
    
    def create_block_attn_mask(self, attention_mask: jnp.ndarray, dtype: jnp.dtype) -> jnp.ndarray:
        """Creates a block attention mask where:
        1. Image/text tokens can attend to themselves
        2. Proprio tokens can attend to image/text and themselves
        3. Action tokens can attend to everything (image/text, proprio, and themselves)
        """
        bsz = attention_mask.shape[0]
        
        # segment boundaries
        segments = {
            'image_text': slice(0, self.max_image_text_tokens),
            'proprio': slice(self.max_image_text_tokens, self.max_image_text_tokens + self.num_proprio_tokens),
            'action': slice(self.max_image_text_tokens + self.num_proprio_tokens, None)
        }
        
        # initialize mask with negative infinity
        causal_mask = jnp.full(
            (bsz, self.total_input_tokens, self.total_input_tokens),
            jnp.finfo(dtype).min,
            dtype=dtype
        )
        
        # length of image/text sequence for each batch, use original attention mask to see padding
        image_text_lengths = jnp.sum(attention_mask, axis=1)
        
        for idx, length in enumerate(image_text_lengths):
            # 1. image/text self-attention
            causal_mask = causal_mask.at[idx, :length, :length].set(0)
            # 2. proprio attending to image/text and itself
            causal_mask = causal_mask.at[
                idx,
                segments['proprio'].start:,
                :segments['proprio'].stop
            ].set(0)
            # 3. action attending to everything
            causal_mask = causal_mask.at[
                idx,
                segments['action'].start:,
                :
            ].set(0)
        
        # add head dimension for multi-head attention
        causal_mask = jnp.expand_dims(causal_mask, axis=1)
        return causal_mask

    def embed_vision_and_text(
        self,
        input_text: jnp.ndarray,
        input_image: jnp.ndarray,
    ) -> jnp.ndarray:
        bsz, seq_len = input_text.shape
        # embed text to embed_dim
        text_embedding = self.text_embedding(input_text)
        # image features from siglip and projector
        image_features = self.siglip(input_image)
        image_embedding = self.multi_modal_projector(image_features)
        norm_image_embedding = image_embedding / (self.image_text_hidden_size**0.5)

        embed_dim = image_embedding.shape[-1]
        input_embedding = jnp.full((bsz, seq_len, embed_dim), self.pad_token_id, dtype=input_image.dtype)

        text_mask = (input_text != self.image_token_index) & (input_text != self.pad_token_id)
        image_mask = (input_text == self.image_token_index)
        input_embedding = input_embedding.at[text_mask].set(text_embedding[text_mask])
        
        def apply_image_mask(batch_mask, batch_embedding, batch_features):
            indices = jnp.where(batch_mask)[0]
            return batch_embedding.at[indices].set(batch_features[:len(indices)])
        
        input_embedding = jax.vmap(apply_image_mask)(
            image_mask,
            input_embedding,
            norm_image_embedding
        )
        return input_embedding

    ############################################################
    # Inference
    ############################################################
    def split_causal_mask(
        self, causal_mask: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """split into one mask for image/text/proprio and one for action"""
        image_text_proprio_mask = causal_mask[..., 
                                              : self.max_image_text_tokens + self.num_proprio_tokens,
                                              : self.max_image_text_tokens + self.num_proprio_tokens,
                                            ]
        action_mask = causal_mask[..., -self.num_action_tokens :, -self.num_action_tokens :]
        return image_text_proprio_mask, action_mask


    def sample_action(
        self,
        input_ids: jnp.ndarray,
        pixel_values: jnp.ndarray,
        causal_mask: jnp.ndarray,
        image_text_position_ids: jnp.ndarray,
        proprio_position_ids: jnp.ndarray,
        action_position_ids: jnp.ndarray,
        proprios: jnp.ndarray,
    ) -> jnp.ndarray:
        image_text_proprio_mask, action_mask = self.split_causal_mask(causal_mask)
        inputs_embeds = self.embed_vision_and_text(input_ids, pixel_values)
        proprio_embeds = self.proprio_embedding(proprios)
        # forward pass thru image_text and proprio, cache the kv
        _, kv_caches = self.moe(
            attention_mask=image_text_proprio_mask,
            input_idx={"image_text": image_text_position_ids, "proprio": proprio_position_ids},
            input_embeddings={"image_text": inputs_embeds, "proprio": proprio_embeds},
            return_caches=True,
        )
        # sample action noise
        key = jax.random.PRNGKey(0)
        action = jax.random.normal(
            key,
            (inputs_embeds.shape[0], self.num_action_tokens, self.action_dim),
            dtype=inputs_embeds.dtype
        )

       # euler integration using kv caches of image_text and proprio
        delta_t = 1.0 / self.num_inference_steps
        t = jnp.zeros((inputs_embeds.shape[0],), dtype=inputs_embeds.dtype)
        
        def euler_integration_step(carry, _):
            action, t, kv_caches = carry
            time_cond = self.time_embedding(t)
            action_embeds = self.action_embedding(action, time_cond)
            action_embeds = self.moe(
                attention_mask=action_mask,
                input_idx={"action": action_position_ids},
                input_embeddings={"action": action_embeds}, # only action tokens
                time_cond=time_cond,
                kv_caches=kv_caches,
                use_cache=True,  # use caches from image_text and proprio
            )["action"]
            action_vel = self.action_decoder(action_embeds)
            action = action + delta_t * action_vel # euler integration
            t = t + delta_t
            return (action, t, kv_caches), None
            
        (action, _, _), _ = jax.lax.scan(
            euler_integration_step,
            (action, t, kv_caches),
            None,
            length=self.num_inference_steps
        )

        if self.action_clip_value:
            action = jnp.clip(action, -self.action_clip_value, self.action_clip_value)
        return action


    ############################################################
    # Flow matching training
    ############################################################
    def time_linear_interpolation(
        self,
        p: jnp.ndarray,
        q: jnp.ndarray,
        t: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Flow matching training using: 
        https://github.com/gle-bellier/flow-matching/blob/main/Flow_Matching.ipynb
        """ 
        t = jnp.expand_dims(t, axis=(1, 2)) 
        return (1 - (1 - self.flow_sig_min) * t) * p + t * q

    def forward(
        self,
        input_ids: jnp.ndarray,
        pixel_values: jnp.ndarray,
        causal_mask: jnp.ndarray,
        image_text_position_ids: jnp.ndarray,
        proprio_position_ids: jnp.ndarray,
        action_position_ids: jnp.ndarray,
        proprios: jnp.ndarray,
        actions: jnp.ndarray,
        t: jnp.ndarray,
    ) -> jnp.ndarray:
        """Training forward pass implementing flow matching loss."""
        # [batch_size, action_tokens, action_dim]
        key = jax.random.PRNGKey(42)
        p = jax.random.normal(key, actions.shape, dtype=actions.dtype)
        q = actions
        # create noisy action between p and q with chosen time t
        psi_t = self.time_linear_interpolation(p, q, t)  

        input_embedding = self.embed_vision_and_text(input_ids, pixel_values)
        proprio_embedding = self.proprio_embedding(proprios)
        time_embedding = self.time_embedding(t)
        action_embedding = self.action_embedding(psi_t, time_embedding)
        action_embedding = self.moe(
            attention_mask=causal_mask,
            input_idx={
                "image_text": image_text_position_ids,
                "proprio": proprio_position_ids,
                "action": action_position_ids,
            },
            input_embeddings={
                "image_text": input_embedding,
                "proprio": proprio_embedding,
                "action": action_embedding,
            },
            time_cond=time_embedding,
            kv_caches={},  # no caching needed during training
        )["action"]

        # [batch_size, action_tokens, action_dim]
        predicted_action_vel = self.action_decoder(action_embedding)
        gt_action_vel = q - (1 - self.flow_sig_min) * p
        return jnp.mean((predicted_action_vel - gt_action_vel) ** 2)