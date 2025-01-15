import os
from typing import Any, Dict, Tuple, Callable, Union
import einops
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
from absl import logging
import wandb
from omegaconf import OmegaConf
import orbax.checkpoint
from transformers import AutoTokenizer
from model.pi0.pi0 import Pi0
from model.pi0.processing_pi0 import Pi0Processor
from data.dataset import make_interleaved_dataset
from data.oxe import make_oxe_dataset_kwargs_and_weights

class TrainState(train_state.TrainState):
    """custom train state for Pi0 model."""
    model: Pi0
    lr_fn: Callable[[Union[int, jnp.ndarray]], float]

    def get_learning_rate(self) -> float:
        """get current learning rate based on step."""
        return float(self.lr_fn(self.step)) 

    def get_input_idx(self, batch_size: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """get position indices for each modality."""
        return self.model.get_input_idx(batch_size)

    def create_block_attn_mask(self, attention_mask: jnp.ndarray, dtype: jnp.dtype) -> jnp.ndarray:
        """create attention mask for transformer blocks."""
        return self.model.create_block_attn_mask(attention_mask, dtype)

def create_train_state(model: Pi0, config_path: str, rng: jax.Array) -> TrainState:
    """create initial training state with separate optimizers for action and VLM parameters."""
    cfg = OmegaConf.load(config_path)
    # same for action and vlm for now 
    learning_rate_fn = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=float(cfg.lr),
        warmup_steps=int(cfg.lr_scheduler.warmup_steps),
        decay_steps=int(cfg.lr_scheduler.first_cycle_steps),
        end_value=float(cfg.lr_scheduler.min_lr)
    )
    
    # separate optimizers for action and vlm params
    action_tx = optax.chain(
        optax.clip_by_global_norm(float(cfg.max_grad_norm)),
        optax.adamw(
            learning_rate=learning_rate_fn,
            weight_decay=float(cfg.weight_decay)
        )
    )
    
    vlm_tx = optax.chain(
        optax.clip_by_global_norm(float(cfg.max_grad_norm)),
        optax.adamw(
            learning_rate=lambda step: learning_rate_fn(step) * 0.1,
            weight_decay=float(cfg.weight_decay)
        )
    )
    
    tx = optax.multi_transform(
        {'action': action_tx, 'vlm': vlm_tx},
        {'action': model.action_expert_parameters, 'vlm': model.trainable_vlm_parameters}
    )
    dummy_bsz = 2
    dummy_batch = {
        'input_ids': jnp.ones((dummy_bsz, int(cfg.max_seq_len)), dtype=jnp.int32),
        'pixel_values': jnp.ones((dummy_bsz, 3, int(cfg.vision.config.image_size), 
                                int(cfg.vision.config.image_size)), dtype=jnp.float32),
        'attention_mask': jnp.ones((dummy_bsz, int(cfg.max_seq_len)), dtype=jnp.int32),
        'proprio': jnp.ones((dummy_bsz, int(cfg.proprio_dim)), dtype=jnp.float32),
        'action': jnp.ones((dummy_bsz, int(cfg.horizon_steps), int(cfg.action_dim)), dtype=jnp.float32),
    }
    
    variables = model.init(
        rng,
        input_ids=dummy_batch['input_ids'],
        pixel_values=dummy_batch['pixel_values'],
        causal_mask=model.create_block_attn_mask(dummy_batch['attention_mask'], jnp.float32),
        image_text_idx=jnp.ones((dummy_bsz, int(cfg.max_image_text_tokens)), dtype=jnp.int32),
        proprio_idx=jnp.ones((dummy_bsz, model.num_proprio_tokens), dtype=jnp.int32),
        action_idx=jnp.ones((dummy_bsz, model.num_action_tokens), dtype=jnp.int32),
        proprio=dummy_batch['proprio'],
        action=dummy_batch['action'],
        t=jnp.ones((dummy_bsz,), dtype=jnp.float32)
    )
    
    model.freeze_static_weights()
    
    return TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
        model=model,
        lr_fn=learning_rate_fn
    )

def sample_beta_time(rng: jax.Array, batch_size: int, alpha: float = 1.5, beta: float = 1.0, sig_min: float = 0.001) -> jax.Array:
    """sample time from a beta distribution."""
    z = jax.random.beta(rng, alpha, beta, shape=(batch_size,))
    t = (1 - sig_min) * (1 - z)  
    return t

@jax.jit
def train_step(state: TrainState, batch: Dict[str, jnp.ndarray], rng: jax.Array) -> Tuple[TrainState, Dict[str, float]]:
    
    def loss_fn(params):
        image_text_idx, proprio_idx, action_idx = state.get_input_idx(batch['input_ids'].shape[0])
        
        attention_mask = state.create_block_attn_mask(
            batch['attention_mask'],
            dtype=batch['input_ids'].dtype
        )
        
        time_rng, dropout_rng = jax.random.split(rng)
        t = sample_beta_time(time_rng, batch['input_ids'].shape[0])
        
        loss = state.apply_fn(
            {'params': params},
            input_ids=batch['input_ids'],
            pixel_values=batch['pixel_values'],
            causal_mask=attention_mask,
            image_text_idx=image_text_idx,
            proprio_idx=proprio_idx,
            action_idx=action_idx,
            proprio=batch['proprio'],
            action=batch['action'],
            t=t,
            rngs={'dropout': dropout_rng}
        )
        
        return loss, {
            'loss': loss,
            'learning_rate': state.get_learning_rate()
        }
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, metrics), grads = grad_fn(state.params)
    
    state = state.apply_gradients(grads=grads)
    return state, metrics

@jax.jit
def val_step(state: TrainState, batch: Dict[str, jnp.ndarray], rng: jax.Array) -> Dict[str, float]:
    """Performs a single validation step with both flow matching loss and action inference."""
    image_text_idx, proprio_idx, action_idx = state.get_input_idx(batch['input_ids'].shape[0])
    
    attention_mask = state.create_block_attn_mask(
        batch['attention_mask'],
        dtype=batch['input_ids'].dtype
    )
    
    # sample time values for flow matching
    time_rng, dropout_rng = jax.random.split(rng)
    t = sample_beta_time(time_rng, batch['input_ids'].shape[0])
    
    # forward pass
    flow_loss = state.apply_fn(
        {'params': state.params},
        input_ids=batch['input_ids'],
        pixel_values=batch['pixel_values'],
        causal_mask=attention_mask,
        image_text_idx=image_text_idx,
        proprio_idx=proprio_idx,
        action_idx=action_idx,
        proprio=batch['proprio'],
        action=batch['action'],
        t=t,
        rngs={'dropout': dropout_rng}
    )
    
    # sample action
    pred_actions = state.model.apply(
        {'params': state.params},
        method=state.model.sample_action,
        input_ids=batch['input_ids'],
        pixel_values=batch['pixel_values'],
        causal_mask=attention_mask,
        image_text_idx=image_text_idx,
        proprio_idx=proprio_idx,
        action_idx=action_idx,
        proprio=batch['proprio'],
        rngs={'dropout': dropout_rng}
    )
    
    action_mse = jnp.mean((pred_actions - batch['actions']) ** 2)
    
    return {
        'flow_loss': float(flow_loss),
        'action_mse': float(action_mse),
    }

def save_checkpoint(state: TrainState, save_dir: str, step: int, is_best: bool = False) -> None:
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    state_dict = {
        'params': state.params,
        'opt_state': state.opt_state,
        'step': state.step,
    }
    
    ckpt_path = os.path.join(save_dir, f"step_{step}")
    checkpointer.save(ckpt_path, state_dict)
    
    if is_best:
        best_path = os.path.join(save_dir, "best_model")
        checkpointer.save(best_path, state_dict)

def train(config_path: str):
    cfg = OmegaConf.load(config_path)
    
    model = Pi0(cfg)
        
    if cfg.load_pretrained:
        model.load_pretrained()
    
    rng = jax.random.PRNGKey(int(cfg.seed))
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(model, config_path, init_rng)
    
    if cfg.resume_from_checkpoint:
        logging.info(f"Resuming from checkpoint: {cfg.resume_from_checkpoint}")
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        state_dict = checkpointer.restore(cfg.resume_from_checkpoint)
        state = state.replace(
            params=state_dict["params"],
            opt_state=state_dict["opt_state"],
            step=state_dict["step"]
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_path, padding_side="right")
    processor = Pi0Processor(tokenizer, cfg.num_image_tokens, cfg.max_seq_len, cfg.vision.config.image_size)

    def process_batch(batch, train: bool = True) -> Dict[str, jnp.ndarray]:
        texts = [text.decode("utf-8") for text in batch["task"]["language_instruction"]]
        images = batch["observation"]["image_primary"]
        images = einops.rearrange(images, "B T H W C -> B (T C) H W")
        inputs = processor(text=texts, images=images)
        proprio = batch["observation"]["proprio"]
        action = batch["action"].squeeze(1) 
    
        causal_mask, image_text_idx, proprio_idx, action_idx = (
            model.build_causal_mask_and_position_ids(inputs["attention_mask"])
        )
        inputs = {
            "input_ids": inputs["input_ids"],
            "pixel_values": inputs["pixel_values"],
            "image_text_idx": image_text_idx,
            "proprio_idx": proprio_idx,
            "action_idx": action_idx,
            "proprio": proprio,
            "action": action,
        }

        if not train:
            image_text_proprio_mask, action_mask = (
                model.split_full_mask_into_submasks(causal_mask)
            )
            inputs["image_text_proprio_mask"] = image_text_proprio_mask
            inputs["action_mask"] = action_mask

        if train:
            # sample flow matching timesteps only during training
            inputs["t"] = sample_beta_time(rng, len(texts))
            inputs["causal_mask"] = causal_mask

        return inputs
    
    oxe_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
        cfg.data.train.dataset_mix,
        cfg.data.train.data_path,
        load_proprio=bool(cfg.data.train.load_proprio),
        load_camera_views=("primary",),
    )
    
    train_data = make_interleaved_dataset(
        oxe_kwargs_list,
        sample_weights,
        train=True,
        split=cfg.data.train.split,
        shuffle_buffer_size=int(cfg.data.train.shuffle_buffer_size),
        batch_size=int(cfg.per_device_batch_size),
        balance_weights=True,
        traj_transform_kwargs=dict(
            window_size=int(cfg.data.train.window_size),
            action_horizon=int(cfg.data.train.action_horizon),
            subsample_length=100,
            skip_unlabeled=bool(cfg.data.train.skip_unlabeled),
        ),
        frame_transform_kwargs=dict(
            image_augment_kwargs={
                "primary": dict(
                    random_resized_crop=dict(
                        scale=[0.8, 1.0],
                        ratio=[0.9, 1.1],
                    ),
                    random_brightness=[0.1],
                    random_contrast=[0.9, 1.1],
                    random_saturation=[0.9, 1.1],
                    random_hue=[0.05],
                ),
            },
            resize_size=dict(primary=(224, 224)),
            num_parallel_calls=int(cfg.data.train.num_parallel_calls),
        ),
        traj_transform_threads=int(cfg.data.train.traj_transform_threads),
        traj_read_threads=int(cfg.data.train.traj_read_threads),
    )
    train_data_iter = map(process_batch, train_data.iterator())
    
    val_data = make_interleaved_dataset(
        oxe_kwargs_list,
        sample_weights,
        train=False,  # no augmentation for validation
        split=cfg.data.val.split,
        shuffle_buffer_size=1,  # no shuffling for validation
        batch_size=int(cfg.per_device_batch_size),
        balance_weights=True,
        traj_transform_kwargs=dict(
            window_size=int(cfg.data.val.window_size),
            action_horizon=int(cfg.data.val.action_horizon),
            subsample_length=100,
            skip_unlabeled=bool(cfg.data.val.skip_unlabeled),
        ),
        frame_transform_kwargs=dict(
            image_augment_kwargs={},  # no augmentation for validation
            resize_size=dict(primary=(224, 224)),
            num_parallel_calls=int(cfg.data.val.num_parallel_calls),
        ),
        traj_transform_threads=int(cfg.data.val.traj_transform_threads),
        traj_read_threads=int(cfg.data.val.traj_read_threads),
    )
    val_data_iter = map(process_batch, val_data.iterator())

    if jax.process_index() == 0:
        wandb.init(
            project=str(cfg.wandb.project),
            entity=str(cfg.wandb.entity),
            config=str(OmegaConf.to_container(cfg, resolve=True)),
            resume=bool(cfg.resume_from_checkpoint),
            id=cfg.wandb_run_id
        )
    
    os.makedirs(str(cfg.log_dir), exist_ok=True)
    
    step = int(state.step)
    best_val_loss = float('inf')
    best_action_mse = float('inf')
    val_metrics_history = []
    
    for epoch in range(int(cfg.n_epochs)): # TODO: check epochs wrt RLDS streaming dataset
        batch = next(train_data_iter)
        inputs = process_batch(batch)
        rng, step_rng = jax.random.split(rng)
        
        state, train_metrics = train_step(state, inputs, step_rng)
        
        if step % int(cfg.log_freq) == 0 and jax.process_index() == 0:
            wandb.log({
                'train/loss': float(train_metrics['loss']),
                'train/learning_rate': float(train_metrics['learning_rate']),
                'train/epoch': epoch,
                'train/step': step,
            })
        
        if step % int(cfg.val_freq) == 0:
            val_metrics_sum = {
                'flow_loss': 0.0,
                'action_mse': 0.0,
            }
            val_count = 0
            
            val_batch = next(val_data_iter)
            inputs = process_batch(val_batch)
            rng, val_rng = jax.random.split(rng)
            batch_metrics = val_step(state, inputs, val_rng)
            
            for k, v in batch_metrics.items():
                val_metrics_sum[k] += float(v)
            val_count += 1
            
            avg_metrics = {
                k: v / val_count for k, v in val_metrics_sum.items()
            }
            val_metrics_history.append(avg_metrics)
            
            if jax.process_index() == 0:
                wandb.log({
                    'val/flow_loss': avg_metrics['flow_loss'],
                    'val/action_mse': avg_metrics['action_mse'],
                    'val/epoch': epoch,
                    'val/step': step,
                })
            
            if avg_metrics['flow_loss'] < best_val_loss:
                best_val_loss = avg_metrics['flow_loss']
                if jax.process_index() == 0:
                    save_checkpoint(state, str(cfg.log_dir), int(step), is_best=True)
                    logging.info(f"New best validation flow loss: {best_val_loss:.4f}")
            
            if avg_metrics['action_mse'] < best_action_mse:
                best_action_mse = avg_metrics['action_mse']
                if jax.process_index() == 0:
                    logging.info(f"New best action MSE: {best_action_mse:.4f}")
        
            if step % int(cfg.save_model_freq) == 0 and jax.process_index() == 0:
                save_checkpoint(state, str(cfg.log_dir), int(step))
            
            step += 1
            if step >= int(cfg.n_updates):
                break
        
        if step >= int(cfg.n_updates):
            break
    
    if jax.process_index() == 0:
        save_checkpoint(state, str(cfg.log_dir), int(step))
        wandb.finish()
    
    logging.info(f"Best validation flow loss: {best_val_loss:.4f}")
    logging.info(f"Best action MSE: {best_action_mse:.4f}")

if __name__ == "__main__":
    train("train_oxe_simple.yaml")
