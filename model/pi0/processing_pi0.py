'''
Follows format of model.paligemma.processing_paligemma
'''
from typing import List

import jax.numpy as jnp

from model.paligemma.processing_paligemma import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, PaliGemmaProcessor, add_image_tokens_to_prompt

RESCALE_FACTOR = 1 / 255.0

def process_images(
    images: jnp.ndarray,
    rescale_factor: float,
    image_mean: jnp.ndarray,
    image_std: jnp.ndarray,
) -> jnp.ndarray:
    # Rescale the pixel values to be in the range [0, 1]
    images = images * rescale_factor
    assert images.ndim == 4, f"Expected 4D tensor, got {images.ndim}D tensor."
    assert (images.shape[1] == 3), f"Expected 3 channels at axis 1, got {images.shape[1]} channels."
    # Normalize the images to have mean 0 and standard deviation 1
    mean = image_mean[None, :, None, None]  # add batch and spatial dimensions
    std = image_std[None, :, None, None]
    images = (images - mean) / std
    return images

class Pi0Processor(PaliGemmaProcessor):

    def __init__(
        self,
        tokenizer,
        num_image_tokens: int,
        max_seq_len: int,
        image_size: int,
    ):
        super().__init__(tokenizer=tokenizer, num_image_tokens=num_image_tokens, image_size=image_size)
        self.max_seq_len = max_seq_len

    def __call__(
        self,
        text: List[str],
        images: jnp.ndarray,
        padding: str = "max_length",
        truncation: bool = True,
    ) -> dict:
        assert (
            len(images) == 1 and len(text) == 1
        ), f"Received {len(images)} images for {len(text)} prompts."
        #TODO: check this assertion, can have text-only or image-only prompts

        self.tokenizer_padding = padding
        pixel_values = process_images(
            images,
            rescale_factor=RESCALE_FACTOR,
            image_mean=jnp.array(IMAGENET_STANDARD_MEAN),
            image_std=jnp.array(IMAGENET_STANDARD_STD),
        )

        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        inputs = self.tokenizer(
            input_strings,
            return_tensors="jax",
            max_length=self.max_seq_len,
            padding=self.tokenizer_padding,
            truncation=truncation,
        )
        output = {"pixel_values": pixel_values, **inputs}
        return output
