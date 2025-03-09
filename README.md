# min-pi0
The models were trained with learning rate 5e-5, global batch size 1024. Input to the model includes single image (256 tokens, no history), max 19 text tokens (so total 256+19=275 tokens), 1 proprio token (no history), and 4 action tokens (chunk size 4). It takes roughly 1.5-2 days on one L40 node (per-GPU bsz 16 and thus gradient accumulation step 8), or 8-12 hours with H100s (bsz 32). torch.compile, bfloat16, and 8-bit optimizer were used to reduce VRAM usage (peak 40GB with bsz 16). Action and proprioception data were normalized in [-1, 1].

Inference involves one forward pass through PaliGemma (utilizing KV cache), and then 10 flow matching steps through the action expert. 
