# src/detokenizer.py

import torch
from flextok.utils.misc import get_bf16_context, get_generator
from flextok.utils.demo import batch_to_pil
from .tokenizer import load_flextok_model

def detokenize_tokens(
    tokens_list,
    timesteps: int = 25,
    guidance_scale: float = 7.5,
    perform_norm_guidance: bool = True,
    seed: int = 0,
    model_name: str = 'EPFL-VILAB/flextok_d18_d28_dfn'
):
    """
    Detokenize each token sequence in `tokens_list` back to images via FlexTok.

    Args:
        tokens_list:          List of torch.Tensor of shape [256] or [1, 256].
        timesteps:            Number of diffusion denoising steps.
        guidance_scale:       Classifier-free guidance scale.
        perform_norm_guidance: Whether to apply Adaptive Projected Guidance.
        seed:                 RNG seed for reproducibility.
        model_name:           HuggingFace model identifier.

    Returns:
        pil_images: List of PIL.Image reconstructed from tokens.
    """
    model, device, bf16 = load_flextok_model(model_name)
    reconst_tensors = []

    with get_bf16_context(bf16):
        for tok in tokens_list:
            t = tok.unsqueeze(0) if tok.dim() == 1 else tok   # ensure [1, 256]
            t = t.to(device)
            recon = model.detokenize(
                t,
                timesteps=timesteps,
                guidance_scale=guidance_scale,
                perform_norm_guidance=perform_norm_guidance,
                generator=get_generator(seed, device),
                verbose=False
            )
            reconst_tensors.append(recon[0].cpu())           # [3, H, W]

    batch = torch.stack(reconst_tensors, dim=0)             # [B, 3, H, W]
    pil_images = batch_to_pil(batch)
    return pil_images
