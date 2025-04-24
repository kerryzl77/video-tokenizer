# src/tokenizer.py

import torch
from flextok.flextok_wrapper import FlexTokFromHub
from flextok.utils.misc import detect_bf16_support, get_bf16_context
from .data_loader import load_gif_frames

def load_flextok_model(
    model_name: str = 'EPFL-VILAB/flextok_d18_d28_dfn',
    device: str = None
):
    """
    Load a pretrained FlexTok encoder/decoder from HuggingFace Hub.

    Returns:
        model:    FlexTokFromHub instance, eval() mode on `device`.
        device:   Torch device string.
        bf16:     Whether bfloat16 is supported on this GPU.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FlexTokFromHub.from_pretrained(model_name).to(device).eval()
    bf16 = detect_bf16_support()
    return model, device, bf16

def tokenize_gif(
    gif_path: str,
    n_frames: int = 16,
    resize: tuple = (256, 256),
    model_name: str = 'EPFL-VILAB/flextok_d18_d28_dfn'
):
    """
    Tokenize the first `n_frames` of the input GIF using FlexTok.

    Args:
        gif_path:   Path to the input GIF.
        n_frames:   Number of frames from the start to tokenize.
        resize:     (width, height) to resize each frame.
        model_name: HuggingFace model identifier.

    Returns:
        tokens_list: List of torch.Tensor, each of shape [256] (the token sequence).
    """
    model, device, bf16 = load_flextok_model(model_name)
    video = load_gif_frames(gif_path, resize, n_frames, device)

    tokens_list = []
    with get_bf16_context(bf16):
        for idx in range(video.shape[0]):
            frame = video[idx].unsqueeze(0)          # shape [1, 3, H, W]
            tokens = model.tokenize(frame)           # shape [1, 256]
            tokens_list.append(tokens[0].cpu())      # move to CPU
    return tokens_list
