# src/data_loader.py

import torch
from PIL import Image, ImageSequence
import torchvision.transforms.functional as TF

def load_gif_frames(
    gif_path: str,
    resize: tuple = (256, 256),
    max_frames: int = None,
    device: str = None
) -> torch.Tensor:
    """
    Load frames from a GIF, resize each to `resize`, convert to a torch.Tensor,
    and stack into a video tensor of shape [T, C, H, W].

    Args:
        gif_path: Path to the input GIF.
        resize:  (width, height) to resize each frame.
        max_frames: Maximum number of frames to load; if None, loads all.
        device:  Torch device string (e.g. 'cuda' or 'cpu'). 
                 If None, auto-detects CUDA if available.

    Returns:
        Tensor of shape [T, 3, H, W], where T = number of frames loaded.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    gif = Image.open(gif_path)
    frames = []
    for i, frame in enumerate(ImageSequence.Iterator(gif)):
        if max_frames is not None and i >= max_frames:
            break
        frame_rgb = frame.convert('RGB').resize(resize, Image.BILINEAR)
        frames.append(frame_rgb)

    if not frames:
        raise ValueError(f"No frames found in {gif_path}")

    tensors = [TF.to_tensor(img).to(device) for img in frames]
    video_tensor = torch.stack(tensors, dim=0)
    return video_tensor
