#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.tokenizer import tokenize_gif
from src.detokenizer import detokenize_tokens

def main():
    p = argparse.ArgumentParser(
        description="Tokenize & detokenize a GIF frame‐by‐frame with FlexTok"
    )
    p.add_argument(
        "gif_path", help="Path to your GIF (e.g. data/raw/v_PoleVault_g01_c01.gif)"
    )
    p.add_argument(
        "--n_frames", type=int, default=20,
        help="How many frames from the start to tokenize"
    )
    p.add_argument(
        "--timesteps", type=int, default=25,
        help="Number of detokenizer diffusion steps"
    )
    p.add_argument(
        "--guidance_scale", type=float, default=7.5,
        help="Classifier‐free guidance scale for detokenizer"
    )
    args = p.parse_args()

    # 1) Tokenize
    tokens = tokenize_gif(
        args.gif_path,
        n_frames=args.n_frames
    )
    print(f"⚡️ Tokenized {len(tokens)} frames.")

    # 2) Detokenize back to PIL images
    recon_images = detokenize_tokens(
        tokens,
        timesteps=args.timesteps,
        guidance_scale=args.guidance_scale
    )
    print(f"⚡️ Reconstructed {len(recon_images)} images.")

    # 3) (Optional) save out
    for i, img in enumerate(recon_images):
        out_path = f"output/frame_{i:03d}.png"
        img.save(out_path)
        print(f"  – Saved {out_path}")

if __name__ == "__main__":
    main()
