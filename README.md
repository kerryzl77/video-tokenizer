# Video Tokenizer

A Python package for tokenizing and detokenizing video frames using a patch-based approach.

## Quick Start

1. **Setup Environment**
```bash
# Create and activate conda environment
conda create -n flextok python=3.10 -y && conda activate flextok

# Install dependencies
pip install -r requirements.txt
```

2. **Prepare Data**
- Place your GIF files in `data/raw/` directory
- Example: `data/raw/v_PoleVault_g01_c01.gif`

3. **Run Tokenization**
```bash
# Advanced usage with parameters
python scripts/run_tokenization.py \
  data/raw/v_PoleVault_g01_c01.gif \
  --n_frames 20 \
  --timesteps 25 \
  --guidance_scale 7.5
```

## Project Structure
```
video-tokenizer/
├── data/                    # Data directory
│   └── raw/                 # Raw GIF files
├── src/                     # Core Python modules
│   ├── data_loader.py       # Loads GIFs and extracts frames
│   ├── tokenizer.py         # Wraps FlexTok tokenization logic
│   └── detokenizer.py       # Wraps FlexTok detokenizer logic
├── scripts/                 # Command-line entrypoints
│   └── run_tokenization.py  
├── notebooks/               # Example notebooks
│   ├── UCF101Frame.ipynb    # Colab for Loading / Filtering UCF101 data
│   └── Flextok_inference_Video.ipynb  # Colab inference code
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Parameters
- `--n_frames`: Number of frames to process (default: 20)
- `--timesteps`: Number of timesteps for tokenization (default: 25)
- `--guidance_scale`: Guidance scale for tokenization (default: 7.5)

## Development
- Core functionality is in the `src` directory
- Command-line tools are in the `scripts` directory
- Example notebooks are in the `notebooks` directory 