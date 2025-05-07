import os
import shutil
import cv2
import random
from tqdm import tqdm
from collections import defaultdict

# ---------------- Configuration ----------------
ROOT = r"E:\video-tokenizer\video_clips\cropped_clips"
OUTPUT = r"E:\video-tokenizer\video_clips\short_clips_by_food"
CATEGORIES = ['PastaSalad', 'TurkeySandwich', 'BaconAndEggs',
              'ContinentalBreakfast', 'Cheeseburger', 'GreekSalad', 'Pizza']
MAX_CLIPS = 15
MIN_DURATION = 5.0   # in seconds
MAX_DURATION = 10.0  # in seconds
SEED = 42

# ---------------- Initialization ----------------
random.seed(SEED)
os.makedirs(OUTPUT, exist_ok=True)
for cat in CATEGORIES:
    os.makedirs(os.path.join(OUTPUT, cat), exist_ok=True)

category_clips = defaultdict(list)

# ---------------- Main Sampling Loop ----------------
for folder in tqdm(sorted(os.listdir(ROOT))):
    folder_path = os.path.join(ROOT, folder)
    if not os.path.isdir(folder_path):
        continue

    for category in CATEGORIES:
        if folder.endswith(category):
            files = sorted([f for f in os.listdir(folder_path) if f.endswith('.mp4')])
            random.shuffle(files)  # deterministic shuffle

            for file in files:
                if len(category_clips[category]) >= MAX_CLIPS:
                    break

                video_path = os.path.join(folder_path, file)
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                cap.release()

                if fps <= 0 or frames <= 0:
                    continue

                duration = frames / fps
                if MIN_DURATION <= duration <= MAX_DURATION:
                    dst_path = os.path.join(OUTPUT, category, file)
                    shutil.copy(video_path, dst_path)
                    category_clips[category].append(dst_path)
                    print(f"✔ Selected [{category}] - {file} ({duration:.2f}s)")

# ---------------- Summary ----------------
print("\n=== Final Selection Summary ===")
for cat in CATEGORIES:
    print(f"{cat}: {len(category_clips[cat])} videos selected")
