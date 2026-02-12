# need to install mutagen first
# pip install mutagen


import os
import random
import shutil
from mutagen.mp3 import MP3

# ========= CONFIG =========
SOURCE_ROOT = r"C:\Users\lm2445\arabic\Calls"              # root folder, need to configure to the path to your "Calls"
DEST_FOLDER = "filtered_5_10min"
MIN_MINUTES = 5
MAX_MINUTES = 10
N_SAMPLES = 1000
SEED = 42
# ===========================

random.seed(SEED)

MIN_SECONDS = MIN_MINUTES * 60
MAX_SECONDS = MAX_MINUTES * 60

os.makedirs(DEST_FOLDER, exist_ok=True)

eligible_files = []

print("Scanning files...")

for root, dirs, files in os.walk(SOURCE_ROOT):
    for file in files:
        if file.lower().endswith(".mp3"):
            full_path = os.path.join(root, file)

            try:
                audio = MP3(full_path)
                duration = audio.info.length

                if MIN_SECONDS <= duration <= MAX_SECONDS:
                    eligible_files.append(full_path)

            except Exception as e:
                print(f"Skipping {full_path}: {e}")

print(f"Total eligible files found: {len(eligible_files)}")

if len(eligible_files) == 0:
    print("No files in specified duration range.")
    exit()

if len(eligible_files) < N_SAMPLES:
    selected_files = eligible_files
    print(f"Only {len(eligible_files)} files available.")
else:
    selected_files = random.sample(eligible_files, N_SAMPLES)

print(f"Copying {len(selected_files)} files...")

for src_path in selected_files:
    filename = os.path.basename(src_path)
    folder_name = os.path.basename(os.path.dirname(src_path))

    new_filename = f"{folder_name}_{filename}"
    dst_path = os.path.join(DEST_FOLDER, new_filename)

    shutil.copy2(src_path, dst_path)

print("Done.")
