from pathlib import Path
import shutil
import re

# ====== Change these paths ======
SOURCE_DIR = Path(r"source_folder")
DEST_DIR = Path(r"output_folder")
# ================================

DEST_DIR.mkdir(parents=True, exist_ok=True)

# Matches an 8-digit block that is NOT the first date block
# Example:
# 20210610_1623348602254_4144_12345678_170_output_transcription_output_deidentified.txt
# becomes:
# 20210610_1623348602254_4144_170_output_transcription_output_deidentified.txt
PHONE_BLOCK_PATTERN = re.compile(r"(?<=_)\d{8}_")

for file_path in SOURCE_DIR.iterdir():
    if not file_path.is_file():
        continue

    # Only process txt files
    if file_path.suffix.lower() != ".txt":
        continue

    # Only copy deidentified files
    if "output_deidentified" not in file_path.name:
        continue

    # Remove the telephone-number block from the filename
    new_name = PHONE_BLOCK_PATTERN.sub("", file_path.name, count=1)

    dest_path = DEST_DIR / new_name

    # Avoid overwriting if duplicate names exist
    if dest_path.exists():
        stem = dest_path.stem
        suffix = dest_path.suffix
        i = 1
        while dest_path.exists():
            dest_path = DEST_DIR / f"{stem}_copy{i}{suffix}"
            i += 1

    shutil.copy2(file_path, dest_path)
    print(f"Copied: {file_path.name} -> {dest_path.name}")

print("Done.")