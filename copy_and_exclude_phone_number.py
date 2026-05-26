from pathlib import Path
import shutil

# Change this to your original folder path
source_folder = Path(r"C:\Users\lm2445\arabic\Arabic_transcribe_deidentify\input_files")

# Change this to your output folder path
target_folder = Path(r"C:\Users\lm2445\arabic\Arabic_transcribe_deidentify\output_files")
target_folder.mkdir(parents=True, exist_ok=True)

# The phone number is the 3rd block after splitting by "_"
# Python index starts from 0, so the 3rd block is index 2
PHONE_BLOCK_INDEX = 2

for file_path in source_folder.iterdir():
    if not file_path.is_file():
        continue

    # Use stem because Windows may hide ".txt"
    # Example:
    # abc_deidentified.txt -> stem is abc_deidentified
    if not file_path.stem.endswith("deidentified"):
        continue

    parts = file_path.stem.split("_")

    if len(parts) <= PHONE_BLOCK_INDEX:
        print(f"Skipped, not enough filename blocks: {file_path.name}")
        continue

    # Remove the 3rd block
    new_parts = parts[:PHONE_BLOCK_INDEX] + parts[PHONE_BLOCK_INDEX + 1:]
    new_stem = "_".join(new_parts)

    # Keep the original file extension, such as .txt
    new_name = new_stem + file_path.suffix
    target_path = target_folder / new_name

    # Avoid overwriting files with the same name
    counter = 1
    while target_path.exists():
        target_path = target_folder / f"{new_stem}_{counter}{file_path.suffix}"
        counter += 1

    shutil.copy2(file_path, target_path)

    print(f"Copied: {file_path.name} -> {target_path.name}")