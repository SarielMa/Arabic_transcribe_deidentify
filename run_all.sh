#!/bin/bash

MP3_FOLDER="/path/to/filtered_5_10min"
SCRIPT_DIR="/path/to/arabic"
OUTPUT_DIR="$SCRIPT_DIR/Arabic_transcribe_deidentify/output"

cd $SCRIPT_DIR

for file in "$MP3_FOLDER"/*.mp3; do

    echo "=========================================="
    echo "Processing $file"

    name=$(basename "$file" .mp3)

    # 1. Transcribe
    python transcribe_chunk_debug_levantine.py --input "$file" --output_name "$name"

    # 2. Deidentify
    python deidentify_debug.py --input "$OUTPUT_DIR/${name}_output_transcription.txt"

done

echo "All files processed."
