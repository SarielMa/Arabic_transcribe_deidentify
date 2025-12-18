#!/usr/bin/env python3

from transformers import pipeline
import torch
import re
import argparse
from pathlib import Path

# -----------------------
# Arguments
# -----------------------
ap = argparse.ArgumentParser()
ap.add_argument(
    "--input",
    required=True,
    help="Path to input transcription text file (.txt)"
)
ap.add_argument(
    "--output",
    default=None,
    help="Optional output file path. If not set, '_output_deidentified.txt' is used."
)
ap.add_argument(
    "--chunk_size",
    type=int,
    default=300,
    help="Chunk size for BERT (default: 300)"
)
args = ap.parse_args()

input_file = Path(args.input).expanduser().resolve()
print("DEBUG: using input_file =", input_file)
if not input_file.exists():
    raise FileNotFoundError(f"Input file not found: {input_file}")

if args.output:
    output_file = Path(args.output).expanduser().resolve()
else:
    output_file = input_file.with_name(
        input_file.stem + "_output_deidentified.txt"
    )

CHUNK_SIZE = args.chunk_size

# -----------------------
# Load NER model
# -----------------------
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

ner = pipeline(
    "ner",
    model="lm2445/for_deidentify",
    grouped_entities=True,
    device=0 if torch.cuda.is_available() else -1
)

# -----------------------
# Read input text
# -----------------------
text = input_file.read_text(encoding="utf-8").strip()

# -----------------------
# Chunking (BERT 512 limit)
# -----------------------
def chunk_text(text, size):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks

chunks = chunk_text(text, CHUNK_SIZE)

# -----------------------
# Process chunks with NER + replacement
# -----------------------
processed_chunks = []

for chunk in chunks:
    results = ner(chunk)
    deidentified = chunk

    for entity in results:
        if "word" not in entity:
            continue

        ent_text = entity["word"]

        if "entity_group" in entity and entity["entity_group"]:
            ent_label = entity["entity_group"]
        elif "entity" in entity and entity["entity"]:
            ent_label = entity["entity"].split("-")[-1]
        else:
            continue

        placeholder = f"<{ent_label}>"

        # Escape entity text to avoid regex issues
        pattern = re.escape(ent_text)
        deidentified = re.sub(pattern, placeholder, deidentified)

    processed_chunks.append(deidentified)

# -----------------------
# Combine chunks
# -----------------------
deidentified_text = "".join(processed_chunks)

# -----------------------
# Save output
# -----------------------
output_file.write_text(deidentified_text, encoding="utf-8")

print(f"De-identified text saved to: {output_file}")

# -----------------------
# GPU memory
# -----------------------
if torch.cuda.is_available():
    peak = torch.cuda.max_memory_allocated()
    print(f"Peak GPU memory usage: {peak / 1024**2:.2f} MB")
