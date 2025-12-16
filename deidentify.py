from transformers import pipeline
import torch
import re
import os

# -----------------------
# Settings
# -----------------------
# you need to set the input text path
input_file = "V8_output_transcription.txt"   
# ------------------------
# the following does not need to modify
output_file = f"{input_file.split('.txt')[0]}_output_deidentified.txt"

CHUNK_SIZE = 300   # safe chunk size for BERT

# -----------------------
# Load NER model
# -----------------------
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

ner = pipeline(
    "ner",
    model="lm2445/for_deidentify",
    grouped_entities=True
)

# -----------------------
# Read input text
# -----------------------
with open(input_file, "r", encoding="utf-8") as f:
    text = f.read().strip()

# -----------------------
# Chunking (fix BERT 512 limit)
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
        # Skip malformed entries
        if "word" not in entity:
            continue

        ent_text = entity["word"]

        # Determine entity label safely
        if "entity_group" in entity and entity["entity_group"]:
            ent_label = entity["entity_group"]
        elif "entity" in entity and entity["entity"]:
            ent_label = entity["entity"].split("-")[-1]
        else:
            continue

        placeholder = f"<{ent_label}>"

        pattern = re.escape(ent_text)
        deidentified = re.sub(pattern, placeholder, deidentified)

    processed_chunks.append(deidentified)

# -----------------------
# Combine chunks
# -----------------------
deidentified_text = "".join(processed_chunks)

# -----------------------
# Print results
# -----------------------
print("=== Original Text ===")
print(text)
print("\n=== De-identified Text ===")
print(deidentified_text)

# -----------------------
# Save output text
# -----------------------
with open(output_file, "w", encoding="utf-8") as f:
    f.write(deidentified_text)

print(f"\nDe-identified text saved to: {output_file}")

# -----------------------
# GPU memory
# -----------------------
if torch.cuda.is_available():
    peak = torch.cuda.max_memory_allocated()
    print(f"Peak GPU memory usage: {peak / 1024**2:.2f} MB")
