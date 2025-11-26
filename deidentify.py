from transformers import pipeline
import torch
import re
import os

# -----------------------
# Settings
# -----------------------
input_file = "V8_output_transcription.txt"                 # <-- your input .txt
output_file = f"{input_file.split(".txt")[0]}_output_deidentified.txt"       # <-- output .txt

# -----------------------
# Load NER model
# -----------------------
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
# Run NER
# -----------------------
results = ner(text)

# -----------------------
# Build replacements
# -----------------------
deidentified_text = text

for entity in results:

    # skip malformed entries
    if "word" not in entity:
        continue

    ent_text = entity["word"]

    # determine entity label safely
    if "entity_group" in entity and entity["entity_group"]:
        ent_label = entity["entity_group"]
    elif "entity" in entity and entity["entity"]:
        ent_label = entity["entity"].split("-")[-1]
    else:
        continue

    placeholder = f"<{ent_label}>"

    pattern = re.escape(ent_text)
    deidentified_text = re.sub(pattern, placeholder, deidentified_text)
# -----------------------
# Print results
# -----------------------
print("=== Original Text ===")
print(text)
print("\n=== NER Output ===")
print(results)
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
peak = torch.cuda.max_memory_allocated()
print(f"Peak GPU memory usage: {peak / 1024**2:.2f} MB")
