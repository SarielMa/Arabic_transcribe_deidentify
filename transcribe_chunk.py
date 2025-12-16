# local arabic audio
import torch
import gc
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torchaudio
import math
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
# -----------------------
# you need to config the file_path and output_name
file_path = r"C:\Users\lm2445\arabic\V8.wav"
output_name = "V8"
# -----------------------
CHUNK_SECONDS = 30   # no overlap

# -----------------------
# Load audio
# -----------------------
waveform, sr = torchaudio.load(file_path)

if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0)

waveform = waveform.squeeze()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "lm2445/for_transribing"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True
).to(device)

processor = AutoProcessor.from_pretrained(model_id)
# ------------------------------------------------------
# FIXED: Only remove forced tokens from CONFIG, not tokenizer
# ------------------------------------------------------
model.config.forced_decoder_ids = None
model.config.suppress_tokens = None


# -----------------------
# Arabic-only Whisper pipeline (IMPORTANT)
# -----------------------
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=device,
    torch_dtype=torch_dtype,
    return_timestamps=False,
    generate_kwargs={
        "language": "ar",        # force Arabic output
        "task": "transcribe",    # do NOT translate
        "forced_decoder_ids": None,   # avoid conflicts
    }
)

# -----------------------
# Chunking logic
# -----------------------
chunk_size = CHUNK_SECONDS * sr
final_text = ""

num_chunks = math.ceil(len(waveform) / chunk_size)
print(f"Total chunks: {num_chunks}")

for i in range(num_chunks):
    print(f"Chunk {i+1}/{num_chunks}")
    start = i * chunk_size
    end = min((i+1) * chunk_size, len(waveform))

    chunk = waveform[start:end]

    result = pipe({"array": chunk.numpy(), "sampling_rate": sr})
    final_text += " " + result["text"]

    del result, chunk
    gc.collect()
    # torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# -----------------------
# Save output
# -----------------------
out_file = f"{output_name}_output_transcription.txt"

with open(out_file, "w", encoding="utf-8") as f:
    f.write(final_text.strip())

print(final_text)

print(f"\nSaved transcription → {out_file}")
if torch.cuda.is_available():
    peak = torch.cuda.max_memory_allocated()
    print(f"Peak GPU memory: {peak / 1024**2:.2f} MB")




# # local arabic audio
# import torch
# import gc
# from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
# import torchaudio
# import math

# torch.cuda.reset_peak_memory_stats()

# file_path = "/home/lm2445/project_pi_sjf37/lm2445/Arabic/1777.wav"
# output_name = "1777"

# CHUNK_SECONDS = 30   # no overlap
# # -----------------------
# # Load audio
# # -----------------------
# waveform, sr = torchaudio.load(file_path)

# if waveform.shape[0] > 1:
#     waveform = waveform.mean(dim=0)

# waveform = waveform.squeeze()

# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# model_id = "lm2445/for_transribing"

# model = AutoModelForSpeechSeq2Seq.from_pretrained(
#     model_id,
#     torch_dtype=torch_dtype,
#     low_cpu_mem_usage=True,
#     use_safetensors=True
# ).to(device)

# processor = AutoProcessor.from_pretrained(model_id)

# pipe = pipeline(
#     "automatic-speech-recognition",
#     model=model,
#     tokenizer=processor.tokenizer,
#     feature_extractor=processor.feature_extractor,
#     device=device,
#     torch_dtype=torch_dtype,
#     return_timestamps=False
# )

# # -----------------------
# # Chunk
# # -----------------------
# chunk_size = CHUNK_SECONDS * sr
# final_text = ""

# num_chunks = math.ceil(len(waveform) / chunk_size)
# print(f"Total chunks: {num_chunks}")

# for i in range(num_chunks):
#     print(f"Chunk {i+1}/{num_chunks}")
#     start = i * chunk_size
#     end = min((i+1) * chunk_size, len(waveform))

#     chunk = waveform[start:end]

#     result = pipe({"array": chunk.numpy(), "sampling_rate": sr})
#     final_text += " " + result["text"]

#     del result, chunk
#     gc.collect()
#     torch.cuda.empty_cache()

# out_file = f"{output_name}_output_transcription.txt"

# with open(out_file, "w", encoding="utf-8") as f:
#     f.write(final_text.strip())

# print (final_text)

# print(f"\nSaved transcription → {out_file}")

# peak = torch.cuda.max_memory_allocated()
# print(f"Peak GPU memory: {peak / 1024**2:.2f} MB")



