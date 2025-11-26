# local arabic audio
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torchaudio

# Reset GPU peak memory stats
torch.cuda.reset_peak_memory_stats()

# -----------------------
# Load audio file
# -----------------------
file_path = "/home/lm2445/project_pi_sjf37/lm2445/Arabic/V8.wav"
output_name = "V8"

waveform, sr = torchaudio.load(file_path)

# Convert to mono if audio has multiple channels
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0)

sample = {"array": waveform.squeeze().numpy(), "sampling_rate": sr}

# -----------------------
# Device setup
# -----------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# -----------------------
# Load model
# -----------------------
model_id = "lm2445/for_transribing"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True
).to(device)

processor = AutoProcessor.from_pretrained(model_id)

# -----------------------
# Build Whisper-ASR pipeline (Arabic transcription only)
# -----------------------
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=device,
    torch_dtype=torch_dtype,
    return_timestamps=False,     # if you want timestamps: True is fine
    generate_kwargs={
        "language": "ar",        # force Arabic
        "task": "transcribe",    # do NOT translate
        "forced_decoder_ids": None,  # avoid conflicts
    },
)

# -----------------------
# Run transcription
# -----------------------
sample["array"] = sample["array"][: int(30 * sample["sampling_rate"])]

result = pipe(sample)
transcribed_text = result["text"]
print(transcribed_text)

# -----------------------
# Save to TXT file
# -----------------------
output_file = f"{output_name}_output_transcription.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(transcribed_text)

print(f"\nTranscription saved to: {output_file}")

# -----------------------
# GPU Memory
# -----------------------
peak_memory = torch.cuda.max_memory_allocated()
print(f"Peak GPU memory usage: {peak_memory / 1024**2:.2f} MB")
