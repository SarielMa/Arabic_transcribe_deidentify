from faster_whisper import WhisperModel

audio_file = "/home/lm2445/project_pi_sjf37/lm2445/Arabic_env/1224_audio/test2.mp3"

model = WhisperModel(
    "HebArabNlpProject/WhisperLevantine",
    device="cpu",
    compute_type="int8"
)

segments, info = model.transcribe(audio_file, language="ar", word_timestamps=True)

transcript_parts = []
for seg in segments:
    transcript_parts.append(seg.text)
    if seg.words:
        for w in seg.words:
            print(f"[{w.start:.2f}s -> {w.end:.2f}s] {w.word}")

print(" ".join(transcript_parts).strip())
