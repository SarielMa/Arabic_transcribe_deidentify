# local arabic audio
import torch
import gc
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torchaudio
import math
from pathlib import Path
#from transformers import pipeline
#import torch
import re
import os

def get_transcribe_and_deidentify(file_path):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    # -----------------------
    # you need to config the file_path and output_name
    # file_path = r"C:\Users\lm2445\arabic\V8.wav"
    # output_name = "V8"
    file_path = file_path
    result_path = "results"
    if not os.path.exists(result_path):
        os.makedirs(result_path, exist_ok=True)
    # -----------------------
    output_name = file_path.stem   # "V8"
    CHUNK_SECONDS = 30   # no overlap
    print ("transcrbing...")
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

    with open(os.path.join(result_path, out_file), "w", encoding="utf-8") as f:
        f.write(final_text.strip())

    print(f"transcribed text is : {final_text}")

    print(f"\nSaved transcription → {out_file}")
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated()
        print(f"Peak GPU memory: {peak / 1024**2:.2f} MB")


    # the following is deidentify
    # +++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++
    print ("deindentifying...")
    # -----------------------
    # Settings
    # -----------------------
    # you need to set the input text path
    #input_file = r"V8_output_transcription.txt"  
    #input_file2 = os.path.join(result_path, out_file) 
    input_file2 = out_file
    # ------------------------
    # the following does not need to modify
    output_file = f"{input_file2.split('.txt')[0]}_output_deidentified.txt"

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
    with open(os.path.join(result_path, out_file), "r", encoding="utf-8") as f:
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
    with open(os.path.join(result_path, output_file), "w", encoding="utf-8") as f:
        f.write(deidentified_text)

    print(f"\nDe-identified text saved to: {output_file}")

    # -----------------------
    # GPU memory
    # -----------------------
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated()
        print(f"Peak GPU memory usage: {peak / 1024**2:.2f} MB")



if __name__ == "__main__":
    #file_path = Path(r"C:\Users\lm2445\arabic\arabic_speech_sample\ARA NORM  0002.wav")
    # need config the file type and folder that contains all the audio files
    file_type = "wav" # can also be 'mp3'
    audio_dir = Path(r"C:\Users\lm2445\arabic\arabic_speech_sample") # the folder containing all the audios
    #wav_files = list(audio_dir.glob("*.wav"))
    for file_path in audio_dir.glob(f"*.{file_type}"):
        print (f"working on the file {file_path}")
        get_transcribe_and_deidentify(file_path)