#!/usr/bin/env python3
"""
Robust Whisper chunk transcription for Arabic (or any language).

Key improvements vs. many minimal scripts:
- Decodes *any* audio format via torchaudio; falls back to ffmpeg->wav if needed.
- Normalizes to mono, 16 kHz PCM-equivalent float32 waveform before chunking.
- Computes chunk boundaries from decoded samples (not file metadata duration).
- Uses Whisper's recommended forced decoder prompt ids (avoids repetitive junk).
- Optional privacy-safe debug logging (no transcript content required).

Example:
  python transcribe_chunk_stable.py --input "C:\path\call.mp3" --output_name call1

If you want overlap:
  python transcribe_chunk_stable.py --input call.mp3 --output_name call1 --chunk_seconds 30 --overlap_seconds 2

Debug (privacy-safe):
  python transcribe_chunk_stable.py --input call.mp3 --output_name call1 --debug
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import torchaudio
except Exception as e:
    torchaudio = None  # type: ignore

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


TARGET_SR = 16000


@dataclass
class ChunkDebug:
    chunk_idx: int
    start_sec: float
    end_sec: float
    duration_sec: float
    num_samples: int
    rms: float
    peak: float
    silence_ratio: float  # fraction of samples with abs(x) < threshold


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def _run_ffmpeg_to_wav(src: Path, dst: Path, target_sr: int = TARGET_SR) -> None:
    """
    Convert audio to mono PCM WAV at target_sr via ffmpeg.
    """
    ffmpeg = _which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH, and torchaudio couldn't decode the input.")
    # -vn: no video, -ac 1 mono, -ar target_sr resample, pcm_s16le output
    cmd = [
        ffmpeg, "-y",
        "-i", str(src),
        "-vn",
        "-ac", "1",
        "-ar", str(target_sr),
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg conversion failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDERR:\n{proc.stderr[-2000:]}"
        )


def _load_audio_any(path: Path, target_sr: int = TARGET_SR) -> Tuple[torch.Tensor, int]:
    """
    Load audio via torchaudio if possible. If decoding fails, try ffmpeg->wav then reload.
    Returns (mono_waveform_float32, sr).
    waveform shape: [num_samples]
    """
    if torchaudio is None:
        raise RuntimeError("torchaudio is not available; please install torchaudio or use ffmpeg preprocessing.")

    try:
        waveform, sr = torchaudio.load(str(path))
    except Exception:
        # fallback: ffmpeg to temp wav then load
        tmp_wav = path.with_suffix(path.suffix + ".tmp16kmono.wav")
        _run_ffmpeg_to_wav(path, tmp_wav, target_sr=target_sr)
        waveform, sr = torchaudio.load(str(tmp_wav))
        try:
            tmp_wav.unlink(missing_ok=True)
        except Exception:
            pass

    # waveform: [channels, samples]
    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if waveform.ndim == 2:
        waveform = waveform.squeeze(0)

    waveform = waveform.to(torch.float32)

    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr

    # normalize if int-ish range
    # torchaudio usually gives float in [-1,1], so this is typically no-op.
    max_abs = float(waveform.abs().max().cpu())
    if max_abs > 1.5:
        waveform = waveform / max_abs

    return waveform, sr


def _chunk_indices(num_samples: int, sr: int, chunk_seconds: float, overlap_seconds: float) -> List[Tuple[int, int]]:
    """
    Return list of (start_sample, end_sample) pairs.
    Boundaries are computed from decoded samples (not metadata duration).
    """
    chunk_len = int(round(chunk_seconds * sr))
    overlap_len = int(round(overlap_seconds * sr))
    if chunk_len <= 0:
        raise ValueError("chunk_seconds too small.")
    if overlap_len < 0 or overlap_len >= chunk_len:
        raise ValueError("overlap_seconds must be >= 0 and < chunk_seconds.")

    step = chunk_len - overlap_len
    indices = []
    start = 0
    while start < num_samples:
        end = min(start + chunk_len, num_samples)
        indices.append((start, end))
        if end == num_samples:
            break
        start += step
    return indices


def _chunk_stats(x: torch.Tensor, silence_thresh: float = 0.01) -> Tuple[float, float, float]:
    """
    Compute privacy-safe stats: RMS, peak, and approx silence ratio.
    """
    if x.numel() == 0:
        return 0.0, 0.0, 1.0
    x_abs = x.abs()
    rms = float(torch.sqrt(torch.mean(x * x)).cpu())
    peak = float(torch.max(x_abs).cpu())
    silence_ratio = float(torch.mean((x_abs < silence_thresh).to(torch.float32)).cpu())
    return rms, peak, silence_ratio


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="Path to an audio file (mp3/wma/wav/m4a...)", default="/home/lm2445/project_pi_sjf37/lm2445/Arabic/V8.wav")
    ap.add_argument("--output_name", help="Base name for outputs (no extension)", default="V8")
    ap.add_argument("--out_dir", default="output", help="Output folder")
    ap.add_argument("--model_id", default="lm2445/for_transribing", help="HF model id")
    ap.add_argument("--language", default="ar", help="Language code (e.g., ar)")
    ap.add_argument("--chunk_seconds", type=float, default=15.0)
    ap.add_argument("--overlap_seconds", type=float, default=1.0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--debug", action="store_true", help="Write privacy-safe debug log (no transcript needed).")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load + normalize audio
    waveform, sr = _load_audio_any(in_path, target_sr=TARGET_SR)

    # Chunk boundaries from decoded samples
    indices = _chunk_indices(waveform.numel(), sr, args.chunk_seconds, args.overlap_seconds)

    # Setup whisper
    # device = 0 if (args.device in ["auto", "cuda"] and torch.cuda.is_available()) else -1
    # torch_dtype = torch.float16 if (device != -1) else torch.float32

    # processor = AutoProcessor.from_pretrained(args.model_id)
    # model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_id, torch_dtype=torch_dtype)
    # if device != -1:
    #     model = model.to("cuda")

    # # Use recommended decoder prompt ids (avoid repetitive junk)
    # forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task="transcribe")

    # asr = pipeline(
    #     "automatic-speech-recognition",
    #     model=model,
    #     tokenizer=processor.tokenizer,
    #     feature_extractor=processor.feature_extractor,
    #     device=device,
    #     torch_dtype=torch_dtype,
    #     # for Whisper, "generate_kwargs" is passed into model.generate()
    #     generate_kwargs={
    #         "forced_decoder_ids": forced_decoder_ids,
    #         "max_new_tokens": args.max_new_tokens,
    #     },
    # )

    # # Run chunks
    # texts: List[str] = []
    # debug_rows: List[Dict] = []

    # for i, (s, e) in enumerate(indices):
    #     chunk = waveform[s:e]

    #     if args.debug:
    #         rms, peak, silence_ratio = _chunk_stats(chunk, silence_thresh=0.01)
    #         debug_rows.append(asdict(ChunkDebug(
    #             chunk_idx=i,
    #             start_sec=s / sr,
    #             end_sec=e / sr,
    #             duration_sec=(e - s) / sr,
    #             num_samples=int(e - s),
    #             rms=rms,
    #             peak=peak,
    #             silence_ratio=silence_ratio,
    #         )))

    #     # HF ASR pipeline accepts numpy array + sampling_rate
    #     out = asr({"array": chunk.cpu().numpy(), "sampling_rate": sr})
    #     text = out["text"] if isinstance(out, dict) and "text" in out else str(out)
    #     texts.append(text.strip())

    # Setup whisper
    device = 0 if (args.device in ["auto", "cuda"] and torch.cuda.is_available()) else -1
    torch_dtype = torch.float16 if (device != -1) else torch.float32
    
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_id, torch_dtype=torch_dtype)
    if device != -1:
        model = model.to("cuda")
    
    # Use recommended decoder prompt ids (avoid repetitive junk)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task="transcribe")
    
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
        torch_dtype=torch_dtype,
        generate_kwargs={
            "forced_decoder_ids": forced_decoder_ids,
            "max_new_tokens": args.max_new_tokens,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.1,
        },
    )
    
    # Run chunks
    texts: List[str] = []
    debug_rows: List[Dict] = []
    print (f"there are totally {len(indices)} chunks")
    for i, (s, e) in enumerate(indices):
        print (f"the chunk {i} is being processed...")
        chunk = waveform[s:e]
    
        rms, peak, silence_ratio = _chunk_stats(chunk, silence_thresh=0.01)
    
        if args.debug:
            debug_rows.append(asdict(ChunkDebug(
                chunk_idx=i,
                start_sec=s / sr,
                end_sec=e / sr,
                duration_sec=(e - s) / sr,
                num_samples=int(e - s),
                rms=rms,
                peak=peak,
                silence_ratio=silence_ratio,
            )))
    
        # Skip near-silent chunks to prevent hallucinated loops
        if silence_ratio > 0.98 or rms < 0.003:
            texts.append("")
            continue
    
        out = asr({"array": chunk.cpu().numpy(), "sampling_rate": sr})
        text = out["text"] if isinstance(out, dict) and "text" in out else str(out)
        texts.append(text.strip())



    # Save outputs
    transcript_path = out_dir / f"{args.output_name}_output_transcription.txt"
    transcript_path.write_text("\n".join(texts) + "\n", encoding="utf-8")

    if args.debug:
        dbg_path = out_dir / f"{args.output_name}_debug.jsonl"
        _write_jsonl(dbg_path, debug_rows)

    print(f"Wrote transcript: {transcript_path}")
    if args.debug:
        print(f"Wrote debug log: {out_dir / f'{args.output_name}_debug.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
