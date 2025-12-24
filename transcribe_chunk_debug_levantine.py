#!/usr/bin/env python3
"""
Robust Arabic transcription with optional chunking.

IMPORTANT:
- HebArabNlpProject/WhisperLevantine is NOT a transformers Whisper checkpoint.
- It is a faster-whisper CTranslate2 model (model.bin), so we use faster-whisper.

Install:
  python -m pip install -U faster-whisper

Optional (recommended) for MP3 decoding fallback:
  Ensure ffmpeg is on PATH

Example:
  python transcribe_chunk_fasterwhisper.py --input "/path/call.mp3" --output_name call1 --debug
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"


import torch

try:
    import torchaudio
except Exception:
    torchaudio = None  # type: ignore

from faster_whisper import WhisperModel

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
    silence_ratio: float


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def _run_ffmpeg_to_wav(src: Path, dst: Path, target_sr: int = TARGET_SR) -> None:
    ffmpeg = _which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH, and torchaudio couldn't decode the input.")
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
    Returns mono float32 waveform in [-1, 1] at target_sr.
    """
    if torchaudio is None:
        raise RuntimeError("torchaudio is not available; please install torchaudio or rely on ffmpeg-only path.")

    try:
        waveform, sr = torchaudio.load(str(path))
    except Exception:
        tmp_wav = path.with_suffix(path.suffix + ".tmp16kmono.wav")
        _run_ffmpeg_to_wav(path, tmp_wav, target_sr=target_sr)
        waveform, sr = torchaudio.load(str(tmp_wav))
        try:
            tmp_wav.unlink(missing_ok=True)
        except Exception:
            pass

    # [channels, samples] -> mono
    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if waveform.ndim == 2:
        waveform = waveform.squeeze(0)

    waveform = waveform.to(torch.float32)

    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr

    max_abs = float(waveform.abs().max().cpu())
    if max_abs > 1.5:
        waveform = waveform / max_abs

    return waveform, sr


def _chunk_indices(num_samples: int, sr: int, chunk_seconds: float, overlap_seconds: float) -> List[Tuple[int, int]]:
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


def _write_wav_pcm16(path: Path, waveform: torch.Tensor, sr: int) -> None:
    """
    Write mono waveform float32 [-1,1] to 16-bit PCM WAV using torchaudio (preferred).
    """
    if torchaudio is None:
        raise RuntimeError("torchaudio not available to write wav chunks.")
    # torchaudio.save expects [channels, samples]
    x = waveform.clamp(-1.0, 1.0).unsqueeze(0).cpu()
    torchaudio.save(str(path), x, sr, encoding="PCM_S", bits_per_sample=16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="Path to an audio file (mp3/wma/wav/m4a...)", default=r"C:\Users\lm2445\arabic\real_test\test2.mp3")
    ap.add_argument("--output_name", help="Base name for outputs (no extension)", default="test2_levantine")
    ap.add_argument("--out_dir", default="output", help="Output folder")

    # faster-whisper model: can be HF repo id OR local directory containing model.bin
    ap.add_argument("--model_id", default="HebArabNlpProject/WhisperLevantine", help="HF repo id or local model dir")
    ap.add_argument("--language", default="ar", help="Language code, e.g. ar")

    ap.add_argument("--chunk_seconds", type=float, default=15.0)
    ap.add_argument("--overlap_seconds", type=float, default=1.0)

    ap.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--compute_type", default="int8",
                    help="faster-whisper compute_type: float16/int8/int8_float16/float32 or 'auto'")

    ap.add_argument("--vad_filter", action="store_true", help="Enable faster-whisper VAD filter (often helps calls).")
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--best_of", type=int, default=5)

    ap.add_argument("--debug", action="store_true", help="Write privacy-safe debug log (no transcript needed).")
    args = ap.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load + normalize audio (mono 16k float32)
    waveform, sr = _load_audio_any(in_path, target_sr=TARGET_SR)

    indices = _chunk_indices(waveform.numel(), sr, args.chunk_seconds, args.overlap_seconds)

    # Setup faster-whisper
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if args.compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    else:
        compute_type = args.compute_type

    model = WhisperModel(
        args.model_id,
        device=device,
        compute_type=compute_type,
    )

    texts: List[str] = []
    debug_rows: List[Dict] = []

    print(f"Total chunks: {len(indices)}")

    # Use temp wav chunks so faster-whisper can read them reliably
    with tempfile.TemporaryDirectory(prefix="fw_chunks_") as tmpdir:
        tmpdir_p = Path(tmpdir)

        for i, (s, e) in enumerate(indices):
            print(f"Processing chunk {i} ...")
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

            # Skip near-silent chunks to prevent hallucinated junk
            if silence_ratio > 0.98 or rms < 0.003:
                texts.append("")
                continue

            chunk_wav = tmpdir_p / f"chunk_{i:06d}.wav"
            _write_wav_pcm16(chunk_wav, chunk, sr)

            segments, info = model.transcribe(
                str(chunk_wav),
                language=args.language,
                vad_filter=args.vad_filter,
                beam_size=args.beam_size,
                best_of=args.best_of,
            )
            chunk_text = " ".join(seg.text.strip() for seg in segments).strip()
            texts.append(chunk_text)

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
