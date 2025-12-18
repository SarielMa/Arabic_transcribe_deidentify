#!/usr/bin/env python3
"""
Run transcription -> de-identification for ONE audio file.

Usage:
  python run_audio_to_deid.py --input path/to/audio.mp3

Requirements:
  - transcribe_chunk_stable.py
  - deidentify_cli.py (or deidentify.py with CLI support)
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        required=True,
        help="Path to input audio file (mp3/wav/wma/etc.)"
    )
    ap.add_argument(
        "--transcribe_script",
        default="transcribe_chunk_debug.py",
        help="Path to transcription script"
    )
    ap.add_argument(
        "--deid_script",
        default="deidentify_debug.py",
        help="Path to de-identification script"
    )
    ap.add_argument(
        "--out_dir",
        default="output",
        help="Output directory"
    )

    # transcription params (safe defaults)
    ap.add_argument("--chunk_seconds", type=float, default=15.0)
    ap.add_argument("--overlap_seconds", type=float, default=1.0)
    ap.add_argument("--language", default="ar")
    ap.add_argument("--max_new_tokens", type=int, default=256)

    args = ap.parse_args()

    audio_path = Path(args.input).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    transcribe_script = Path(args.transcribe_script).resolve()
    deid_script = Path(args.deid_script).resolve()
    out_dir = Path(args.out_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    base = audio_path.stem

    # Expected output from transcription script
    transcript_path = out_dir / f"{base}_output_transcription.txt"
    deid_path = out_dir / f"{base}_output_deidentified.txt"

    python_exe = sys.executable

    # -------------------------
    # Step 1: Transcription
    # -------------------------
    print("=== Step 1: Transcribing audio ===")

    cmd_transcribe = [
        python_exe, str(transcribe_script),
        "--input", str(audio_path),
        "--output_name", base,
        "--out_dir", str(out_dir),
        "--chunk_seconds", str(args.chunk_seconds),
        "--overlap_seconds", str(args.overlap_seconds),
        "--language", args.language,
        "--max_new_tokens", str(args.max_new_tokens),
    ]

    subprocess.run(cmd_transcribe, check=True)

    if not transcript_path.exists():
        raise RuntimeError(f"Transcription failed, file not found: {transcript_path}")

    # -------------------------
    # Step 2: De-identification
    # -------------------------
    print("=== Step 2: De-identifying transcript ===")

    cmd_deid = [
        python_exe, str(deid_script),
        "--input", str(transcript_path),
        "--output", str(deid_path),
    ]

    subprocess.run(cmd_deid, check=True)

    if not deid_path.exists():
        raise RuntimeError(f"De-identification failed, file not found: {deid_path}")

    print("\n=== DONE ===")
    print(f"Transcript:     {transcript_path}")
    print(f"De-identified:  {deid_path}")


if __name__ == "__main__":
    main()
