#!/usr/bin/env python3
"""
Batch: walk Calls/YYYYMMDD/*.mp3 -> transcribe -> deidentify
Preserves folder structure (YYYYMMDD) and original mp3 stem in output filenames.

Example:
  python run_calls_tree_to_deid.py --calls_root "D:\Data\Calls"
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls_root", required=True, help="Path to Calls/ folder (contains YYYYMMDD subfolders)")
    ap.add_argument("--out_root", default="output", help="Output root folder")
    ap.add_argument("--transcribe_script", default="transcribe_chunk_debug.py", help="Transcription script filename/path")
    ap.add_argument("--deid_script", default="deidentify_debug.py", help="De-id script filename/path")
    ap.add_argument("--format", default=".wav", help="can also be .mp3")

    # transcription params
    ap.add_argument("--chunk_seconds", type=float, default=15.0)
    ap.add_argument("--overlap_seconds", type=float, default=1.0)
    ap.add_argument("--language", default="ar")
    ap.add_argument("--max_new_tokens", type=int, default=256)

    # behavior
    ap.add_argument("--recursive", action="store_true", help="Also search mp3 in nested subfolders under each date")
    ap.add_argument("--continue_on_error", action="store_true", help="Skip failed files and continue")
    ap.add_argument("--debug", action="store_true", help="Enable transcription debug jsonl")

    args = ap.parse_args()

    calls_root = Path(args.calls_root).expanduser().resolve()
    if not calls_root.exists():
        raise FileNotFoundError(f"Calls folder not found: {calls_root}")

    # Resolve scripts relative to THIS wrapper file (robust on Windows)
    here = Path(__file__).resolve().parent
    transcribe_script = (Path(args.transcribe_script) if Path(args.transcribe_script).is_absolute()
                         else (here / args.transcribe_script)).resolve()
    deid_script = (Path(args.deid_script) if Path(args.deid_script).is_absolute()
                   else (here / args.deid_script)).resolve()

    if not transcribe_script.exists():
        raise FileNotFoundError(f"Transcribe script not found: {transcribe_script}")
    if not deid_script.exists():
        raise FileNotFoundError(f"De-id script not found: {deid_script}")

    out_root = Path(args.out_root).expanduser().resolve()
    transcripts_root = out_root / "transcripts"
    deid_root = out_root / "deidentified"
    transcripts_root.mkdir(parents=True, exist_ok=True)
    deid_root.mkdir(parents=True, exist_ok=True)

    # Find mp3 files
    format = args.format
    #pattern = "**/*.mp3" if args.recursive else "*.mp3"
    pattern = f"**/*{format}" if args.recursive else f"*{format}"
    mp3_files = sorted(calls_root.glob(f"*/{pattern}"))  # at least one level: Calls/YYYYMMDD/...

    print(f"Found {len(mp3_files)} {format} files under {calls_root}")
    if not mp3_files:
        return 0

    py = sys.executable
    ok, fail = 0, 0

    for idx, mp3 in enumerate(mp3_files, start=1):
        rel = mp3.relative_to(calls_root)          # e.g., 20210609/1623...._3.mp3
        rel_dir = rel.parent                       # e.g., 20210609
        stem = mp3.stem                            # keep original informative name

        # Mirror date folder in outputs
        t_out_dir = transcripts_root / rel_dir
        d_out_dir = deid_root / rel_dir
        t_out_dir.mkdir(parents=True, exist_ok=True)
        d_out_dir.mkdir(parents=True, exist_ok=True)

        transcript_path = t_out_dir / f"{stem}_output_transcription.txt"
        deid_path = d_out_dir / f"{stem}_output_deidentified.txt"

        print(f"\n[{idx}/{len(mp3_files)}] {rel}")

        # 1) Transcribe (writes into t_out_dir)
        cmd_t = [
            py, str(transcribe_script),
            "--input", str(mp3),
            "--output_name", stem,
            "--out_dir", str(t_out_dir),
            "--chunk_seconds", str(args.chunk_seconds),
            "--overlap_seconds", str(args.overlap_seconds),
            "--language", args.language,
            "--max_new_tokens", str(args.max_new_tokens),
        ]
        if args.debug:
            cmd_t.append("--debug")

        try:
            subprocess.run(cmd_t, check=True)
            if not transcript_path.exists():
                raise RuntimeError(f"Expected transcript not found: {transcript_path}")

            # 2) De-identify (reads transcript, writes into d_out_dir)
            cmd_d = [
                py, str(deid_script),
                "--input", str(transcript_path),
                "--output", str(deid_path),
            ]
            subprocess.run(cmd_d, check=True)

            if not deid_path.exists():
                raise RuntimeError(f"Expected de-id output not found: {deid_path}")

            ok += 1
        except Exception as e:
            fail += 1
            print(f"ERROR on {rel}: {e}")
            if not args.continue_on_error:
                raise

    print(f"\nDone. OK={ok}, FAIL={fail}")
    print(f"Transcripts:   {transcripts_root}")
    print(f"De-identified: {deid_root}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
