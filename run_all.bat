@echo off
setlocal enabledelayedexpansion

REM ===== CONFIG =====
set MP3_FOLDER=C:\Users\lm2445\arabic\Arabic_transcribe_deidentify\filtered_5_10min
set SCRIPT_DIR=C:\Users\lm2445\arabic\Arabic_transcribe_deidentify
set OUTPUT_DIR=%SCRIPT_DIR%\output
REM ==================

cd %SCRIPT_DIR%

for %%f in (%MP3_FOLDER%\*.mp3) do (

    echo ==========================================
    echo Processing %%~nxf

    REM Get filename without extension
    set NAME=%%~nf

    REM 1. Transcribe
    python transcribe_chunk_debug_levantine.py --input "%%f" --output_name !NAME!

    REM 2. Deidentify (uses the generated transcription txt)
    python deidentify_debug.py --input "%OUTPUT_DIR%\!NAME!_output_transcription.txt"

)

echo All files processed.
pause
