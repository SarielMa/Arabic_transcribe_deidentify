# Arabic NER & De-identification Project

This repository provides an environment setup and example code for performing Arabic Named Entity Recognition (NER) and text de-identification using Hugging Face transformer models.

---

# 1. Environment Setup

## 1.1 Install Miniconda
If Conda is not installed, download Miniconda from:
https://docs.conda.io/en/latest/miniconda.html

## 1.2 Create the environment
Make sure `environment.yml` is in your working directory.

```bash
conda env create -f environment.yml
```

If you are using Windows system, use the following command instead:

```bash
conda env create -f .\environment_win.yml
```

## 1.3 Activate the environment

```bash
conda activate arabic_proj
```
## 1.4 (Optional) Enable the environment in Jupyter Notebook
If you plan to run code inside Jupyter Notebook:
```bash
conda install ipykernel
python -m ipykernel install --user --name arabic_proj --display-name "arabic_proj_2025"
```
You can then select arabic_proj_2025 inside Jupyter.

# 2. Run the script

## 2.1 Run the transcribing:
<!-- 
```bash
python transcribe.py
```
### 2.1.1 if the audio is too long, use the chunk version -->

```bash
python .\transcribe_chunk_debug.py --input "C:\Users\lm2445\arabic\V8.wav" --output_name V8
```

This is tested on windows, you can modify the input and output_name as needed

### 2.1.1 If the input is Levantine Arabic not Modern Standard Arabic (MSA), please the Levantine version:

First, need to install some dependency (Do this inside the conda environment of (arabic_proj)):

```bash
python -m pip install -U faster-whisper
```

Then use the following to run it:
```bash
python .\transcribe_chunk_debug_levantine.py --input "C:\Users\lm2445\arabic\test2.mp3" --output_name test2_levantine
```

## 2.2 modify the input text path of deidentify.py, then

```bash
python deidentify.py
```

### 2.2.1 Or use the debug version as well:

```bash
 python .\deidentify_debug.py --input "C:\Users\lm2445\arabic\Arabic_transcribe_deidentify\output\V8_output_transcription.txt"
```

# 3. Transcribe and deidentify one single audio file in one run <kbd>TO BE DONE</kbd>

```bash
python run_audio_to_deid_single.py --input "C:\Users\lm2445\arabic\V8.wav"

```

# 4. Run all the .mp3 audio files <kbd>TO BE DONE</kbd>

## Data Folder Structure

The audio data are organized in a date-based folder structure.  
All call recordings (`.mp3`) are stored under daily subfolders.

```
Calls/
├─ 20210609/
│  ├─ 1623239388485_4104_667_3.mp3
│  ├─ 1623240123456_4104_667_4.mp3
│  └─ ...
├─ 20210610/
│  ├─ 1623325xxxxx_...._...._12.mp3
│  └─ ...
├─ 20210611/
│  └─ ...
└─ ...
```

Related metadata files:

```
Dataset_Root/
├─ Calls/
│  └─ (YYYYMMDD folders with .mp3 files)
├─ output_with_filenames.xlsx
└─ Matched_data_Embrace.xlsx
```

## run:

```bash
python .\run_calls_tree_to_deid.py --calls_root "C:\Users\lm2445\arabic\Calls" --out_root ".\output_all" --continue_on_error --format ".mp3"
```


<!-- ## 2.3 Or you can run the whole thing:

```bash
python transcribe_chunk_plus_deidentify.py
```
It can transcribe and deidentify all the audio file of specific types in a specific folder. You need to set the type (wav, mp3, etc) and the folder path in transcribe_chunk_plus_deidentify.py -->
