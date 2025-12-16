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

## 2.1 modify the input audio path in transcribe.py, then

```bash
python transcribe.py
```
### 2.1.1 if the audio is too long, use the chunk version

```bash
python transcribe_chunk.py
```

## 2.2 modify the input text path of deidentify.py, then

```bash
python deidentify.py
```
