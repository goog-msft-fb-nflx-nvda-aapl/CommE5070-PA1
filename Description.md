# Homework 1: Music Era and Release-Market Classification

---

## Outline

- Overview
- Detailed Explanation
- Submission
- Scoring
- Rules
- Timeline

---

## Overview: Single-label Classification

- model input: 30-second excerpt
- model output:
  - One label per excerpt
  - **Top-1:** highest score
  - **Top-3:** three highest scores
  - Same prediction rule for A and B
  - Six class scores

---

## Detailed Explanation

- **Dataset**
  - Task 1: Release-decade classification
  - Task 2: Release-market classification
- Suggested methods
- Evaluation
- Optional experiments

---

## Dataset

- **Source:** Discogs-VI recordings with an available YouTube link
  - **Dataset A:** 1,290 recordings from six US release decades
  - **Dataset B:** 1,002 recordings from six release markets in the 1980s
- Each example is the middle 30 seconds of one recording
- **Audio:** WAV, mono, PCM signed 16-bit, 24 kHz
- Anonymous sample IDs and stripped title and artist metadata
- Artists do not overlap across train, validation, and test
- **Link:** [Google Drive](https://drive.google.com/drive/folders/1C8RymiLbr-EGmkxh2Ap5TIybnJYqNsb4?usp=sharing)
% pwd
/Users/chun-feitan/Desktop/CommE5070/PA1
% tree dataset
dataset
├── dataset_A.zip
├── dataset_B.zip
└── prediction_format_example_NOT_ANSWERS.json

---

## Task 1: Release-Decade Classification

[Audio Signal (Waveform)] → [Feature Extraction (Encoder)] → [Feature Vector / Embedding] → [Classifier] → [Decade Labels]

**Pipeline Breakdown:**

1. **Input:** Audio Waveform Signal
2. **Feature Extraction (Encoder):**
* **Baseline:**
* 1. librosa: https://librosa.org/doc/latest/index.html
* 2. torchaudio: https://docs.pytorch.org/audio/stable/index.html

3. **Representation:** Feature / Embedding Vector
4. **Classifier:**
* **Baseline:**
* 1. scikit-learn: https://scikit-learn.org/stable/
* 2. MERT: https://huggingface.co/m-a-p/MERT-v1-330M




5. **Output (Classes):**
* 1960s
* 1970s
* 1980s
* 1990s
* 2000s
* 2010s



---

**Dataset A:** US releases, six balanced decade labels, 1,026 / 132 / 132 split

### What You Need to Do

- Train a model with any features extracted from the audio.
- Need to report the features you use and the model implementation clearly.
- Need to report the validation result with **confusion matrix**, **top-1 accuracy**, and **top-3 accuracy**.
- Remember to utilize **standardization** (e.g., mean, std), **pooling**, and **normalization** to ensure consistent feature scales, reducing overfitting, and improving model stability and performance during training.
- Discuss whether errors occur mainly between neighboring decades.
  - The report must explain what the model learned and where neighboring decades are confused.

---

## Task 2: Release-Market Classification

[Audio Signal (Waveform)] → [Feature Extraction (Encoder)] → [Feature Vector / Embedding] → [Classifier] → [Market Labels]

**Pipeline Breakdown:**

1. **Input:** Audio Waveform Signal
2. **Feature Extraction (Encoder):**
* **Baseline:**
* 1. librosa: https://librosa.org/doc/latest/index.html
* 2. torchaudio: https://docs.pytorch.org/audio/stable/index.html





3. **Representation:** Feature / Embedding Vector
4. **Classifier:**
* 1. scikit-learn: https://scikit-learn.org/stable/
* 2. MERT: https://huggingface.co/m-a-p/MERT-v1-330M





5. **Output (Classes):**
* US
* UK
* Brazil
* Spain
* Germany
* Italy



---

**Dataset B:** 1980s releases, six balanced market labels, 798 / 102 / 102 split

*Release market is not the artist's nationality, language, ethnicity, or recording location.*

### What You Need to Do

- Train a model with any features extracted from the audio.
- Need to report the features you use and the model implementation clearly.
- Need to report the validation result with **confusion matrix**, **top-1 accuracy**, and **top-3 accuracy**.
- Remember to utilize **standardization** (e.g., mean, std), **pooling**, and **normalization** to ensure consistent feature scales, reducing overfitting, and improving model stability and performance during training.
- Discuss whether errors occur mainly between release markets.

---

## Baseline Method 1: Short-Chunk CNN

- Convert the waveform into a log-mel spectrogram
- During training, sample a short segment from the 30-second excerpt
- Use a convolutional network to predict six class scores
- During evaluation, combine predictions from multiple segments

audio --> log-mel spectrogram --> CNN --> class probabilities

## Baseline Method 2: Pretrained Audio Features

- Use MERT or another pretrained audio encoder
- Pool frame-level representations into one recording-level feature
- Train a classifier such as logistic regression, SVM, or MLP
- Compare frozen features with fine-tuning if resources allow

audio --> pretrained encoder --> pooling --> classifier

---

## Evaluation

1. **Confusion matrix**
   - Compare predicted labels with the true labels.
   - State whether the matrix shows counts or normalized values.

2. **Top-k accuracy**
   - Report **Top-1** and **Top-3**.
   - Top-3 counts a sample as correct when the true label appears among the three submitted labels.

> Use **train** for fitting, **validation** for model selection, and **test** for final reporting.

---

## Data Pre-processing / Data Augmentation

### Audio Processing

| Library    | URL | 
|------------|------------|
| librosa    | https://librosa.org/doc/latest/index.html |
| torchaudio | https://docs.pytorch.org/audio/stable/index.html | 

### Source Separation

- Open-Unmix: https://github.com/sigsep/open-unmix-pytorch
- Spleeter: https://github.com/deezer/spleeter
- Demucs: https://github.com/adefossez/demucs

### Guidelines

- Preserve cues that may carry decade or market information
- Avoid transformations that change the label meaning
- Compare the original mixture with vocal and accompaniment stems
- Report every preprocessing and augmentation step

---

## Required Experiments

- Compare 5, 10, 15, and 30 seconds of audio
- Use multiple excerpts from the same recording during training
- Treat Task 1 as year regression or hierarchical prediction
- Compare mixture, vocal, and accompaniment inputs for Task 2
- Visualize learned embeddings with t-SNE or UMAP
- Analyze confusion patterns and dataset bias

### Audio Language Model

- Run an audio language model on both Dataset A and Dataset B
- Use every sample in the selected split and return one valid answer for every sample
- Describe the prompt and output parsing method
- Report Top-1, Top-3, confusion matrices, and invalid-output handling
- Compare at least two prompt designs or explain why you selected one prompt
- **Models:** 
  - Qwen2-Audio: https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct
  - Audio Flamingo: https://huggingface.co/nvidia/audio-flamingo-3

---

## Submission

- **Report HTML** to NTU COOL
- **Prediction JSON** to NTU COOL
- **Source code and trained checkpoint** in an open-access cloud-drive folder
- **requirements.txt** and **README.md** in the same folder

> We may select submissions and run inference. The uploaded files must reproduce the submitted predictions.
> **Do not** upload the released dataset, model caches, or unrelated checkpoints.

### Report

- Use a 16:9 PPT or PPT-like format and submit it as HTML
- **File name:** `<studentID>_report.html`
- The report must be understandable without oral explanation
- Explain methods and results for both tasks
- Include at least one confusion matrix for each task
- Discuss errors, limitations, and reproducibility
- Cite every public codebase, pretrained model, and paper used

### Prediction

- **File name:** `<studentID>.json`
- Include every Dataset A and Dataset B test sample exactly once
- Return top-3 labels in descending confidence order

### Code & Model Checkpoint

* Upload all your source code and model to a cloud drive, open access permissions, and then upload the link to the NTU Cool assignment HW1_report in comments, as well as include it on the first page of the report
* You will need to upload requirements.txt
* I'll run :

```bash
pip install -r requirements.txt

```

If you have used third-party programs that cannot be installed directly via 'pip install,' please write the URL and install method command by command on **your readme file**.

---

**Example `requirements.txt`:**

```text
numpy
torch
torchaudio
tqdm
scikit-learn
matplotlib
joblib

```

> Please don't ask me to install libraries that are not needed for the inference process...

- You will also need to upload a **readme file** to guide the TA on how to perform inference on your model (the same test set will be used; ensure the code can run with the test set path on the TA's device)
- The inference code should output the **top 3 predictions** in descending accuracy order

---

## Scoring

| Component | Weight |
|-----------|--------|
| Report    | 50%    |
| Accuracy  | 50%    |

- **A−:** All basic requirements are completed properly.
- **A:** The work goes beyond the basic requirements in implementation, performance, or analysis.
- **A+:** A creative approach with exceptionally strong implementation and accuracy may receive an A+.

**Class Sharing:** Three students with representative submissions will be invited to present their work in class.

For each task:
S= (Top-1 + 0.5 x Top-3)
Accuracy points = 50 x (S_A + S_B) / 2

---

## Rules
- Do not use additional labeled audio unless the course explicitly permits it
- Do not train or tune on validation or test examples
- Do not hard-code predictions from the released labels
- You may use public code and pretrained models with proper citation
- Your uploaded code and checkpoint must reproduce the submitted predictions

---

### Late Submission Policy

| Days Late | Penalty |
|-----------|---------|
| 1 day     | -20%    |
| 2 days    | -40%    |
| 3 days    | -60%    |
| After 3 days | No score |

### All Things You Need to Submit Before 10/05 23:59

- `StudentID_report.pdf`
- Cloud drive link containing:
  - `readme` file
  - `requirements.txt`
  - Codes and model to run inference
  - Other codes
- Test set prediction


### env ###
 - GPU server can be access via ssh gsm-gpu2
 - there is no sudo permission on the GPU server
 - use tmux and conda to run experiment and do env mgmt.
 - only edit code data under jtan@TO-sv-td-h200nvlnode02:~$ pwd
/home/jtan do not access/edit other user folders or shared folders.
 - GPU 0 is malfunctioning so please use GPU 1 - 3 ( via CUDA_VISIBLE_DEVICES=... etc)

### general rule ###
1. no need to say anything in chat unless you want to confirm/ask anything.
2. create a markdown/text (easy for you to track) experiement log and the ablation to do, etc.
3. experiment on GPU, work log on Mac.
4. if you need access token of GitHub, HuggingFace, just let me know. I will provide.

 ### initial steps ###
 1. read the Description.md
 2. read the two lecture notes related to this homework (some fundamentals)
    lecture02_classification.md	
    lecture02b_transcription.md
 3. do preliminary EDA on the dataset.
 4. 
 create deep research survey on state of the art method (models, data set, libraries, etc) and their availability on Github, HuggingFace, etc; what else EDA do we need to do before the training and post traininng; any tricks (TTA, etc.)
 5. once the survey result is acquired, we will start implementing.
 6. for all the baseline methods, we must implement them and do the ablations (different combination of baseline encoder x classifier) and report the result.