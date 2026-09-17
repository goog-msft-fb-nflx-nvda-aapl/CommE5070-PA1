## 1. Encoders for Song-Level Classification (~30s Clips)

Open-source pretrained audio and music foundation models vary in architecture, native sample rate, and pooling strategies suitable for frozen feature extraction (linear probing) or fine-tuning:

* **MERT (v1-330M)**
* **Repository / Model ID:** `m-a-p/MERT-v1-330M` (also `m-a-p/MERT-v1-95M`)
* **License:** CC BY-NC 4.0
* **Params & Embedding Dim:** 330M parameters, 1024-dim outputs across 24 Transformer layers.
* **Native Sample Rate:** 24 kHz (matches your audio natively without resampling). Frame rate is 75 Hz (1 feature vector per 13.3ms).
* **Pooling Strategy:** Layer-weighted sum followed by temporal global mean pooling or max pooling. The authors recommend taking a learnable weighted combination of all 24 layer representations before temporal pooling, as lower/intermediate layers capture timbral/production cues while deeper layers capture high-level semantics.
* **Probing vs. Fine-tuning:** With ~170–230 clips/class, **frozen linear probing** (Linear/Ridge/SVM on pooled embeddings) is standard. Fine-tuning the full 330M model at this scale risks severe overfitting unless using low learning rates ($\sim 10^{-5}$) or LoRA adapter layers.


* **MusicFM**
* **Repository / Model ID:** GitHub `minzwon/musicfm` / HuggingFace `minzwon/MusicFM`
* **License:** MIT
* **Params & Embedding Dim:** ~100M parameters (Conformer backbone), 1024-dim outputs at a 25 Hz frame rate.
* **Native Sample Rate:** 24 kHz / 16 kHz (check model version; operates natively at 25Hz frame rate).
* **Pooling Strategy:** Global average pooling across time or adaptive average pooling to collapse the 25 Hz frame representations into a single 1024-dim clip vector.
* **Probing vs. Fine-tuning:** Linear probing is strongly recommended for datasets under 2,000 samples. Fine-tuning degrades performance on small tagging tasks due to catastrophic forgetting.


* **Music2Vec**
* **Repository / Model ID:** `m-a-p/music2vec-v1`
* **License:** CC BY-NC 4.0
* **Params & Embedding Dim:** 95M parameters, 768-dim output embeddings.
* **Native Sample Rate:** 16 kHz (requires resampling from 24 kHz).
* **Pooling Strategy:** Temporal mean pooling across the output sequence of the final Transformer layer.


* **LAION-CLAP / MS-CLAP**
* **Repository / Model ID:** `laion/clap-htsat-fused` (or `laion/clap-htsat-unfused`), `microsoft/msclap-2023`
* **License:** MIT (LAION) / Microsoft Research License (MS-CLAP)
* **Params & Embedding Dim:** ~110M–150M parameters. Output projection dimension is 512-dim (LAION) or 1024-dim (MS-CLAP).
* **Native Sample Rate:** 48 kHz (LAION) or 44.1 kHz (MS-CLAP); require resampling from 24 kHz.
* **Pooling Strategy:** CLAP architectures internally pool frame-level spectrogram representations via HTS-AT (Hierarchical Token-Semantic Audio Transformer) to produce a native 1D clip-level embedding.
* **Probing vs. Fine-tuning:** Linear probing on the joint audio embedding vector is standard and highly parameter-efficient.


* **Jukebox-Derived Encoders (Jukemir)**
* **Repository / Model ID:** GitHub `kalpeshp256/jukemir` (OpenAI Jukebox 5B/1B backbones)
* **License:** MIT
* **Params & Embedding Dim:** 1B to 5B parameters; extracts 4800-dim representations from layer 36 or 48.
* **Native Sample Rate:** 44.1 kHz (requires resampling).
* **Pooling Strategy:** Mean and max pooling concatenated over time. Highly expressive for timbral/era tasks, but computationally heavy.



---

## 2. Prior Published Work on Era & Market Classification

* **Release-Decade / Year Classification**
* **Wang et al. (2024)**, *"Music Era Recognition Using Supervised Contrastive Learning"* (arXiv:2407.05368): Formulates decade recognition as a supervised classification task on the Million Song Dataset (MSD). Using an audio CNN baseline, accuracy is ~45–54% within a 3-year tolerance window.
* **Year Regression Baselines (MSD)**: Classic MIR literature (e.g., Eerola 2011, Schuller et al. 2011) evaluated release year prediction using Echo Nest timbral features. Standard performance reaches a Mean Absolute Error (MAE) of **7.5 to 9.5 years**.
* **Error Distributions**: Literature consistently notes that prediction errors cluster in **adjacent decades** (e.g., misclassifying a 1980s track as 1970s or 1990s) due to continuous evolutionary shifts in instrumentation, mastering, and production style.


* **Geographic / Release-Market Classification**
* **Kedyte et al. (2017)**, *"Geographical Origin Prediction of Folk Music"* (ISMIR) & **Gomez et al.**: Handled geographic origin as regression/classification (predicting latitude/longitude or country) for traditional/folk music.
* **Commercial Release Market (Discogs Category)**: Published audio MIR work explicitly classifying *commercial release markets* (e.g., US vs. UK vs. Brazil pressings in the 1980s) is scarce. Most MIR geographic tasks focus on traditional folk music or ethnomusicology rather than commercial release metadata. Release market differences in popular music stem primarily from regional mastering practices, local vinyl/CD pressing standards, dynamic range choices, and broadcast equalization targets.



---

## 3. Discogs-VI Dataset Provenance & Metadata Repurposing

* **Provenance & Citation:** Created by the Music Technology Group (MTG) at Universitat Pompeu Fabra (Araz et al., ISMIR 2024; GitHub: `MTG/discogs-vi-dataset`, model repo: `raraz15/Discogs-VINet`).
* **Original Purpose:** Constructed specifically for Cover Song Identification (CSI) / Version Identification (VI), grouping tracks into release "cliques".
* **Dataset Size:** ~1.9 million versions in 348,000 cliques (Discogs-VI-YT subset contains ~493,000 versions).
* **Metadata Repurposing:** Repurposing Discogs-VI editorial metadata (the `country` and `year` fields) for single-label era and release-market audio classification is a novel task setup introduced in this coursework. Utilizing Discogs metadata avoids the label noise typical of crowd-sourced streaming tags, though it reflects *commercial distribution entity* rather than recording location or artist nationality.

---

## 4. Classifier Heads & Recipes for High-Dimensional, Small-N Datasets

When training classifiers on 768–1024 dimensional embeddings with $N \approx 798\text{--}1026$ samples (~133–230 per class):

* **Classifier Architectures**
* **Linear Probing (Logistic Regression / Ridge Classifier):** Standard $L_2$-regularized Logistic Regression ($C \in [10^{-3}, 10^1]$) or Ridge Classifier. Highly robust against overfitting when $D \approx N$.
* **Support Vector Machine (SVM):** RBF-kernel or Linear SVC with grid-searched hyperparameter $C$ and $\gamma = \text{'scale'}$.
* **MLP Head:** 2-layer MLP ($\text{Input} \rightarrow 256 \rightarrow \text{Output}$) with high Dropout ($p = 0.5$), Layer Normalization, and AdamW optimizer with heavy Weight Decay ($10^{-2}$).


* **Cross-Validation Strategy**
* **GroupKFold (Stratified):** Must group by `artist_id` to enforce artist-disjoint splits across folds. Failing to group by artist leads to data leakage via artist-specific production styles.


* **Dimensionality Reduction & Regularization**
* **Principal Component Analysis (PCA):** Fit PCA on the training split to retain 95%–99% variance. This typically reduces 1024-dim MERT/MusicFM embeddings to ~50–150 dimensions, stabilizing downstream MLP classifiers.



---

## 5. Augmentations & Test-Time Augmentation (TTA)

* **Safe Audio Augmentations**
* **SpecAugment:** Frequency masking (1–2 channels, max width 8–16 bins) and time masking (1–2 blocks) applied to spectrograms or intermediate feature maps.
* **Mixup ($\alpha = 0.2$):** Linear interpolation of log-mel spectrograms and standard one-hot labels.
* **Additive Noise / Gain Scaling:** Injecting low-level Gaussian noise ($\text{SNR} = 20\text{--}30\text{ dB}$) or random gain shifts ($\pm 3\text{ dB}$) to simulate varying transfer media quality.


* **Flagged / Dangerous Augmentations**
* **Pitch-Shifting:** **Avoid or use extreme caution.** Pitch-shifting alters master tuning, key, and vocal formants. In decade/market classification, historical tuning standards, tape speed variances, and vocal timbres are core discriminative features.
* **Time-Stretching:** Can introduce phase artifacts and alter transient sharpness, distorting production characteristics characteristic of specific eras (e.g., 1980s gated reverb).


* **Test-Time Augmentation (TTA) / Multi-Segment Averaging**
* Extract $K$ overlapping clips (e.g., 3.7s, 5s, 10s, or 15s) across the 30-second recording using a sliding window with 50% overlap.
* Compute softmax probability distributions for each clip via the trained model and average the predictions: $P(\text{song}) = \frac{1}{K} \sum_{k=1}^K P(\text{clip}_k)$.



---

## 6. Source Separation (Demucs v4)

**Demucs v4 (Hybrid Transformer Demucs / `htdemucs`)** by Meta AI is the current best-in-class, easy-to-install open-source stem separator.

* **Installation Command:**
```bash
pip install -U demucs

```


* **2-Stem Separation Command (Vocals vs. Accompaniment):**
```bash
demucs --two-stems vocals track.wav

```


* This outputs two WAV files: `vocals.wav` (vocal stem) and `no_vocals.wav` (accompaniment stem) in `./separated/htdemucs/track/`.



---

## 7. Audio Language Models (ALMs) for Closed-Set Classification

* **Qwen2-Audio-7B-Instruct**
* **Model ID:** `Qwen/Qwen2-Audio-7B-Instruct`
* **License:** Apache 2.0 (permissive, commercial use allowed).
* **VRAM Requirements:** ~14–16 GB in FP16/BF16. Easily fits on a single H200 NVL GPU (141 GB VRAM).


* **NVIDIA Audio Flamingo 3 / Music Flamingo**
* **Repository:** GitHub `NVIDIA/audio-flamingo`
* **License:** MIT for code; NVIDIA OneWay Non-Commercial License for model weights (built on Qwen2.5-7B backbone).
* **VRAM Requirements:** ~16–20 GB in FP16.
* **SOTA Status:** Audio Flamingo 3 (NeurIPS 2025 Spotlight) and its music-specialized variant **Music Flamingo** set state-of-the-art benchmarks on open-source audio/music understanding.


* **Constraining ALMs for Fixed 6-Way Classification**
* **Prompt Design:**
```text
System: You are a music metadata classification assistant.
User: <audio> Analyze this 30-second audio excerpt. Classify the release decade of this track. 
Choice options: [1960s, 1970s, 1980s, 1990s, 2000s, 2010s]. 
Respond ONLY with one of the exact strings listed above. Do not include punctuation, explanations, or extra words.

```


* **Logit Constraining:** Use HuggingFace `PrefixConstrainedLogitsProcessor` or set custom logit bias so the LLM generation head is physically restricted to sampling tokens corresponding to the 6 allowed class strings.
* **Parsing & Invalid Output Handling:**
* Use regular expressions (`r'\b(1960s|1970s|1980s|1990s|2000s|2010s)\b'`) or Levenshtein fuzzy string matching to map raw responses to valid labels.
* Unconstrained ALMs exhibit invalid output rates of 1% to 8% (e.g., generating conversational filler like "The release decade is 1980s."). Constrained decoding reduces invalid output to 0%.





---

## 8. Loudness War Trends Across Decades & Confound Mitigation

* **Documented Phenomenon:** Serrà et al. (2012), *"Measuring the Evolution of Contemporary Western Popular Music"* (*Scientific Reports*), analyzed 464,411 tracks (1955–2010). They formally quantified a continuous multi-decadal rise in average RMS volume and a corresponding decrease in dynamic range (the "loudness war").
* **Historical Loudness Metrics:**
* **1960s–1970s:** Average RMS levels $\sim -18\text{ to }-16\text{ dBFS}$ ($\sim -18\text{ LUFS}$).
* **1990s–2000s:** Peak loudness war; RMS levels climbed to $-8\text{ to }-6\text{ dBFS}$ ($\sim -9\text{ to }-7\text{ LUFS}$).
* **2010s:** Normalized streaming standards (Spotify/Apple Music targeting $-14\text{ to }-16\text{ LUFS}$) caused loudness levels to plateau.


* **Classification Confound:** Classifiers trained on raw audio can exploit global RMS energy as a shortcut to classify decades, ignoring timbral or musical content.
* **Mitigation Strategy:** Preprocess all input audio with **ITU-R BS.1770-4 loudness normalization** (e.g., using `pyloudnorm` in Python) to normalize all clips to a constant target loudness (e.g., $-24.0\text{ LUFS}$) before computing log-mel spectrograms or passing audio into foundation encoders.

---

## 9. Ordinal & Ranked Metrics for Decade Classification

Since release decades possess a natural chronological order, standard accuracy treats an error of 1 decade (predicting 1970s for a 1980s clip) identically to an error of 5 decades (predicting 1960s for a 2010s clip). MIR literature applies specific ordinal metrics:

* **Mean Absolute Error (MAE):**

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N \vert{}y_i - \hat{y}_i\vert{}$$



Expressed either in years (e.g., $\text{MAE} = 8.2\text{ years}$) or in decade steps ($\vert{}y_i - \hat{y}_i\vert{} \in \{0, 1, 2, 3, 4, 5\}$).
* **Adjacent Class Accuracy ($\text{Acc}_{\pm 1}$):**
Measures the percentage of predictions within $\pm 1$ decade of the ground truth label:

$$\text{Acc}_{\pm 1} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\vert{}y_i - \hat{y}_i\vert{} \le 1)$$


* **Quadratic Weighted Kappa ($\kappa_w$):**
Evaluates agreement between true and predicted ordinal categories while penalizing larger distance errors quadratically:

$$w_{i,j} = \frac{(i - j)^2}{(N_{\text{classes}} - 1)^2}$$


* **Earth Mover's Distance (EMD) / Wasserstein Loss:**
Measures the distance between predicted softmax probability distributions and ordinal target distributions, providing a continuous ordinal loss function during model training.