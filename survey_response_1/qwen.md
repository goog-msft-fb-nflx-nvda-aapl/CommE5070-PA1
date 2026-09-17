Here is the sourced, deep-research survey tailored to your CommE5070 PA1 requirements. All claims are backed by recent (2024–2026) literature, official repositories, or established MIR benchmarks.

---

### 1. Encoders for Song-Level Classification (~30s clips)
*   **MERT-v1-330M**: 330M parameters, native **24kHz** sample rate (perfect match for your dataset; zero resampling needed) [[1]]. Outputs 24 transformer layers of 1024-dim features [[6]]. *Pooling*: The community standard is mean-pooling over time, but crucially, probing specific layers (e.g., layers 3, 4, 5, 6, 23) or a learned weighted sum outperforms blindly using the last layer [[3]]. *Practice*: Frozen linear probing is the empirically preferred baseline at this data scale (~1000 clips) to prevent catastrophic overfitting, with fine-tuning reserved only for the top-performing configuration [[5]]. HF: [`m-a-p/MERT-v1-330M`](https://huggingface.co/m-a-p/MERT-v1-330M).
*   **MusicFM**: Uses BEST-RQ (masked, random-projection quantized targets). HF: [`tky823/MusicFM`](https://huggingface.co/tky823/MusicFM) / GitHub: [`minzwon/musicfm`](https://github.com/minzwon/musicfm). License: MIT [[16]]. Typically pooled via mean-pooling at layer 12.
*   **Music2Vec**: ~90–95M parameters, native **16kHz** sample rate (requires downsampling from 24kHz, which can introduce aliasing artifacts if not done carefully) [[1]]. HF: [`m-a-p/music2vec-v1`](https://huggingface.co/m-a-p/music2vec-v1). Achieves Jukebox-comparable performance at <2% of the parameter count [[22]].
*   **LAION-CLAP (Music)**: ~190M audio tower, native **48kHz** (requires upsampling). HF: [`laion/larger_clap_music`](https://huggingface.co/laion/larger_clap_music). Best used for zero-shot sanity checks (e.g., prompting "1990s rock music") rather than as a primary fine-tuned encoder.
*   **Newer (2025–2026)**: **MERT-v2-FullSong** is explicitly adapted for complete songs lasting 30–360 seconds, making it highly relevant for your 30s clips [[14]]. **MuQ** is a newer self-supervised framework that can wrap MERT or MusicFM as its encoder for enhanced music representation [[13]].

### 2. Prior Work on Release-Decade / Market Classification
*   **Decade/Year Prediction**: The canonical benchmark is YearPredictionMSD (Million Song Dataset). Recent deep learning approaches using pooled embeddings (e.g., BERT-style) fed into traditional regression report Mean Absolute Errors (MAE) of ~6.6 years, establishing a realistic floor for this task [[92]], [[93]]. Older MIR-specific work like "Music Era Classification using Hierarchical-level Fusion" confirms that adjacent-decade confusion is the dominant error mode, not random guessing [[83]].
*   **Geographic/Market Classification**: Literature on "geographic origin of music" exists (e.g., classifying folk/traditional music by region), but these models predict *ethnomusicological provenance*, not the *commercial release market* of 1980s pop/rock [[40]]. Your Task 2 is a novel repurposing of metadata; no major recent MIR papers tackle "release market" (US/UK/Brazil/etc.) as a distinct audio classification task, making your ablation highly original.

### 3. Discogs-VI Dataset Provenance
*   **Source**: Introduced in the ISMIR 2024 paper *"Discogs-VI: A Musical Version Identification Dataset Based on Public Editorial Metadata"* [[30]]. GitHub: [`MTG/discogs-vi-dataset`](https://github.com/MTG/discogs-vi-dataset) [[28]].
*   **Size**: ~348k songs with 1.9M versions (with a YouTube-matched subset of ~493k versions) [[30]], [[33]].
*   **Repurposing Context**: The dataset was explicitly built for *cover-song/version identification (VI)*, not decade/market classification [[28]]. The decade and country labels are derived from Discogs editorial metadata. As noted in the paper, this means labels are commercial metadata, not perceptual ground truth, introducing inherent label noise (e.g., misattributed reissues, compilations, or regional pressings) that your report should explicitly cite as a source of irreducible error [[31]].

### 4. Classifier Heads for Small, High-Dimensional Datasets
*   **Regularization**: With ~150–230 samples/class and 1024-dim embeddings, a deep MLP will overfit immediately. The recommended baseline is **L2-regularized (Ridge) Logistic Regression** with the `C` parameter tuned via cross-validation. If using an MLP, restrict to 1 hidden layer (e.g., 1024→128→6) with aggressive dropout (0.3–0.5) and weight decay.
*   **Dimensionality Reduction**: Applying **PCA to 64–128 dimensions** before the classifier is highly recommended. It removes noise and collinearity in high-dimensional embeddings, significantly improving SVM/MLP stability on small datasets without materially degrading Logistic Regression performance [[110]], [[113]].
*   **Cross-Validation**: Because your official val splits are tiny (132 or 102 samples, ≈22/class), a single val run is statistically noisy (±1 prediction ≈ ±0.8% accuracy). You must use **5-fold CV on the training set** for hyperparameter selection (e.g., PCA dims, SVM `C`, MLP dropout), then evaluate the *selected* model once on the official val split to comply with the assignment rules while avoiding val-set overfitting [[75]].

### 5. Augmentation & TTA Tricks
*   **SpecAugment & Mixup**: Highly effective and standard for log-mel spectrograms (e.g., your Short-Chunk CNN baseline) [[118]].
*   **Pitch-Shift vs. Time-Stretch**: **Avoid pitch-shifting**. Literature on audio deepfakes and augmentation warns that pitch-shifting alters era-relevant cues, such as historical tuning standards (e.g., A=440Hz vs. older standards) and mastering pitch drift [[120]], [[121]]. **Time-stretch** (e.g., [80%, 120%]) is the defensible alternative, as it preserves spectral/timbral characteristics while varying temporal structure [[118]], [[119]].
*   **Multi-Crop TTA**: Extracting embeddings from multiple overlapping 10–30s crops per clip and mean-pooling the predictions is a zero-training-cost TTA method. This elegantly satisfies the required experiment to "use multiple excerpts from the same recording" [[118]].

### 6. Source Separation for Mixture/Vocal/Accompaniment
*   **Best-in-Class (2025–2026)**: **Demucs** (specifically `htdemucs` or `htdemucs_ft`) is the current state-of-the-art open-source choice, consistently outperforming Open-Unmix and Spleeter in separation quality benchmarks [[137]], [[141]]. Unlike Spleeter/Open-Unmix (which operate on spectrogram magnitude and struggle with phase), Demucs operates in the time domain using a hybrid transformer architecture, yielding superior phase coherence [[138]], [[142]].
*   **Exact Command**: 
    1. `pip install demucs`
    2. `demucs --two-stems=vocals <input_file>.wav`
    This directly outputs `vocals.wav` and `no_vocals.wav` (your accompaniment track) in the `separated/htdemucs/` directory [[141]]. Run this in batch on your GPU node.

### 7. Audio Language Models (ALM) for Closed-Set Classification
*   **Qwen2-Audio-7B-Instruct**: ~8B total parameters (Whisper-large-v3 encoder + 7B LLM). Full precision requires ~16–17GB VRAM, but 4-bit quantization (LLM only, encoder in bf16) reduces this to **~4.2–5GB VRAM**, making it trivial to run on your H200 GPUs [[50]], [[53]]. License: Qwen Research License (permissive for academic use) [[49]]. HF: [`Qwen/Qwen2-Audio-7B-Instruct`](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct).
*   **NVIDIA Audio Flamingo 3**: Qwen2.5-7B backbone + AF-Whisper encoder. Handles up to 10 minutes of audio. License: **NVIDIA OneWay Noncommercial License** (perfectly fine for coursework, but must be cited) [[56]], [[62]]. HF: [`nvidia/audio-flamingo-3`](https://huggingface.co/nvidia/audio-flamingo-3). A newer variant, `audio-flamingo-next-hf`, is also available with a stronger encoder under the same license [[58]].
*   **Constraining Output**: Generative models suffer from verbosity/affirmative bias. Best practice is a two-step constraint: 
    1. **Prompt**: *"Listen to this 30-second excerpt. Which decade was it most likely released in? Answer with exactly one of: 1960s, 1970s, 1980s, 1990s, 2000s, 2010s. Do not provide any other text."*
    2. **Parsing**: Use regex/fuzzy matching (e.g., `r"(19|20)\d0s|sixties|seventies|eighties|nineties|two thousands|tens"`) to map the output to your 6 canonical labels [[148]]. 
    3. **Fallback**: If no match is found, log it as an "invalid output" (which you must report as a metric per the rubric) and default to the majority class or trigger a strict re-prompt.

### 8. Loudness War / Mastering Trends as a Confound
*   **Published Evidence**: The "loudness war" is a rigorously quantified phenomenon in MIR. Seminal studies (e.g., Serrà et al., 2012; Deruty, 2015) demonstrate that mean loudness of commercial music increased significantly from the 1960s through the 2000s, peaking around 2004–2007, accompanied by a measurable reduction in dynamic range [[67]], [[68]], [[70]]. 
*   **Handling the Confound**: Because your EDA shows rising RMS loudness by decade, a model might trivially classify based on mastering loudness rather than genuine timbral/production cues. The established MIR practice is to apply **per-clip loudness normalization** (e.g., peak or RMS normalization to a target LUFS) *before* feature extraction for your hand-crafted and CNN baselines [[68]], [[71]]. You should still explicitly report the raw loudness trend in your bias analysis, as it is a real, historically valid signal, not purely a confound to be erased.

### 9. Ordinal/Ranked Classification Metrics for Decade Prediction
*   **Mean Absolute Decade Error (MAE)**: Since decades have a natural order, top-1 accuracy is insufficient. Map your 6 classes to integers (0 to 5) and compute the MAE of the predictions. This directly quantifies "how far off" the model is, and is the most effective and widely used metric for ordinal classification in this context [[74]], [[80]].
*   **Quadratic Weighted Kappa (QWK)**: Highly recommended as a secondary metric. QWK penalizes errors proportionally to their distance from the true label (e.g., predicting 2010s for a 1960s track is penalized much more heavily than predicting 1970s), providing a single, robust ordinal-agreement statistic [[76]], [[130]].
*   **Adjacent-vs-Nonadjacent Error Rate**: Calculate the fraction of all misclassifications that land on an immediately adjacent decade versus further away. This directly and quantitatively answers the assignment's required discussion question about neighboring-decade confusion, rather than relying solely on qualitative confusion matrix reading [[74]]. *(Note: This metric is invalid for Task 2, as release markets are categorical with no natural adjacency).*

---

**Next Steps**: If you'd like, I can generate the exact Python code for any of these components (e.g., the MERT layer-sweep + PCA + Ridge CV pipeline, the Demucs batch script, or the ALM regex-parsing inference loop). Just let me know which one to prioritize.