# Music Transcription (audio → scores)
### Deep Learning for Music Analysis and Generation — 2026 edition
**Yi-Hsuan Yang, Ph.D.** — yhyangtw@ntu.edu.tw

---

## Slide 2 — Music AI; or *Music Information Research* (MIR)

**Diagram:** Two parallel pipelines branch from an `audio` input.

- **Music analysis** (left, highlighted): `audio → labels` and `audio → score`
  - **Music semantic labeling**: audio → genre (classical), audio → emotion (yearning), audio → other attributes (slow/fast)
  - **Music transcription (audio2score)**: audio → note (pitch, onset, offset), audio → instrument (flute, cello), audio → meter (4/4), audio → key (E-flat major)
  - Applications: music understanding, music search, music recommendation

- **Music generation** (right, greyed out): `random seed + labels → AI composer → score → AI performer (score2audio) → audio (new song)`
  - Sub-areas: MIDI generation, audio generation, MIDI-to-audio generation

---

## Slide 3 — Reference 1: FMP Notebook
https://www.audiolabs-erlangen.de/resources/MIR/FMP/C1/C1.html

Table of notebook parts (each with HTML/IPYNB links):

| Part | Title | Notions, Techniques & Algorithms |
|---|---|---|
| B | Basics | Basic Python/Jupyter/Anaconda info, environments, visualizations |
| 0 | Overview | Overview of the notebooks |
| 1 | Music Representations | Music notation, MIDI, audio signal, waveform, pitch, loudness, timbre |
| 2 | Fourier Analysis of Signals | Discrete/analog signal, sinusoid, exponential, Fourier transform, DFT, FFT, STFT |
| **3** | **Music Synchronization** *(highlighted)* | Chroma feature, dynamic programming, dynamic time warping (DTW), alignment, user interface |
| 4 | Music Structure Analysis | Similarity matrix, repetition, thumbnail, homogeneity, novelty, evaluation, precision/recall, F-measure, scape plot |
| **5** | **Chord Recognition** *(highlighted)* | Harmony, music theory, chords, scales, templates, HMM, evaluation |
| 6 | Tempo and Beat Tracking | Onset, novelty, tempo, tempogram, beat, periodicity, Fourier analysis, autocorrelation |
| 7 | Content-Based Audio Retrieval | Identification, fingerprint, indexing, inverted list, matching, version/cover song |
| 8 | Musically Informed Audio Decomposition | Harmonic/percussive separation, signal reconstruction, instantaneous frequency (F0), trajectory, NMF |

---

## Slide 4 — Reference 2: ISMIR 2018 & 2021 Tutorials
https://rachelbittner.weebly.com/other-resources.html

**Programming MIR Baselines from Scratch: Three Case Studies** (ISMIR 2021)
- Part 1: Transcription with NMF (Ethan Manilow)
- Part 2: Pitch Tracking with pytorch (Rachel Bittner)
- Part 3: Instrument Classification with OpenL3 & Tensorflow (Mark Cartwright)
- Recording available online

**Fundamental Frequency Estimation in Music** (ISMIR 2018)
- Part 1: Pitch (Alain de Cheveigné)
- Part 2: Polyphonic fundamental frequency estimation (Rachel Bittner)
- Part 3: Applications (Johana Devaney)

---

## Slide 5–6 — Monophonic F0 Estimation
*(Part 1: Pitch — Alain de Cheveigné)*
https://drive.google.com/file/d/1uWCwE03dM0j0Ptp1g5vPsT56nPd0DCW2/view

- Pitch depends on the **fundamental frequency (F0)**, not on waveform shape.
- **Frequency-domain F0 analysis**
  - E.g., return the frequency bin with the highest energy per frame
  - Problems: harmonics can be stronger than the fundamental; missing fundamental, etc.
- **Time-domain F0 analysis** (based on **auto-correlation**) — **YIN (2002)**, **pYIN (2014)**
  - `librosa.pyin(y, *, fmin, fmax, sr=22050, ...)`

**Diagram:** A periodic waveform is "shifted and compared" against itself at increasing time lags to compute the autocorrelation function:

$$A_t(\tau) = \sum_{W} x(t)\,x(t-\tau)$$

where the sum is taken over an integration window, comparing the signal `x(t)` (product) against a time-lagged copy `x(t-τ)`. The resulting autocorrelation curve shows a peak at the lag corresponding to the pitch period (the plot shows a peak near 5 ms lag → pitch).

---

## Slide 7 — F0 Estimation
*(Part 2 — Rachel Bittner)*
https://drive.google.com/file/d/1Jmj6tNBGFMlElEQldgNZ2_YgD17gS6Sv/view

- **Monophonic** input vs **polyphonic** input (e.g., singing solo vs. mixed song)
- **Single F0 (melody)** vs **multiple F0** estimation (e.g., vocal melody extraction)
- **F0 estimation (Hz)** vs **note estimation** (MIDI pitch)
- Time quantization or not
- Output instrument labels or not

**Diagram:** Waveform splits into three example outputs, each a frequency (Hz, log scale 65–1046) vs. time (sec) scatter plot: **Melody**, **Bass**, and **Multiple-f0** (which shows several overlapping pitch contours simultaneously).

---

## Slide 8 — F0 Estimation: Background
https://drive.google.com/file/d/1Jmj6tNBGFMlElEQldgNZ2_YgD17gS6Sv/view

- Related tasks: frame-level F0 estimation, note estimation, streaming, score transcription
- Time resolution
- Frequency resolution
- Voicing
- Instrument labeling

**Diagram:** Two scatter plots of MIDI pitch number (40–80) vs. time (0–6 sec). The left plot shows raw, undifferentiated pitch-vs-time points (black); the right plot shows the same points recolored/grouped by instrument/voice (four colors: blue, green, red, cyan), illustrating how "streaming" separates overlapping note tracks into distinct voices.

---

## Slide 9 — Melody Extraction vs. Note Transcription

- **Melody extraction**: outputs F0 (can reflect *overshoot*, *vibrato*, *glissando*, etc.)
- **Note transcription**: outputs MIDI pitch (*quantized in frequency*)

**Diagram 1** (source: M4Singer paper): Frequency (Hz, 200–480) vs. Time (s, 0–8.7). Two overlaid curves — a continuous, wavy **F0** curve (blue) that fluctuates with vibrato/expressiveness, and a stair-step **Note Pitch** curve (red) that shows the quantized discrete note values.

**Diagram 2** (Saitou et al., "Speech-to-singing synthesis," WASPAA 2007): Fundamental frequency vs. time (ms, 0–5000). Shows a dashed **melody contour (musical score)** — clean step function — vs. a solid **melody contour with F0 fluctuations** — the real sung F0, annotated with four phenomena:
1. **Overshoot** — note attack overshoots then settles
2. **Preparation** — small dip/anticipation before a note change
3. **Vibrato** — oscillation around sustained pitch
4. **Fine fluctuations** — micro-variations throughout

---

## Slide 10 — Pianoroll vs MusicXML

- **Time quantization**: quarter note, eighth note, etc.
- More than that…
- Usually, the target is to output a single-track or multi-track **pianoroll**

**Diagram** (Figure 1.12 from Müller, *FMP*, Springer 2015): Top shows conventional **sheet music notation** (staff with notes); bottom shows the corresponding **pianoroll** (pitch on y-axis, time on x-axis, notes as horizontal bars). Four matching red ovals connect specific note groups in the sheet music to their corresponding bar patterns in the pianoroll, illustrating the audio/score correspondence.

---

## Slide 11 — From an ML/DL Viewpoint

- **Per song**: genre classification
  - Predictions can be made per chunk, then aggregated over time (e.g., averaging logits, or majority voting on chunk-level decisions)
- **Per short-time chunk**: audio event detection, instrument activity detection
  - e.g., output is a matrix **[class × time]**
- **Per time-frequency point**: F0 estimation, multi-pitch estimation
  - e.g., output is a matrix **[frequency × time]**

**Diagram 1:** A heatmap with rows = audio event classes (Speech, Telephone bell ringing, Male speech, Telephone, Inside/small room, Ringtone, Inside/large room or hall, Burping/eructation, Conversation, Narration/monologue) and columns = time (0–7 seconds), colored by detection confidence — illustrating the [class × time] output matrix.

**Diagram 2:** The same Multiple-f0 scatter plot from Slide 7 (Frequency Hz vs. Time sec), illustrating the [frequency × time] output matrix.

---

## Slide 12 — Polyphonic F0 Estimation: Typical Approach
https://drive.google.com/file/d/1Jmj6tNBGFMlElEQldgNZ2_YgD17gS6Sv/view

**Diagram (pipeline flowchart):**

```
Polyphonic Music (waveform)
        ↓
Time Frequency Transform (spectrogram)
        ↓
Model (deep net) ──+
        ↓            \
Optimization Function  \
                         ↓
                Salience Representation (time-frequency salience map)
                         ↓
                Voicing Determination + Decoding
                         ↓
                  f0 Time Series (output: list of time, freq pairs)
```

The optimization function loops back to refine the model; the salience representation feeds a voicing/decoding stage (illustrated with a binary voicing curve and a path-decoding lattice) that produces the final f0 time series output (table of time, freq values).

---

## Slide 13 — Supervised Training
https://drive.google.com/file/d/1Jmj6tNBGFMlElEQldgNZ2_YgD17gS6Sv/view

- Labeled data is hard to collect → slows research progress
  - Single F0 estimation for mono signals is nearly solved
  - Multi-F0 estimation still has room for improvement
  - Note transcription for arbitrary instruments remains hard
    - Exception: **piano** note transcription — nearly solved, partly thanks to high-quality piano synthesizers

**Diagram 1:** Screenshot of an audio annotation tool, showing waveform/spectrogram with note-level annotation markers. Annotated callout: **~50 annotator hours for ~7 hours of music**, illustrating the high cost of manual labeling.

**Diagram 2:** Illustration of a data-collection workflow: a pianist plays a real piano, producing both **real-world audio output** and simultaneous **MIDI output** captured to a laptop, which is used for (1) collecting and playing back real-world audio data and (2) checking note-level annotation ("note following").

---

## Slide 14 — Guitar Transcription: Need to be Invariant to Audio Effects
(Examples provided by Positive Grid)
Chen et al., "Towards automatic transcription of polyphonic electric guitar music: A new dataset and a multi-loss transformer model," ICASSP 2022

**Diagram:** Illustrates a **dry** guitar signal (single-coil electric guitar) being processed through different signal-chain "pedalboard" configurations, each producing a different timbre from the same notes:

- **In the Clouds**: Gate → Mod (Uni-Vibe) → Supreme Clean amp (matched to Jazz Clean) → Delay → Reverb
- **Overdriven Verb Icon**: Gate → Drive → Overdriven Lux Verb amp (matched to Celest V-30s) → Reverb
- **Lazy Down**: Pitch/Filter → '59 Tweed Lux V2 amp → Pitch/Filter (Octaver) → Tweed Lux (reverb) → Delay

This demonstrates why a transcription model must be robust to very different audio effects/timbres applied to the same underlying notes.

---

## Slide 15 — Evaluation Metrics for Pitch-Related Tasks
https://craffel.github.io/mir_eval/

**Single- and multi-pitch estimation**
- Correct if within 0.5 semitones of a reference frequency
- Chroma accuracy (ignores octave)
- Voicing measures

**Piano transcription**
- Onset tolerance window: 50 ms
- Offset tolerance window: 20% of note duration
- Onset only vs. onset+offset

**Chord transcription**
- root, majmin, majmin_inv, thirds, triads, tetrads, sevenths
- overseg, underseg, seg

---

## Slide 16 — ISMIR 2021 Tutorial: Programming MIR Baselines from Scratch – Pitch Tracking
https://github.com/rabitt/ismir-2021-tutorial-case-studies/tree/main/pitch_tracking

- Video online: https://drive.google.com/file/d/18LNaKy2ymFjEWj19gHy25wgtFV4pdML0/view

**Diagram:** Screenshot of a VS Code IDE (Rachel Bittner's tutorial recording) showing a `train.py` file for pitch tracking with classes `PitchData(Dataset)` and `PitchSalience` (Conv2d/BatchNorm2d layers), plus a terminal running the training script.

---

## Slide 17 — Tasks & Exemplar Models (overview)

- **Melody extraction**
  - F0 estimation in monophonic music
  - F0 estimation in polyphonic music
- **Leadsheet recognition**: quantized melody + chord
- **Multipitch estimation & multi-instrument transcription**

---

## Slide 18 — Melody Extraction from "Monophonic" Music: CREPE
https://github.com/marl/crepe
Kim et al., "CREPE: A convolutional representation for pitch estimation," ICASSP 2018

- Makes a prediction every 1,024 samples
- Frequency resolution: 20 cents (1/5 semitone)
- Outperforms DSP-based methods such as pYIN and SWIPE

**Diagram:** A 1D-CNN architecture: input is 1024 raw audio samples → six `conv1d` blocks (sizes/strides/filters: 512/stride4/1024 filters → 64/128 filters ×4 with maxpool → 64/512 filters) progressively reducing dimensionality (1024→128→64→32→16→8→4) while increasing channel depth, → reshape → fully connected (2048) → output layer of 360 units (pitch bins from C1 to B7) representing a probability distribution over pitch.

**Table** (evaluation on RWC-synth dataset, Raw Pitch Accuracy at different cent thresholds):

| Threshold | CREPE | pYIN | SWIPE |
|---|---|---|---|
| 50 cents | **0.999 ± 0.002** | 0.990 ± 0.006 | 0.963 ± 0.023 |
| 25 cents | **0.999 ± 0.003** | 0.972 ± 0.012 | 0.949 ± 0.026 |
| 10 cents | **0.995 ± 0.004** | 0.908 ± 0.032 | 0.833 ± 0.055 |

---

## Slide 19 — Vocal Melody Extraction from "Polyphonic" Music: RMVPE
https://github.com/Dream-High/RMVPE
Wei et al., "RMVPE: A robust model for vocal pitch estimation in polyphonic music," INTERSPEECH 2023

- Uses a deep **U-Net** and **GRU** to directly extract vocal F0 from polyphonic music
- Robust to different types of noise

**Diagram:** U-Net-style architecture. Input: Wav → Log Mel-Spectrogram → BN → 5 encoder layers (REB blocks, channels 1→16→32→64→128→256) → 4 intermediate layers (ICB blocks, channels up to 512) → 5 decoder layers (RDB blocks, mirroring the encoder back down to 16) with skip connections ("Skip Hidden Feature Filters," RCB blocks) linking corresponding encoder/decoder levels → Conv2D → BiGRU → FC+Sigmoid → Pitch Prediction output.

---

## Slide 20 — Melody Extraction and Multi-F0 Estimation: DeepSalience
https://github.com/rabitt/ismir2017-deepsalience
Kim et al., "Deep salience representations for F0 estimation in polyphonic music," ISMIR 2017

- Input: **harmonic constant-Q transform (HCQT)** with 6 channels
- Model: 5-layer CNN (**no strides, no pooling → no shift-invariance**)
  - (5×5) kernel: covers 1 semitone in frequency and 50 ms in time
  - (70×3) kernel: covers 14 semitones in frequency, to capture relationships between frequency content within an octave (i.e., harmonic relationships)

**Diagram:** Left: a stack of HCQT feature maps (6 channels, frequency × time) collapse into a single salience map (frequency × time). Right: the CNN architecture — 5 layers of decreasing spatial size but shown as blocks with kernel sizes 5×5, 5×5, 3×3, 3×3, then a wide 70×3 kernel, and a final 1×1 output layer producing a 360×50 salience map from 6→128→64→64→64→8→1 channels.

---

## Slide 21 — Tasks & Exemplar Models
- Melody extraction *(covered)*
- **Leadsheet recognition**: "quantized" melody + chord *(next)*
- Multipitch estimation & multi-instrument transcription

---

## Slide 22 — Lead Sheet Transcription: Sheet Sage
https://github.com/chrisdonahue/sheetsage
Donahue et al., "Melody transcription via generative pre-training," ISMIR 2022

- Predicts both melody and chord
  - Uses a large pretrained model called **Jukebox** as the backbone, then does transfer learning
  - Uses a **Transformer** to learn the language model (LM) for melody and chords
  - Computationally heavy but pretty accurate
  - Lighter alternative: **BTC** (https://github.com/jayg996/BTC-ISMIR19)

**Diagram:** Raw waveform → **Jukebox** backbone (outputs uniformly spaced in time) → combined with positional/time embeddings → **Transformer** decoder (one input/output per sixteenth note) → autoregressively predicts a token sequence (e.g., `E4, F4, G4, ∅`) which renders to both **MIDI** and conventional **Score** notation.

---

## Slide 23 — Lead Sheet Transcription: Sheet Sage-Pro
https://github.com/pingw220/sheetsage-pro
Wang et al., "SheetSage-Pro: Bilingual lyric-aligned lead-sheet transcription from music audio," ISMIR-LBD 2026

- Component-wise upgrade over Sheet Sage:
  - "Mel-Band RoFormer" for vocal separation
  - "Beat This!" for beat/downbeat tracking
  - "GAME" for melody extraction
  - Jiang's model for chord recognition
  - "All-in-One" for structure/meter analysis
- Output: **melody + chord + lyrics**
  - Lyrics with timestamps
  - Via lyrics transcription or audio-lyrics alignment

**Table** (comparison of Sheet Sage vs. Sheet Sage-Pro):

| Task | Metric | Sheet Sage | SheetSage-Pro |
|---|---|---|---|
| Melody, ZH | Note F1@50 ↑ | 0.049 | **0.250** |
| Melody, EN | Note F1@50 ↑ | 0.112 | 0.340 |
| ST500, ZH | Note F1@50 ↑ | 0.400 | 0.578 |
| ST500, ZH | Onset F1@50 ↑ | 0.514 | 0.645 |
| Chords | MajMin WCSR ↑ | 0.621 | **0.904** |
| Chord bounds | Boundary F1 ↑ | 0.559 | 0.799 |
| Lyrics, ZH | Syll. onset ↓ | — | 24 ms |
| Lyrics, EN | Word onset ↓ | — | 33 ms |

---

## Slide 24 — Lead Sheet Transcription: Sheet Sage2
https://huggingface.co/m-a-p/SheetSage2

**Diagram:** An encoder-decoder architecture:
- **MERT2-FS encoder**: audio waveform → frozen Conformer backbone (24×, with trainable LoRA adapters) → event embeddings
- **SheetSage2 decoder**: a stack of 6× Transformer decoder blocks, each with masked self-attention → add & norm → cross-attention (to encoder outputs, with "layer mix") → add & norm → feed-forward → add & norm, followed by linear + softmax to predict the next event, autoregressively generating a unified **event stream** with fields: time, beat, section, key, chord, melody
- Output tokens (`time`, `beat`, `section`, `key`, `chord`, `melody`, `duration`, `shift`) render into: (a) a conventional two-staff score (vocal + instrumental) with chord symbols (e.g., F♯m, D) and (b) an **ABC notation** string (e.g., `M:4/4 L:1/32 Q:1/4=128 K:F#m V:Vocal "F#m"a4a4c'8a8c'8|"D"a8...`)
- Example event-stream table showing: time (43.13–45.00s), beat (1/4–4/4), section (chorus), key (F♯ minor), chord (F♯m, A5, C♯6, D), melody notes (with "vocal" tag), duration, and shift values — illustrating how the model represents music as a column-wise event stream where a "shift" token advances the time cursor.

---

## Slide 25 — Tasks & Exemplar Models
- Melody extraction *(covered)*
- Leadsheet recognition *(covered)*
- **Multipitch estimation & multi-instrument transcription** *(next)*
  - Multipitch estimation in a single instrument (piano; output is a single-track pianoroll)
  - Multipitch estimation in general music (output is a single-track pianoroll)
  - Multi-instrument transcription (output is a multi-track pianoroll)

---

## Slide 26 — Piano Transcription: Onset-and-Frames
https://magenta.tensorflow.org/onsets-frames
Hawthorne et al., "Onsets and Frames: Dual-objective piano transcription," ISMIR 2018

**Diagram (architecture):** Log Mel-Spectrogram input feeds two parallel stacks:
- **Onset stack**: Conv Stack → FC Sigmoid → BiLSTM → Onset Predictions → (Onset Loss)
- **Frame stack**: Conv Stack → FC Sigmoid, concatenated with the onset stack's BiLSTM output → BiLSTM → FC Sigmoid → Frame Predictions → (Frame Loss)

The onset predictions are used to gate/restrict the frame predictions (dual-objective training).

**Diagram (piano-roll results):** Two pitch-vs-time plots (pitch 20–60). Top: raw **Frame and Onset Predictions** (blue = frame activity, magenta = onset markers). Bottom: **Frame Predictions Restricted by Onset Predictions**, recolored to show correctness — yellow = true positive, red = false positive/error, green = other — demonstrating how gating frame predictions by onsets cleans up the transcription.

---

## Slide 27 — Pitch, Onset, Offset, Velocity
https://magenta.tensorflow.org/datasets/maestro
Hawthorne et al., "Enabling factorized piano music modeling and generation with the MAESTRO dataset," ICLR 2019

> "We partnered with organizers of the International Piano-e-Competition for the raw data used in this dataset. During each installment of the competition virtuoso pianists perform on Yamaha Disklaviers which, in addition to being concert-quality acoustic grand pianos, utilize an integrated high-precision MIDI capture and playback system."

**Diagram 1:** A single piano note's audio waveform envelope over time (0–4s), labeled with its **Onset** (sharp amplitude rise, ~time 1) and **Offset** (decay to near-zero, ~time 3.5).

**Diagram 2 (table, "Dynamic's note velocity"):** Maps musical dynamic markings to MIDI velocity values and descriptive "voice" loudness:

| Dynamic | Velocity | Voice |
|---|---|---|
| ppp | 16 | Whispering |
| pp | 33 | Almost at a whisper |
| p | 49 | Softer than speaking voice |
| mp | 64 | Speaking voice |
| mf | 80 | Speaking voice |
| f | 96 | Louder than speaking |
| ff | 112 | Speaking loud |
| fff | 126 | Yelling |

Also shows symbols for decrescendo (diminuendo), crescendo, and accent. (Source: freeonlinesheetmusic.wordpress.com)

---

## Slide 28 — Piano Transcription with GAN
Kim & Bello, "Adversarial learning for improved onsets and frames music transcription," ISMIR 2019

**Diagram:** Extends the Onsets-and-Frames architecture (Mel spectrogram → 4 parallel CNN/BiLSTM branches producing onsets, offsets, velocity, and — via concatenation of onset+offset branches — frames) with an adversarial training setup: the Transcription Model's prediction `G(X) = Ŷ` and the ground truth `Y` are both fed to a **Discriminator** that predicts real/fake, producing an adversarial loss `L_cGAN` in addition to the standard task loss `L_task`.

---

## Slide 29 — High-resolution Piano Transcription
https://github.com/bytedance/piano_transcription
Kong et al., "High-resolution piano transcription with pedals by regressing onsets and offsets times," TASLP 2021

**Diagram:** Log mel spectrogram feeds four parallel Conv+GRU branches predicting **Velocity regression**, **Onset regression**, **Frame classification**, and **Offset regression** (onset/offset/frame branches interact via shared GRU layers). Right panel details the shared Conv+GRU block: alternating Conv layers (channels 48→64→96→128, each ×2) with frequency-pooling (stride 2) between them, then Flatten → FC (768) → biGRU (256) ×2 → FC (88, one per piano key).

---

## Slide 30 — Seq2Seq Transformers for Piano Transcription
Hawthorne et al., "Sequence-to-sequence piano transcription with Transformers," ISMIR 2021

- Jointly models audio features and language-like output dependencies
- Possible to pre-train the decoder

**Diagram:** A segmented waveform window → spectrogram → Transformer **Encoder** → autoregressive **Decoder** (sampling) → **MIDI-like output tokens**, e.g. `<time 73.1> <vel 61> <pitch 60> <time 73.3> <vel 82> <pitch 64> <time 73.5> <vel 0> <pitch 60>`, which render as a piano-roll with note bars (e.g., note C4 from 73.1–73.5s, note E4 starting at 73.3s).

**Table** (MAESTRO v1.0.0 results, F1 scores):

| Model | Onset, Offset & Velocity F1 | Onset & Offset F1 | Onset F1 |
|---|---|---|---|
| Transformer (ours) | **82.18** | **83.46** | 95.95 |
| Kong et al. 2020 | 80.92 | 82.47 | **96.72** |
| Kwon et al. 2020 | – | 79.36 | 94.67 |
| Kim & Bello 2019 | 80.20 | 81.30 | 95.60 |
| Hawthorne et al. 2019 | 77.54 | 80.50 | 95.32 |

---

## Slide 31 — Note Transcription from Polyphonic Music: Basic Pitch
https://github.com/spotify/basic-pitch
Bittner et al., "A lightweight instrument-agnostic model for polyphonic note transcription and multipitch estimation," ICASSP 2022

- For **instrument-agnostic** note transcription and multi-pitch estimation
- Pure CNN-based, for being light-weight

**Diagram (NMP architecture, "Fig. 1"):** Audio → CQT → Harmonic Stacking → Conv2D (32, 5×5, stride 1×3) → BatchNorm → ReLU → three parallel small conv branches producing posteriorgram outputs:
- **Y_o** (onset): via extra Conv2D layers and a final 1×3 conv + sigmoid, concatenated with contour info
- **Y_p** (pitch/contour): via 8-channel and 1-channel conv + sigmoid
- **Y_n** (note/multipitch): via 32-channel 7×7 conv (stride 1×3) and 1-channel 7×3 conv + sigmoid

Also shown: a promotional screenshot describing Basic Pitch as "a free audio-to-MIDI converter with pitch bend detection, built by Spotify" with 3 usage steps: (1) record or drop a recording of any single instrument, (2) get a MIDI version back, (3) download the MIDI to fine-tune in a DAW.

---

## Slide 32 — Note Transcription from Polyphonic Music: Harmonica
https://www.oulongshen.xyz/amt
Ou et al., "Harmonica: Accurate and lightweight instrument-agnostic music transcription," arXiv 2026

**Diagram 1 (architecture):** Audio → CQT Spectrogram → Frontend Block (2× ResNet Block, 5×5 conv) → Trunk Block (×l: Harmonic Conv with 14 offsets, then ResNet Block 3×3 conv, residual loop) → three output heads: **Onset Head**, **Pitch Head**, **Offset Head** (auxiliary supervision) → Onset Prediction + Sustain Prediction → Decode → Note Sequence. A side panel illustrates the "Harmonic Conv" mechanism: input feature maps at harmonically-related frequencies (5f, 4f, 3f, 2f, f, f/2, f/3, f/4, f/5) are combined via learned offsets into a single output feature map at frequency f.

**Diagram 2 (scatter plot):** Frame F1 score vs. number of parameters (log scale, 30K–100M) comparing Harmonica model sizes (nano 26.3K, small 137K, medium 679K, large 3.2M, x-large 15.1M — plotted as a rising blue curve reaching ~0.90 F1) against other baselines (Basic Pitch, TriAD, HPPNet-sp, PerceiverTF, HFSFormer, SFT-CRNN, Transkun, hFT-Transformer off-scale at .283) and seq2seq baselines (MT3, YourMT3+ original/retrained, MuScriptor) — Harmonica achieves higher F1 at far fewer parameters than most baselines.

---

## Slide 33 — Multi-Instrument Transcription: Omnizart
https://github.com/Music-and-Culture-Technology-Lab/omnizart
Wu et al., "Omnizart: A general toolbox for automatic music transcription," JOOS 2021

**Diagram:** Training data pipeline: music signal → manual annotation / auto-alignment → music annotation (instrument class, pitch class, onset time, offset time). Model pipeline: music signal → Fourier transform/filterbanks (signal processing, extracting attack, harmonics nf₀, fundamentals f₀) → Encoder (stack of Convolution, Pooling, Residual, Self-attention layers) → Decoder (mirrored stack) → output representation, predicting per-instrument (Instrument 1, Instrument 2, ... Instrument N) onset/offset piano-rolls (frequency axis × time axis).

---

## Slide 34 — Multi-Instrument Transcription: MT3
Gardner et al., "MT3: Multi-task multitrack music transcription," ICLR 2022

**Diagram:** A single unified **MT3** model is trained/evaluated across six datasets, each with different instrumentation, and outputs a separate pianoroll track per instrument for each:
- **MAESTRO** (piano only) → 1 piano track
- **Cerberus4** → piano, guitar (×2), drums tracks
- **GuitarSet** → 1 guitar track
- **MusicNet** → multiple orchestral instrument tracks (e.g., violin, horn)
- **Slakh2100** → piano, guitar, bass, drums, synth/other tracks
- **URMP** → multiple chamber-ensemble instrument tracks (violin, guitar, etc.)

---

## Slide 35 — Multi-Instrument Transcription: MuseScriptor
https://github.com/muscriptor/muscriptor
Rouard et al., "MuScriptor: An Open model for multi-instrument music transcription," ISMIR 2026

- SOTA model trained with 170k recordings (11k hours) with aligned note annotations, alongside a synthetic dataset of 1.45M MIDIs
- Excellent results when instrument tags are given

**Diagram:** Two piano-roll comparison plots (pitch C2–C5 vs. time 5–30s) showing transcription accuracy color-coded as TP (true positive, blue), FN (false negative / ground-truth only, green), and FP (false positive / model only, orange hatched). Top: **YourMT3+** baseline shows more green/orange (errors). Bottom: **MuScriptor** (trained on synthetic + real + RL data) shows substantially more blue (correct predictions) and fewer errors, illustrating its superior accuracy.
