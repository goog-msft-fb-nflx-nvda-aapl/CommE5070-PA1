---
## Page 1

Yi-Hsuan Yang Ph.D. 
yhyangtw@ntu.edu.tw
Music Classification
(audio → labels)
Deep Learning for Music Analysis and Generation
2026 edition

![Diagram/Image 1 on Page 1](images/page1_img1_Im1.jpg)
![Diagram/Image 2 on Page 1](images/page1_img2_Im2.jpg)

---
## Page 2

Music AI; or Music Information Research (MIR)
• Music analysis
– music understanding
– music search
– music recommendation
• Music generation
– MIDI generation
– audio generation
– MIDI-to-audio generation

![Diagram/Image 3 on Page 2](images/page2_img1_Im3.jpg)
![Diagram/Image 4 on Page 2](images/page2_img2_Im4.png)
![Diagram/Image 5 on Page 2](images/page2_img3_Im5.png)

---
## Page 3

Reference 1: KAIST Course & ISMIR 2021 Tutorial
For fundamentals of deep learning and music classification
• https://mac.kaist.ac.kr/~juhan/gct634/Slides/05.%20music%20classification%20-%20deep%20learning.pdf
• https://music-classification.github.io/tutorial/landing-page.html

![Diagram/Image 6 on Page 3](images/page3_img1_Im6.png)

---
## Page 4

Reference 2
• Deep Learning An MIT Press book (2016) by Ian Goodfellow and Yoshua Bengio and Aaron Courville
https://www.deeplearningbook.org/
• Deep Learning - Foundations and Concepts (2024) by Christopher M. Bishop & Hugh Bishop
https://link.springer.com/book/10.1007/978-3-031-45468-4
• Deep Learning 101 for Audio-based MIR (2024) by Geoffroy Peeters, Gabriel Meseguer-Brocal, Alain Riou & Stefan Lattner
https://geoffroypeeters.github.io/deeplearning-101-audiomir_book/task_musicprocessing.html

![Diagram/Image 7 on Page 4](images/page4_img1_Im7.png)

---
## Page 5

Outline
• Music classification: Basics
• ML-based music classification (and hand-crafted audio features)
• DL-based music classification

![Diagram/Image 8 on Page 5](images/page5_img1_Im8.png)

---
## Page 6

Different Classification Tasks
• Single-label vs multi-label
• Song-level vs instance-level
– instance/chunk/clip/segment (various names)

![Diagram/Image 9 on Page 6](images/page6_img1_Im9.png)
https://music-classification.github.io/tutorial/part1_intro/what-is-music-classification.html

---
## Page 7

Single-label vs Multi-label Classification
• Single-label classification
– One-hot; one out of many (mutually exclusive classes)
– Can be binary or multi-class classification
– Activation function (in DL): softmax (sum-to-one)
– Output interpretation: argmax
– Loss function: categorical cross entropy (CE)
• Multi-label classification
– Multi-hot; some out of many (may associate with multiple classes simultaneously)
– Activation function (in DL): sigmoid (each in [0,1], not sum-to-one)
– Output interpretation: >= 0.5 (or other thresholds)
– Loss function: binary cross entropy (BCE)

![Diagram/Image 10 on Page 7](images/page7_img1_Im10.png)
https://www.singlestore.com/blog/a-guide-to-softmax-activation-function/

---
## Page 8

Song-level vs Instance-level Classification
• Song-level: make a prediction for the entire song (or a long audio clips), without specifying the temporal location/activation of each class
• Instance-level: need to mark the temporal location/activation of each class
• It’s related to the length of the model input
– Make a prediction per STFT frame
– Make a prediction every second
– Make a prediction for a 30-second spectrogram
– Make a prediction per musical note (variable-length)
– Make a prediction per musical section (verse, chorus, etc) (variable-length)

![Diagram/Image 11 on Page 8](images/page8_img1_Im11.png)

---
## Page 9

Different Classification Tasks
• The most explored music classification tasks in MIR
• Many others
– singer/composer classification
– technique classification
– audio event detection

![Diagram/Image 12 on Page 9](images/page9_img1_Im12.png)
https://music-classification.github.io/tutorial/part1_intro/what-is-music-classification.html

---
## Page 10

Genre/Style Classification
• A conventional category that identifies some pieces of music as belonging to a shared tradition or set of conventions
• Good accessible entry point
• Key challenges
– Fuzzy boundaries
– Overlaps & ambiguities
– Evolving definitions (e.g., Pop)
  ◦ Acoustic signatures and genre definitions shift dynamically across decades
– Hierarchical structure
  ◦ Subgenres

![Diagram/Image 13 on Page 10](images/page10_img1_Im13.png)

---
## Page 11

Emotion/Mood Classification
• A primary reason people listen to music
• Key challenges
– Semantic ambiguity: emotion descriptors (e.g., “bittersweet”, “melancholy”) are subtle, overlapping, and hard to standardize
  ◦ Example taxonomy: https://www.allmusic.com/moods
– Annotation noise: highly subjective
– Time-varying dynamics: music exhibits dynamic emotional trajectories

![Diagram/Image 14 on Page 11](images/page11_img1_Im14.png)

---
## Page 12

Emotion/Mood Classification/"Regression"
• Song-level or instance-level
– music emotion variation detection
• Classification vs. regression
– arousal: energy or neuro-physiological stimulation level
– valence: pleasantness or positive/negative affective states
– popular taxonomy: 4Qs of the valence/arousal plane
• Perceived vs. felt emotion

![Diagram/Image 15 on Page 12](images/page12_img1_Im15.png)
https://musemap.org/resources/gems
Geneva Emotional Music Scale (GEMS)
https://github.com/juansgomez87/datasets_emotion

---
## Page 13

Instrument Classification/Detection
• Song-level or instance-level
– singing activity detection
– instrument activity detection
• Hierarchical taxonomy

Ref1: Krause et al., “Hierarchical classification of singing activity, gender, and type in complex music recordings,” ICASSP 2022
Ref2: Krause et al., “Hierarchical classification for instrument activity detection in orchestral music recordings,” TASLP 2023

![Diagram/Image 16 on Page 13](images/page13_img1_Im16.png)

---
## Page 14

Music Tagging
MagnaTagATune (https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset)
Ref: Law et al., “Evaluation of algorithms using games: the case of music annotation,” ISMIR 2009
• Top 50 by categories (source)
– genre: classical, techno, electronic, rock, indian, opera, pop, classic, new age, dance, country, metal
– instrument: guitar, strings, drums, piano, violin, vocal, synth, female, male, singing, vocals, no vocals, harpsichord, flute, no vocal, sitar, man, choir, voice, male voice, female vocal, harp, cello, female voice, choral
– mood: slow, fast, ambient, loud, quiet, soft, weird
– etc: beat, solo, beats

![Diagram/Image 17 on Page 14](images/page14_img1_Im17.png)

---
## Page 15

Technique Classification
• Electric guitar
– bend, vibrato, hammer-on, pull-off, slide
– https://zenodo.org/record/1414806
• Singing voice
– breathy, vibrato, vocal fry, etc
– https://zenodo.org/record/1193957

![Diagram/Image 18 on Page 15](images/page15_img1_Im18.png)

---
## Page 16

Technique Classification: Vibrato and Tremolo
• Tremolo: periodic variations in amplitude (amplitude modulations)
• Vibrato: periodic variations in frequency (frequency modulations)
– Wind and bowed instruments generally use vibratos with an extent of less than half a semitone either side
– Tremolo and vibrato do not necessarily evoke a perceived change in loudness or pitch of the tone

https://en.wikipedia.org/wiki/Vibrato
Ref: Sundberg, “Acoustic and psychoacoustic aspects of vocal vibrato,” 1994

![Diagram/Image 19 on Page 16](images/page16_img1_Im19.png)

---
## Page 17

Audio Event Detection
Audio Set (http://research.google.com/audioset/)
Ref: Gemmeke et al., “Audio Set: An ontology and human-labeled dataset for audio events,” ICASSP 2017
• 527 audio classes
– Over 2M audio clips from YouTube
– Each “10 second”
• Widely used benchmark for audio classification and audio captioning
• Can be useful for sound design

![Diagram/Image 20 on Page 17](images/page17_img1_Im20.png)

---
## Page 18

Audio Event Detection
https://github.com/qiuqiangkong/audioset_tagging_cnn
• Useful for
– Music/singing/speech detection; instrument activity detection

![Diagram/Image 21 on Page 18](images/page18_img1_Im21.png)

---
## Page 19

Different Classification Tasks

| Task | Single-label? | Multi-label? | Song-level? | Instance-level? |
| --- | --- | --- | --- | --- |
| Genre analysis | | | | |
| Emotion analysis | | | | |
| Instrument analysis | | | | |
| Technical analysis | | | | |
| Audio event detection | | | | |

• It depends on your dataset and your problem formulation
• Single-label, song-level classification is the simplest setting (good for beginners)
• Instance-level annotations are harder to collect (so less research)
• We can do instance level first, then aggregate the result into the song level

![Diagram/Image 22 on Page 19](images/page19_img1_Im22.png)

---
## Page 20

Outline
• Music classification: Basics
• ML-based music classification (and hand-crafted audio features)
• DL-based music classification

![Diagram/Image 23 on Page 20](images/page20_img1_Im23.png)

---
## Page 21

ISMIR 2024 Test-of-Time (ToT) Awardee

Ref: Tzanetakis et al, “Automatic musical genre classification of audio signals,” ISMIR 2001

![Diagram/Image 24 on Page 21](images/page21_img1_Im24.png)

---
## Page 22

Musical Genre Classification of Audio Signals
• “A musical genre is characterized by the common characteristics shared by its members. These characteristics typically are related to the instrumentation, rhythmic structure, and harmonic content of the music.”
• “In this paper, [...] three feature sets for representing timbral texture, rhythmic content and pitch content are proposed.”
• “The performance and relative importance of the proposed features is investigated by training statistical pattern recognition classifiers using real-world audio collections. [...] Using the proposed feature sets, classification of 61% for ten musical genres is achieved.”

https://www.cs.cmu.edu/~gtzan/work/pubs/tsap02gtzan.pdf

![Diagram/Image 25 on Page 22](images/page22_img1_Im25.png)

---
## Page 23

The GTZAN Dataset
• “The dataset consists of 1000 audio tracks each 30 seconds long. It contains 10 genres, each represented by 100 tracks. The tracks are all 22,050Hz Mono 16-bit audio files in WAV format”
https://www.tensorflow.org/datasets/catalog/gtzan
• The genres are:
– blues, classical, country, disco, hiphop
– jazz, metal, pop, reggae, rock

![Diagram/Image 26 on Page 23](images/page23_img1_Im26.png)

---
## Page 24

Representing the Audio as Features is Needed in ML/DL
• Given: {x1, y1, ..., xN, yN}
– xi ∈ ℝ^M: feature representation; a “vector”
– yi ∈ {-1, +1}: class label

![Diagram/Image 27 on Page 24](images/page24_img1_Im27.png)

---
## Page 25

The Three Feature Sets Used By GTZAN
• They are basically statistics
• Timbral texture features: Statistics from the waveform and magnitude spectrogram
– Spectral centroid
– Spectral rolloff
– Spectral flux
– Time domain zero crossings
– MFCCs
– Low-energy feature
• Rhythmic content features
– Statistics from the result of a simple DSP-based “beat estimator” (based on sub-band auto-correlation)
• Pitch content features
– Statistics from pitch histogram analysis (also based on auto-correlation, yet operating on much shorter time frames than beat detection)

![Diagram/Image 28 on Page 25](images/page25_img1_Im28.png)

---
## Page 26

The Rhythmic Content Features Used By GTZAN
Fig. 2 shows a beat histogram for a 30-s excerpt of the song “Come Together” by the Beatles. The two main peaks of the BH correspond to the main beat at approximately 80 bpm and its first harmonic (twice the speed) at 160 bpm. Fig. 3 shows four beat histograms of pieces from different musical genres. The upper left corner, labeled classical, is the BH of an excerpt from “La Mer” by Claude Debussy. Because of the complexity of the multiple instruments of the orchestra there is no strong self-similarity and there is no clear dominant peak in the histogram. More strong peaks can be seen at the lower left corner, labeled jazz, which is an excerpt from a live performance by Dee Dee Bridgewater. The two peaks correspond to the beat of the song (70 and 140 bpm). The BH of Fig. 2 is shown on the upper right corner where the peaks are more pronounced because of the stronger beat of rock music.

![Diagram/Image 29 on Page 26](images/page26_img1_Im29.png)

---
## Page 27

The Rhythmic Content Features Used By GTZAN
• Rhythmic content features: Statistics from a “beat histogram”
– A0, A1 (tempo clarity): relative amplitude (divided by the sum of amplitudes) of the first, and second histogram peak
– RA = A1/A0 (ratio of the two peaks)
– P1, P2: period of the two peaks (BPM)
– SUM (beat strength): overall sum of the histogram

![Diagram/Image 30 on Page 27](images/page27_img1_Im30.png)

---
## Page 28

The Timbral Texture Features Used By GTZAN
• Question: how to compute features/statistics from the spectrogram (or waveform)?
– The number of features cannot be too large
– The features have to be somehow “meaningful”

![Diagram/Image 31 on Page 28](images/page28_img1_Im31.png)

---
## Page 29

Spectral Centroid
• Each frame of a magnitude spectrogram is normalized and treated as a distribution over frequency bins, from which the mean (centroid) is extracted per frame
• A measure of spectral shape
• Higher centroid values imply “brighter” textures with more high frequencies
• Other statistics can also be used
– bandwidth, skewness, kurtosis

Ref: Tzanetakis and Cook, “Musical genre classification of audio signals,” TASLP 2002

![Diagram/Image 32 on Page 29](images/page29_img1_Im32.png)

---
## Page 30

Spectral Rolloff
• The frequency for a spectrogram bin such that at least roll_percent (0.85 by default) of the energy of the spectrum in a frame is contained in this bin and the bins below
• Can be used to approximate the maximum (or minimum) frequency by setting roll_percent to a value close to 1 (or 0)
• Another measure of spectral shape

Ref: McFee et al., “librosa: Audio and music signal analysis in python,” 2015

![Diagram/Image 33 on Page 30](images/page30_img1_Im33.png)

---
## Page 31

Spectral Contrast
• Each frame of a spectrogram is divided into multiple sub-bands
• Compute the mean energy for each sub-band
• Compare the mean energy in the top quantile (peak energy) to that of the bottom quantile (valley energy)
– High contrast values generally correspond to clear, narrow-band signals, while low contrast values correspond to broad-band noise
• Alternatively: entropy of the sub-band mean energy

Ref: McFee et al., “librosa: Audio and music signal analysis in python,” 2015

![Diagram/Image 34 on Page 31](images/page31_img1_Im34.png)

---
## Page 32

Spectral Flux
• How quickly the power spectrum of a signal changes over time
• Usually calculated as the L2-difference between two adjacent normalized spectra
• This feature is also often used for musical onset detection (more related to rhythm)

Ref1: https://en.wikipedia.org/wiki/Spectral_flux
Ref2: https://librosa.org/librosa_gallery/auto_examples/plot_superflux.html

![Diagram/Image 35 on Page 32](images/page32_img1_Im35.png)

---
## Page 33

Log Mel-Spectrogram
• The Mel scale is a perceptual scale of pitches judged by listeners to be equal in distance from one another
• Finer resolution in the low-frequency range (NOT exactly logarithmic scale)
• Dimension reduction
linear scale: hundreds of frequency bins
mel scale: tens of frequency bands

![Diagram/Image 36 on Page 33](images/page33_img1_Im36.png)

---
## Page 34

Log Mel-Spectrogram
https://music-classification.github.io/tutorial/part2_basics/input-representations.html
Figure from librosa

![Diagram/Image 37 on Page 34](images/page34_img1_Im37.png)

---
## Page 35

Mel-frequency cepstral coefficients (MFCC)
• Procedure
1. Compute the spectrogram
2. Grouping the FFT bins according to the perceptually motivated Mel-filter bank
3. Taking logs and DCT for uncorrelating the resulting features
• “Frequency analysis” on the log mel spectrum (frequency → “quefrency”)
• Low-order coefficients: slow variations across frequency (vocal tract filter or spectral envelope)
• High-order coefficients: rapid oscillations across adjacent mel bands (excitation source; pitch)
• Compact representation of the spectrum (1,024-dim → 128 → 13 coefficients)
– Somehow capture the energy distribution in the spectrum
– Less interpretable

Ref: McFee et al., “librosa: Audio and music signal analysis in python,” 2015

![Diagram/Image 38 on Page 35](images/page35_img1_Im38.png)

---
## Page 36

Spectral Features Can be Used to Build a Classifier in ML
• Procedure
1. Compute the features per STFT frame (e.g., 13-D frame-level features)
2. Temporal pooling over time for each audio clip (e.g., by taking the mean and variance; leading to 26-D clip-level features)
3. Use that as input to a classifier (e.g., random forest, or support vector machine)
• Limits
– Clear physical meaning but not sophisticated enough and limited semantic meaning
– Only the classifier is trainable; the features are hand-crafted

![Diagram/Image 39 on Page 36](images/page36_img1_Im39.png)

---
## Page 37

The GTZAN Classifier
• Each 30-sec audio is represented as a “single” vector (that is composed of the timbre, rhythmic and pitch features)
• Then train a classifier using Gaussian mixture model (GMM) or K-nearest neighbor (K-NN)
• Confusion matrix
– Columns: GT / rows: predictions
– “Classical music is misclassified as jazz music for pieces with strong rhythm from composers like Leonard Bernstein and George Gershwin.”
– “Rock music has the worst classification accuracy and is easily confused with other genres which is expected because of its broad nature.”

![Diagram/Image 40 on Page 37](images/page37_img1_Im40.png)

---
## Page 38

More on the Mel-Spectrogram
• Widely-used feature representation for musical audio
– Easy to compute and understand
– Reasonably rich information
– Reasonable size
– Can be used as input to computer vision (CV) models
– Possible to go back from mel-spectrograms to waveforms via a “vocoder”
• Used in all types of tasks, for both music analysis and generation
• MFCC is not preferred due to information loss

![Diagram/Image 41 on Page 38](images/page38_img1_Im41.png)

---
## Page 39

Library: Torchaudio
https://pytorch.org/audio/0.11.0/tutorials/audio_feature_extractions_tutorial.html

![Diagram/Image 42 on Page 39](images/page39_img1_Im42.png)

---
## Page 40

Library: LibROSA
https://librosa.org/doc/latest/index.html

![Diagram/Image 43 on Page 40](images/page40_img1_Im43.png)

---
## Page 41

Library: LibROSA
Ref: McFee et al., “librosa: Audio and music signal analysis in python,” 2015

![Diagram/Image 44 on Page 41](images/page41_img1_Im44.png)

---
## Page 42

Library: Audio Commons Audio Extractor
https://github.com/AudioCommons/ac-audio-extractor

![Diagram/Image 45 on Page 42](images/page42_img1_Im45.png)

---
## Page 43

More on LibROSA
https://librosa.org/doc/latest/index.html

![Diagram/Image 46 on Page 43](images/page43_img1_Im46.png)

---
## Page 44

More on LibROSA
https://librosa.org/doc/latest/index.html

![Diagram/Image 47 on Page 44](images/page44_img1_Im47.png)

---
## Page 45

More on LibROSA
https://librosa.org/doc/latest/index.html

![Diagram/Image 48 on Page 45](images/page45_img1_Im48.png)

---
## Page 46

“Feature Learning” in DL
(the blocks inside the black lines are learned)

Ref: Nam et al., “Deep learning for audio-based music classification and tagging,” IEEE Signal Processing Magazine, 2019
(a) Feature engineering
(b) Low-level feature learning
(c) Convolution neural networks
(d) End-to-end learning

![Diagram/Image 49 on Page 46](images/page46_img1_Im49.png)

---
## Page 47

The Use of Hand-crafted Features in DL
• Hand-crafted features
– Clear physical meaning; interpretable
• Used as input to music classifiers in the early days
• Can be used alongside learned features
– The “deep & wide” architecture

Ref: Cheng et al., “Wide & deep learning for recommender systems,” DLRS 2016

![Diagram/Image 50 on Page 47](images/page47_img1_Im50.png)

---
## Page 48

The Use of Hand-crafted Features in DL
• Can be used as objective metrics or loss functions for DL models

“Our analysis of the errors produced by source separators shows that waveform models [Demucs; middle subfigure] tend to introduce more high-frequency noise, while spectrogram models [Open-Unmix; right subfigure] tend to lose transients and high frequency content. We introduce objective measures [spectral rolloff] to quantify both kinds of errors”

Ref: Schaffer et al., “Music separation enhancement with generative modeling,” ISMIR 2022

![Diagram/Image 51 on Page 48](images/page48_img1_Im51.png)

---
## Page 49

Moving Beyond Hand-Crafted Features/Statistics
• Log magnitude spectrogram: log10 |X(t,f)| + ε
• Log Mel-spectrogram: log10 |M · X(t,f)| + ε
• Constant-Q Transform (CQT)
• Chromagram
• All four are time-frequency representations

![Diagram/Image 52 on Page 49](images/page49_img1_Im52.png)

---
## Page 50

Constant-Q Transform (CQT)
https://music-classification.github.io/tutorial/part2_basics/input-representations.html
• STFT (linearly-spaced frequencies): f_k = k · Δf
• CQT (logarithmically-spaced): f_k = f_0 · 2^(k/B) (B = 12 for semitones)
– Logarithmic frequency, logarithmic magnitude
– Constant “Q”-factor: using long windows for low frequencies, and short windows for high frequencies

![Diagram/Image 53 on Page 50](images/page50_img1_Im53.png)

---
## Page 51

Constant-Q Transform (CQT)
• Logarithmically spaced center frequencies: f_k = f_0 · 2^(k/B)
– CQT places its center frequencies on a geometric/logarithmic scale
– This aligns the analysis bins directly with musical pitch intervals (e.g., semitones) and human auditory perception
• Frequency-dependent bandpass filtering (constant-Q): Δf_k = f_k / Q
– Instead of using a single fixed window length across all frequencies, CQT operates as a bank of bandpass filters where the filter bandwidth scales proportionally with the center frequency
– long windows (narrow bandpass filters) at low frequencies and short windows (wide bandpass filters) at high frequencies
• Accordingly, constant musical resolution across all octaves

![Diagram/Image 54 on Page 51](images/page51_img1_Im54.png)

---
## Page 52

Constant-Q Transform (CQT)
• Good for pitch-related tasks (e.g., music transcription)
• Downsides
– Basis functions are non-orthogonal and overcomplete
  ◦ Slower to compute & invertibility issues
– Time-frequency trade-offs (e.g., long windows at low freq)
• Library
– https://github.com/archinetai/cqt-pytorch
– https://github.com/eloimoliner/CQT_pytorch

Ref: Hung et al, “Multitask learning for frame-level instrument recognition,” ICASSP 2019

![Diagram/Image 55 on Page 52](images/page52_img1_Im55.png)

---
## Page 53

Pitch Class Profile / Chromagram
https://musicinformationretrieval.com/chroma.html
• “A chroma vector (Wikipedia) is a typically a 12-element feature vector indicating how much energy of each pitch class, {C, C#, D, D#, E, ..., B}, is present in the signal”
– i.e., ignore octaves

![Diagram/Image 56 on Page 53](images/page53_img1_Im56.png)

---
## Page 54

Pitch Class Profile / Chromagram
• Good for tasks such as cover song identification or chord recognition

Ref1: Müller et al, “Audio matching via chroma-based statistical features,” ISMIR 2005
Ref2: Serra et al, “Chroma binary similarity and local alignment applied to cover song identification,” TASLP 2008
https://essentia.upf.edu/tutorial_similarity_cover.html

![Diagram/Image 57 on Page 54](images/page54_img1_Im57.png)

---
## Page 55

Pitch Class Profile / Chromagram
• Good for tasks such as cover song identification or chord recognition

https://www.audiolabs-erlangen.de/resources/MIR/FMP/C5/C5S2_ChordRec_Templates.html
Ref: Cho et al, “On the relative importance of individual components of chord recognition systems,” TASLP 2014

![Diagram/Image 58 on Page 55](images/page55_img1_Im58.png)

---
## Page 56

Pitch Class Profile / Chromagram

https://librosa.org/doc/latest/auto_tutorials/03-advanced/plot_chroma.html

![Diagram/Image 59 on Page 56](images/page56_img1_Im59.png)

---
## Page 57

ISMIR 2025 Test-of-Time (ToT) Awardee

Ref: Müller et al, “Audio matching via chroma-based statistical features,” ISMIR 2005

![Diagram/Image 60 on Page 57](images/page57_img1_Im60.png)

---
## Page 58

Different Features for Different Tasks
• Timbre representation: Spectrogram → mel-spectrogram → MFCC
• Harmonic representation: Spectrogram → CQT → chroma feature

Ref: Humphrey et al, “Feature learning and deep architectures: new directions for music informatics,” JIIS 2013

![Diagram/Image 61 on Page 58](images/page58_img1_Im61.png)

---
## Page 59

Different Features for Different Tasks
• Combine different features to train a classifier for tasks such as music genre classification or music emotion classification

https://maelfabien.github.io/machinelearning/Speech10/#spectrogram

![Diagram/Image 62 on Page 59](images/page59_img1_Im62.png)

---
## Page 60

Different Features are Needed for Different Tasks
• Timbre representation: Spectrogram → mel-spectrogram → MFCC
• Harmonic representation: Spectrogram → CQT → chroma feature
• Wait... why the window size is different?

Ref: Humphrey et al, “Feature learning and deep architectures: new directions for music informatics,” JIIS 2013

![Diagram/Image 63 on Page 60](images/page60_img1_Im63.png)

---
## Page 61

Recap: Math in STFT
• Frequency spacing: Fs / N
– Longer window size (N) → finer frequency resolution (but larger resulting STFT) → can localize events along the frequency axis
• Temporal spacing: Fs / H
– Smaller hop size (H) → finer temporal resolution (but larger resulting STFT) → can localize events along the time axis
• If H = N/R (e.g., R = 4)
– Temporal resolution: R / Fs (no good freq/time resolution at the same time)

![Diagram/Image 64 on Page 61](images/page61_img1_Im64.png)

---
## Page 62

Real Example 1: Piano Transcription
• Curtis Hawthorne et al., “Onsets and Frames: Dual-objective piano transcription,” ISMIR 2018
https://archives.ismir.net/ismir2018/paper/000019.pdf
Q: What is the frequency resolution? And the temporal resolution?

![Diagram/Image 65 on Page 62](images/page62_img1_Im65.png)

---
## Page 63

Real Example 2: Beat Tracking
• Sebastian Böck, Florian Krebs, and Gerhard Widmer, “Joint beat and downbeat tracking with recurrent neural networks,” ISMIR 2016
http://www.cp.jku.at/research/papers/Boeck_etal_ISMIR_2016.pdf
120 BPM = 120 beats per minute = 2 beats per second
(16th note in 4/4 meter: 125 ms)

![Diagram/Image 66 on Page 63](images/page63_img1_Im66.png)

---
## Page 64

Different STFT Parameters are Needed for Different Tasks
• Timbre/rhythm related tasks tend to use smaller STFT windows
• Pitch/harmony related tasks tend to use longer STFT windows
• Make sure you use the right Fs, window size and hop size!
– Especially when using models from open source projects
– STFTs of the same matrix size may have different physical meanings

![Diagram/Image 67 on Page 64](images/page64_img1_Im67.png)

---
## Page 65

Other Time-Frequency Representations
• STFT of multiple window sizes (often seen in DL papers)
• CQT
• Wavelets
• Scattering transform
• In DL, STFT is a common choice

![Diagram/Image 68 on Page 65](images/page65_img1_Im68.png)

---
## Page 66

Outline
• Music classification: Basics
• ML-based music classification (and hand-crafted audio features)
• DL-based music classification

![Diagram/Image 69 on Page 66](images/page66_img1_Im69.png)

---
## Page 67

Reference: KAIST Course & ISMIR 2021 Tutorial
For fundamentals of deep learning and music classification
• https://mac.kaist.ac.kr/~juhan/gct634/Slides/05.%20music%20classification%20-%20deep%20learning.pdf
• https://music-classification.github.io/tutorial/landing-page.html

![Diagram/Image 70 on Page 67](images/page67_img1_Im70.png)

---
## Page 68

Feature Learning
(the blocks inside the black lines are learned)

Ref: Nam et al., “Deep learning for audio-based music classification and tagging,” IEEE Signal Processing Magazine, 2019
(a) Feature engineering
(b) Low-level feature learning
(c) Convolution neural networks
(d) End-to-end learning

![Diagram/Image 71 on Page 68](images/page68_img1_Im71.png)

---
## Page 69

Feature Learning by Convolutional Layers

Ref: Kereliuk et al, “Deep learning and music adversaries,” IEEE Trans. Multimedia, 2015

![Diagram/Image 72 on Page 69](images/page69_img1_Im72.png)

---
## Page 70

Downsampling Layers: Convolution

https://github.com/vdumoulin/conv_arithmetic

![Diagram/Image 73 on Page 70](images/page70_img1_Im73.png)

---
## Page 71

Padding & Stride

From: https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html

![Diagram/Image 74 on Page 71](images/page71_img1_Im74.png)

---
## Page 72

Pooling
• Both strided convolutions and pooling reduce the spatial dimensions (time frames and frequency bins) of feature maps in a CNN
• Pooling introduces local translation invariance

From: https://d2l.ai/chapter_convolutional-neural-networks/pooling.html#pooling

![Diagram/Image 75 on Page 72](images/page72_img1_Im75.png)

---
## Page 73

Convolution: Locality and Translation Invariance
• “Convolution + pooling” may lead to translation invariance
• See Dr. Juhan Nam’s slides (GCT634)

![Diagram/Image 76 on Page 73](images/page73_img1_Im76.png)

---
## Page 74

Different Convolution Approaches
• 1D CNNs
• 2D CNNs
• Sample-level CNNs

![Diagram/Image 77 on Page 74](images/page74_img1_Im77.png)

---
## Page 75

1D CNNs vs. 2D CNNs
• 1D CNNs
– The filter size of the first conv layer covers the entire frequency range (can cover multiple frames)
– Fast to train
– Time-invariant but not pitch-invariant
• 2D CNNs
– Significantly increases the number of parameters and thus need more computational resources
– More flexible and powerful
– Might be pitch-invariant

![Diagram/Image 78 on Page 75](images/page75_img1_Im78.png)

---
## Page 76

1D CNNs vs. 2D CNNs
(figure from Dr. Juhan Nam’s slides (GCT634))

![Diagram/Image 79 on Page 76](images/page76_img1_Im79.png)

---
## Page 77

1D CNN
(From Prof. Nam' slides)

![Diagram/Image 80 on Page 77](images/page77_img1_Im80.png)

---
## Page 78

Input Audio Representation
https://music-classification.github.io/tutorial/part2_basics/input-representations.html
• Log magnitude spectrogram
– Be viewed as a raw audio representation
– Discard phase: the human auditory system is insensitive to phase information
– Log compression: human perception of loudness is closer to a logarithmic scale
• Log Mel-spectrogram
– Based on a Mel-scale, which is nonlinear and approximates human perception
– Reduces the number of frequency band greatly (1,024 → 128)

![Diagram/Image 81 on Page 78](images/page78_img1_Im81.png)

---
## Page 79

Sample-level CNN
• Work on audio samples (e.g., two or three samples) rather than a typical window size (e.g., 512 samples)
• Longer training time; need larger compute

Ref: Lee et al., “Sample-level deep convolutional neural networks for music auto-tagging using raw waveforms,” SMC 2017

![Diagram/Image 82 on Page 79](images/page79_img1_Im82.png)

---
## Page 80

Convolutional Recurrent Neural Networks
• See Dr. Juhan Nam’s slides (GCT634)
• Use RNN for temporal summary
• The CRNN model slightly outperforms the CNN models but it is slower

Ref: Choi et al., “Convolutional recurrent neural networks for music classification,” ICASSP 2017
https://music-classification.github.io/tutorial/part3_supervised/architectures.html

![Diagram/Image 83 on Page 80](images/page80_img1_Im83.png)

---
## Page 81

Short-Chunk CNN
https://github.com/minzwon/sota-music-tagging-models/blob/master/training/model.py
Ref: Won et al., “Evaluation of CNN-based automatic music tagging models,” SMC 2020
• Fixed short inputs: short audio chunks (3.69 seconds)
• Small 3x3 kernels (similar to VGG-net)
• Aggressive strided max-pooling: downsamples both time and frequency axes repeatedly across 6–7 convolutional blocks
• Global aggregation: aggregates predictions across short audio chunks of a long audio track via average pooling during inference
• Good data efficiency and generalization; lightweight; easy to implement

![Diagram/Image 84 on Page 81](images/page81_img1_Im84.png)

---
## Page 82

Short-Chunk CNN
https://github.com/minzwon/sota-music-tagging-models/blob/master/training/model.py
Ref: Won et al., “Evaluation of CNN-based automatic music tagging models,” SMC 2020
(batch_size, time_samples)
(batch_size, freq_bins, time_frames)
(batch_size, 1, freq_bins, time_frames)

![Diagram/Image 85 on Page 82](images/page82_img1_Im85.png)

---
## Page 83

Short-Chunk CNN
Ref: Won et al., “Evaluation of CNN-based automatic music tagging models,” SMC 2020
https://github.com/minzwon/sota-music-tagging-models/blob/master/training/modules.py

![Diagram/Image 86 on Page 83](images/page83_img1_Im86.png)

---
## Page 84

Short-Chunk CNN and Others
https://github.com/minzwon/sota-music-tagging-models

![Diagram/Image 87 on Page 84](images/page84_img1_Im87.png)

---
## Page 85

Short-Chunk CNN and Others
https://github.com/minzwon/sota-music-tagging-models

![Diagram/Image 88 on Page 85](images/page85_img1_Im88.png)

---
## Page 86

Exemplar Models
https://music-classification.github.io/tutorial/part3_supervised/architectures.html

![Diagram/Image 89 on Page 86](images/page86_img1_Im89.png)

---
## Page 87

Data Augmentation
• Methods that add modified copies to a dataset, from the existing data
– Create variations of natural data
– Can act as a regularizer to reduce the problem of overfitting
– Also help NN models become robust to complex variations of natural data, which improves their generalization performance

https://music-classification.github.io/tutorial/part3_supervised/data-augmentation.html
https://www.ibm.com/think/topics/data-augmentation

![Diagram/Image 90 on Page 87](images/page87_img1_Im90.png)

---
## Page 88

Audio Data Augmentation
https://music-classification.github.io/tutorial/part3_supervised/data-augmentation.html

![Diagram/Image 91 on Page 88](images/page88_img1_Im91.png)

---
## Page 89

torchaudio_augmentations
https://pytorch.org/audio/stable/tutorials/audio_data_augmentation_tutorial.html
https://music-classification.github.io/tutorial/part3_supervised/tutorial.html

![Diagram/Image 92 on Page 89](images/page89_img1_Im92.png)

---
## Page 90

Audio Degradation Toolbox
• Exemplar use case: simulate the case of smartphone recording

https://github.com/sevagh/audio-degradation-toolbox

![Diagram/Image 93 on Page 90](images/page90_img1_Im93.png)

---
## Page 91

pyrubberband
• Time stretch: make it faster/slower without changing the pitch
• Pitch shift
• (ps. There is a cool function called “timemap_stretch”; check it out yourself)

https://github.com/bmcfee/pyrubberband

![Diagram/Image 94 on Page 91](images/page91_img1_Im94.png)

---
## Page 92

Sample Code
https://music-classification.github.io/tutorial/part3_supervised/tutorial.html

![Diagram/Image 95 on Page 92](images/page92_img1_Im95.png)

---
## Page 93

Sample Code

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

https://music-classification.github.io/tutorial/part3_supervised/tutorial.html

![Diagram/Image 96 on Page 93](images/page93_img1_Im96.png)

---
## Page 94

Exemplar Model: PANNs
https://github.com/qiuqiangkong/audioset_tagging_cnn
Ref: Kong et al., “PANNs: Large-scale pretrained audio neural networks for audio pattern recognition,” TASLP 2020
• From sound event detection
• Work on 10-sec at once
• Can be used as a pre-trained model
– Produce audio embeddings that have been used by CLAP (https://github.com/LAION-AI/CLAP) in learning audio-text joint embedding space for text-to-audio generation

![Diagram/Image 97 on Page 94](images/page94_img1_Im97.png)

---
## Page 95

Exemplar Model: Audio Spectrogram Transformer
https://github.com/YuanGongND/ast
• From sound event detection
• Work on 10-second input
• Use Vision Transformer (ViT) based architecture
– The first convolution-free, purely attention-based model for audio classification
• May need larger amount of training data and compute

Ref: Gong et al., “AST: Audio Spectrogram Transformer,” INTERSPEECH 2021

![Diagram/Image 98 on Page 95](images/page95_img1_Im98.png)

---
## Page 96

Exemplar Model: HTS-AT
Ref: Chen et al., “HTS-AT: A hierarchical token-semantic audio transformer for sound classification and detection,” ICASSP 2022
Ref: Wu et al., “Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation,” arXiv 2022

![Diagram/Image 99 on Page 96](images/page96_img1_Im99.png)

---
## Page 97

Evaluation Metrics for Music Classification
• The output of DL-based classifiers are usually probabilities
– multi-class classification: softmax
– multi-label classification: sigmoid
• Probability vs decision
– outputting probabilities is fine at training time
– but, at inference time, need to “make decisions”
– usually by thresholding (e.g., at 0.5)

![Diagram/Image 100 on Page 97](images/page97_img1_Im100.png)

---
## Page 98

Evaluation Metrics for Music Classification
• Classification accuracy (top1, top3)
• Precision, recall, F-score
• ROC-AUC
– obtained by threshold sweeping
– “micro” vs “macro” average

https://music-classification.github.io/tutorial/part2_basics/evaluation.html

![Diagram/Image 101 on Page 98](images/page98_img1_Im101.png)

---
## Page 99

Tricks when Training DL Models
• Avoid overfitting
– Dropout
– Weight decay (L1, L2 norm of weights)
– Early stopping
– Reduce model size
– Data augmentation
• Overfitting is better than underfitting
– Scale up the model till it overfits
– Then try to mitigate overfitting
• Try different learning rates and optimizers

![Diagram/Image 102 on Page 99](images/page99_img1_Im102.png)

---
## Page 100

Album/Artist-Filtered vs. Random Splits
• Train/validation/test split
– Use training data for training, validation data for parameter tuning, and the (hidden) test data for evaluation
• Data leakage: song, album, and artist splits vs. random splits
– Clips from the same song/album/artist should NOT be found in multiple partitions
  ◦ NN models may memorize unique acoustic fingerprints rather than learning generalized genre, style, or emotion
– Album-split: mitigation of production leakage
– Artist-split: measuring out-of-distribution generalizability
  ◦ But this depends; e.g., we don’t do artist-split for singer classification

![Diagram/Image 103 on Page 100](images/page100_img1_Im103.png)

---
## Page 101

Class Imbalance & Confusion Matrix
• The problem: uneven, long-tailed distribution of real-world data
• The “accuracy paradox”: simply predicting everything as the majority classes would already give you good accuracy
• A naïve workaround: subsampling the majority class
• Use confusion matrix as a diagnostic tool
– See if the model is defaulting to majority classes
– Also check intuitively close classes
– Trick: “row-normalize” the confusion matrix to see the proportion of correctly classified instances per class regardless of class size

actual
predicted

![Diagram/Image 104 on Page 101](images/page101_img1_Im104.png)