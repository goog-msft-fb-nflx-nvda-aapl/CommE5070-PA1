I treated the pasted markdown as the research brief, including the six specific questions about release-decade/market classification, newer encoders, ALM transfer, and low-resource fine-tuning. 

## Bottom line

The research points to a fairly specific diagnosis:

**Task 1 and Task 2 are probably not the same kind of acoustic problem.** Decade has strong, globally persistent production/timbral signatures. Release market is a noisier proxy for geography, distribution, language, label ecosystem, and mastering/production practice. Discogs itself defines release country as the market where the release was sold/distributed, not where it was recorded or manufactured. ([Discogs][1])

That makes your repeated pattern—MERT fine-tuning/TTA/ensembling helping Task 1 but not Task 2—quite plausible rather than evidence that the experiments are simply badly tuned.

The most interesting things to try now are:

1. **MuQ and MuQ-MuLan as frozen probes**, especially MuQ-MuLan for Task 2.
2. **Music Flamingo's exposed audio embeddings**, not just generation.
3. **Teacher-logit distillation from AF3/Music Flamingo into a small MERT/MuQ classifier.**
4. **LoRA/top-layer-only MERT fine-tuning with LLRD**, rather than another full fine-tune.
5. **An explicit investigation of what “market” is acoustically correlated with**, including label/version/mastering leakage and class-pair confusion.

I would **not** spend another large compute budget on an unmodified full-MERT fine-tune yet.

---

# 1. Literature directly relevant to your tasks

### A. Music era classification: there is now unusually close literature

A 2024 paper, *Music Era Recognition Using Supervised Contrastive Learning and Artist Information*, explicitly formulates music-era recognition as classification and uses supervised contrastive learning. Their audio-only model reaches **54% accuracy within a ±3-year tolerance on MSD**; incorporating artist information gives another ~9% improvement. ([arXiv][2])

The paper is useful for your Task 1 because it identifies exactly the phenomenon your results exhibit: nearby eras are intrinsically difficult because differences in instrumentation/style can be subtle. It also motivates **supervised contrastive loss**, which you have not apparently tried on MERT embeddings. ([arXiv][2])

However, its protocol is **not directly comparable** to your six-decade artist-disjoint task, so I would use its *methodology*, not its 54% number as a benchmark.

[Music Era Recognition paper](https://arxiv.org/abs/2407.05368?utm_source=chatgpt.com)

### B. A very recent 2026 paper is particularly relevant to your artist-disjoint constraint

*Measuring Cross-Cultural Style Diffusion Through Era Classification: US and Korean Popular Music* is an August 2026 preprint/ISMIR 2026 paper. It explicitly uses **artist-aware splitting** and trains small CNNs from scratch so that the learned temporal representation is not contaminated by cross-cultural pretraining. On Billboard audio, six architectures achieve roughly **67.0–71.2% decade macro accuracy**, with pooled macro accuracy **69.0 ± 2.0%**. ([ResearchGate][3])

This is probably the closest methodological reference I found for your **artist-disjoint Task 1**.

The important conceptual point is that their authors deliberately avoided MERT/Jukebox because internet-scale pretraining could already contain music from the target cultures, which would confound their particular scientific question. That does **not** mean MERT is inappropriate for your assignment; it means pretrained representations can encode cultural information before your classifier ever sees the training labels. ([ResearchGate][3])

Their use of 30-second crops and artist-aware evaluation also makes your setup substantially more defensible than many older music-era experiments.

[2026 cross-cultural era-classification paper](https://arxiv.org/abs/2608.10980?utm_source=chatgpt.com)

### C. Older era literature explains why Task 1 is intrinsically learnable

Serrà et al. demonstrated long-term changes in pitch, timbre and loudness across decades of Western popular music, including homogenization of timbre and increasing loudness. ([Nature][4])

Interiano et al. analyzed **500,000+ UK releases from 1985–2015** and found systematic multi-decadal changes in acoustic attributes. ([DOI][5])

So there is good independent evidence that decade leaves a broad acoustic footprint.

That is fundamentally different from asking:

> “In which country was this particular commercial release marketed?”

which is not itself an acoustic property.

---

# 2. What I found for release-country / release-market classification

I found substantially less directly comparable work.

The closest older work is Gómez, Haro & Herrera's *Music and Geography: Content Description of Musical Audio from Different Parts of the World*, which studied geographical/cultural categories from audio and reported **86.68%** accuracy for Western-vs-non-Western classification using 23 audio features. ([ISMIR 2009][6])

But that task is extremely different from yours.

More recently, a 2025 paper, *Geographic-Origin Music Classification from Numerical Audio Features*, reports **99.53% SVM accuracy** and 98.58% CNN accuracy on the UCI Geographical Origin of Music dataset. The dataset contains 1,059 tracks from 33 non-Western regions. ([Journal UAD][7])

I would **not** use those numbers as a sanity benchmark for Task 2. Your labels are:

> commercial release market

whereas that literature is largely:

> geographic/cultural origin of the music.

Discogs explicitly reinforces this distinction: release country is where the release was sold/distributed, not where it was manufactured. ([Discogs][1])

### My confidence: high

I did not find a published benchmark that is genuinely close to:

**six-way release-market classification + Western commercial recordings + all 1980s + artist-disjoint.**

So your Task 2 may genuinely be a relatively unusual problem.

---

# 3. Why Task 2 is resisting improvements

I think there are **four structural reasons**, with fairly high confidence.

### 3.1 You removed Task 1's strongest cue by conditioning everything on the 1980s

Task 1 permits the model to exploit:

* instrumentation evolution
* recording technology
* production conventions
* dynamic range/loudness
* synthesis technology
* mix/mastering trends
* drum sounds
* frequency balance
* vocal production
* general timbral palette

Those all have substantial temporal drift documented in the literature. ([Nature][4])

Task 2 holds the decade constant, so much of that variance becomes nuisance variation.

### 3.2 Release market is a noisy proxy for culture

A US release and a UK release can contain essentially the same musical style, language, performer, instruments and recording technique.

Conversely, two releases in the same market can be acoustically very different.

This is not merely theoretical: Discogs' definition means the label reflects **commercial distribution**, not necessarily the origin of the audio. ([Discogs][1])

That weakens the mapping:

**audio → label**

considerably.

### 3.3 The remaining signal may be more semantic/contextual than timbral

This is the part I find particularly interesting given your AF3 result.

Your AF3 zero-shot model gets approximately **58.8%**, substantially above the frozen MERT probe for Task 2, despite being worse on Task 1.

That is exactly the sort of result I would expect if the useful representation contains things such as:

* language/accent
* genre/subgenre associations
* regional instrumentation
* musical-cultural associations
* production conventions
* label/market priors
* higher-level descriptions of the recording

rather than just low-level timbre.

That is where a music-aware audio-language model can plausibly outperform MERT.

### 3.4 The apparent differences among your Task-2 models are partly obscured by the small validation set

Your Task 2 validation set has only **102 examples**.

For an accuracy around 47%, the binomial standard error is roughly **4.9 percentage points**, before accounting for model-selection effects.

So a result such as 47.1% vs 44.1% is not strong evidence that one representation is genuinely better. You ideally want **paired bootstrap confidence intervals or paired permutation/McNemar-style testing on the same examples**, rather than reading too much into 2–3 percentage-point movements.

This is especially relevant because you have already observed a recurring “technique helps Task 1, not Task 2” pattern.

---

# 4. New encoders I would actually test

## 4.1 MuQ — highest-priority conventional encoder

**Model ID:** `OpenMuQ/MuQ-large-msd-iter`

MuQ was introduced in 2025 specifically as a self-supervised music representation model. Its training objective uses Mel residual vector quantization rather than the target mechanism used by MERT. The authors report improvements over prior music SSL models across a range of MIR tasks. ([arXiv][8])

The released model is about **300M parameters**, requires **24-kHz audio**, and is directly usable for hidden-state extraction. ([GitHub][9])

That makes it unusually convenient for your existing WAV format.

### Why it matters for you

This is the cleanest experiment:

> Same dataset, same classifier, same splits, replace MERT with MuQ.

That isolates **pretraining objective / representation** rather than changing the whole pipeline.

**Confidence: high.**

[MuQ paper](https://arxiv.org/abs/2501.01108?utm_source=chatgpt.com)
[MuQ GitHub](https://github.com/tencent-ailab/MuQ?utm_source=chatgpt.com)

---

## 4.2 MuQ-MuLan — particularly interesting for Task 2

**Model ID:** `OpenMuQ/MuQ-MuLan-large`

This is a ~700M music-text joint embedding model trained contrastively on music-text pairs. ([GitHub][9])

For your particular problem, this is conceptually important because Task 2 may require **cultural/semantic information** that a pure audio SSL model does not emphasize.

You should not assume it will win—the training objective is not release-market prediction—but I would absolutely test:

```text
MuQ-MuLan audio embedding
       ↓
PCA / small bottleneck
       ↓
LogReg / linear SVM
```

and then:

```text
MuQ + MuQ-MuLan
       ↓
concatenate after dimensionality reduction
       ↓
linear classifier
```

The combination could be more useful than either alone.

**Confidence: medium-high for Task 2; medium for Task 1.**

---

## 4.3 Music Flamingo — strongest new ALM experiment

**Model ID:** `nvidia/music-flamingo-2601-hf`

This is an **8B** model specifically adapted for music understanding, built on AF3. It was trained with extensive music data and explicitly targets music characteristics including harmony, structure, timbre, lyrics and cultural context. ([Hugging Face][10])

Most importantly for your project, its implementation exposes **audio hidden states / projected audio embeddings** and a `get_audio_features` path. ([Hugging Face][11])

That means you do not need to treat it as merely:

> prompt → generated text.

You can use:

> audio → Music Flamingo representation → classifier.

This is probably the most interesting way to attack your observed AF3 phenomenon.

**Confidence: high as an experiment; unknown whether it will outperform AF3 on your particular labels.**

The model is currently released under NVIDIA's **OneWay Noncommercial License**, so check that against your course's rules before using it in a graded submission. ([Hugging Face][10])

[Music Flamingo model](https://huggingface.co/nvidia/music-flamingo-2601-hf?utm_source=chatgpt.com)

---

## 4.4 CLaMP 3 — unusually relevant to the “market/culture” hypothesis

**Model ID:** `sander-wood/clamp3`

CLaMP 3 is a 2025 multilingual music representation system trained on **2.31M music-text pairs**, metadata spanning **194 countries**, and 27 languages, with generalization to 100 languages. It explicitly supports audio and zero-shot classification. ([Sanderwood][12])

The important caveat is that its audio path uses **MERT features**, so it is not a completely independent acoustic backbone. ([Hugging Face][13])

Nevertheless, its contrastive alignment to multilingual metadata makes it a fascinating experiment for Task 2.

I would test:

```text
audio
 → CLaMP3 audio embedding
 → classifier
```

and especially:

```text
audio → CLaMP3
                 \
                  concatenate → classifier
                 /
audio → MERT
```

**Confidence: medium-high for Task 2; low-medium for Task 1.**

[CLaMP 3 project](https://sanderwood.github.io/clamp3/?utm_source=chatgpt.com)
[CLaMP 3 weights](https://huggingface.co/sander-wood/clamp3?utm_source=chatgpt.com)

---

## 4.5 Audio Flamingo Next — current generalist follow-up

**Model ID:** `nvidia/audio-flamingo-next-hf`

AF-Next is an 8B successor in the Audio Flamingo line, with stronger speech/sound/music training, longer context and newer audio-language modeling. ([Hugging Face][14])

For your experiment, I would treat it primarily as:

> “Can the improvement persist with a newer generalist ALM?”

rather than assuming it is a better music encoder than Music Flamingo.

**Confidence: medium.**

[Audio Flamingo Next](https://huggingface.co/nvidia/audio-flamingo-next-hf?utm_source=chatgpt.com)

---

# 5. How I would exploit the AF3 result

This is the area where I see the largest potential improvement.

You already have an unusually valuable teacher:

**AF3: 58.8% Task 2**

while your MERT pipeline is ~47%.

Do not throw that information away by treating AF3 as merely a competing classifier.

## A. Soft-label distillation

For each training example obtain AF3's six-class probability vector:

$$
p_T(y|x)
$$

Then train your MERT classifier against both the hard label and the teacher distribution:

$$
L=(1-\lambda)CE(y,p_S)+
\lambda T^2 KL(p_T^T\Vert p_S^T)
$$

I would start with:

```text
temperature T = 2
lambda = 0.5
```

and later test T ∈ {2, 4}, λ ∈ {0.25, 0.5, 0.75}.

The useful information is not only the teacher's top prediction. A teacher saying

```text
US 0.55
UK 0.31
Germany 0.06
...
```

contains substantially more information than a hard pseudo-label of `US`.

There is current audio literature supporting representation/logit distillation and cross-model distillation as useful strategies for audio classification. CMKD, for example, shows that CNN and transformer models can act as effective teachers for one another; recent work also uses multi-layer foundation-model representations as distillation targets. ([DOI][15])

**Confidence: high.**

---

## B. Teacher embedding + MERT embedding

Since AF3 exposes audio hidden states, you can extract:

```text
MERT embedding       → 256/128 dims
AF3/MF embedding     → 256/128 dims
                         ↓
                    concatenate
                         ↓
                    MLP / LogReg
```

Do **not** concatenate the raw thousands-dimensional representations and train a large MLP on 798 samples.

For your dataset size I would force both branches through small bottlenecks.

For example:

```text
MERT      → Linear → 128
AF3       → Linear → 128
                       ↓
                    256 dims
                       ↓
                 Linear → 64
                       ↓
                  6 classes
```

**Confidence: medium-high.**

---

## C. Teacher prediction as an auxiliary feature

An even cheaper experiment:

```text
MERT embedding
      +
AF3 6-class probability vector
      ↓
small classifier
```

This tests whether the AF3 gain is simply complementary decision-boundary information.

It costs almost no training compute.

**Confidence: high as a diagnostic, medium as a final model.**

---

# 6. Improve the zero-shot ALM experiment itself

Your AF3 result may still be leaving performance on the table.

You reported that one Task-2 prompt reached 58.8% but also had substantial invalid free-form outputs. I would stop using free-form generation as the primary evaluation.

For each recording, evaluate the six class hypotheses independently:

```text
The audio is most consistent with a commercial release
targeted at the US market.

Answer only: yes or no.
```

Do the same for UK, Brazil, Spain, Germany and Italy.

Then construct six calibrated scores.

Even better, use **multiple semantically equivalent prompts** and average the normalized scores.

For example:

```text
Which release market is most consistent with this audio?
```

versus

```text
Which commercial music market is this recording most acoustically
and culturally associated with?
```

versus

```text
Which of these six markets would be the most plausible release
market for this recording?
```

Do not let the model see a different answer set ordering each time. Randomize class order across runs, then map back to canonical class IDs.

This is likely more useful than another classifier sweep because your AF3 result already says the model contains signal that your MERT classifier is not using.

---

# 7. Fine-tuning MERT: what I would change

Your full fine-tune:

> all 330M parameters, lr 2e-5, 15 epochs

is exactly the regime in which a ~800-sample dataset can over-specialize.

I would test **three modifications together**, rather than six independent tricks.

### Configuration

**Train only the upper 6–8 MERT layers + classifier.**

Use:

```text
layers 0–15/18 frozen
upper layers trainable
classifier trainable
```

Then use **layer-wise learning-rate decay** for the trainable layers, approximately:

```text
top layer       2e-5
next            1.8e-5
...
bottom trainable layer ~1e-5
```

LLRD is explicitly used in modern audio/speech transformer fine-tuning to stabilize adaptation and reduce overfitting; AudioMAE's standard fine-tuning recipe, for example, exposes layer decay around 0.75. ([DeepWiki][16])

Recent 2026 work also emphasizes parameter-efficient adapters for audio spectrogram transformers in low-resource settings. ([サイエンスダイレクト][17])

### LoRA

For a second run:

```text
LoRA rank r = 8
alpha = 16
dropout = 0.1
```

on attention projections, initially:

```text
q_proj
v_proj
```

and optionally:

```text
k_proj
o_proj
```

Do **not** LoRA every matrix in the first experiment. With only ~800 examples, that makes the experiment harder to interpret.

**Confidence: medium-high.**

There is broad recent evidence that PEFT can outperform or stabilize full fine-tuning of large audio foundation models, but I did not find evidence specifically showing “LoRA fixes six-way release-market classification.” That narrower claim would be speculative. ([サイエンスダイレクト][17])

---

# 8. Augmentation: one important caveat for your problem

I would use stronger augmentation than you did, but **not blindly**.

For generic audio classification, time/frequency masking and mixup are common fine-tuning tools. AudioMAE exposes frequency masking, time masking and mixup in its fine-tuning recipe. ([DeepWiki][16])

For **Task 1**, those are fairly safe.

For **Task 2**, there is a problem:

some of the very characteristics you are trying to classify may be:

* spectral balance
* mastering characteristics
* compression
* production style
* vocal treatment
* language-specific frequency patterns

So aggressive frequency masking could delete exactly the market signal.

I would start with:

```text
random crop: yes
small gain perturbation: yes
small additive noise: yes
time masking: mild
frequency masking: very mild
heavy codec augmentation: no initially
```

And importantly, **do not normalize away global loudness / spectral characteristics before establishing your baseline**, because those may actually be useful Task-1 cues and potentially some Task-2 production cues.

---

# 9. One experiment I think you have probably overlooked

## Recover the artist grouping from Discogs-VI if possible

This may matter more than another model.

Your current limitation is:

> no artist IDs in your anonymized manifest → plain StratifiedKFold for model selection.

But the original Discogs-VI metadata itself contains artist/release metadata. ([GitHub][18])

If your anonymization preserves any stable filename/track/release identifier, try joining it back to the original Discogs-VI metadata.

That would let you do:

```text
training set
    ↓
GroupKFold / StratifiedGroupKFold by artist
    ↓
model selection
```

instead of ordinary StratifiedKFold.

For a dataset this small, model-selection leakage is potentially significant.

The fact that the 2026 era paper explicitly emphasizes artist-aware splitting is another reason I would treat this as an important methodological issue. ([ResearchGate][3])

**Confidence: high that it is worth investigating; uncertain whether your anonymization permits the join.**

[Discogs-VI dataset/code](https://github.com/MTG/discogs-vi-dataset?utm_source=chatgpt.com)

---

# 10. A diagnostic experiment specifically for Task 2

Before spending more GPU time, calculate **pairwise separability**.

Train six one-vs-one classifiers on fixed MERT embeddings:

```text
US vs UK
US vs Brazil
US vs Spain
...
Germany vs Italy
```

and report ROC-AUC for every pair.

This gives you a matrix such as:

| Pair       | AUC |
| ---------- | --: |
| US–UK      |   ? |
| US–Brazil  |   ? |
| US–Spain   |   ? |
| US–Germany |   ? |
| US–Italy   |   ? |
| UK–Brazil  |   ? |
| ...        | ... |

This can reveal whether Task 2 is actually:

```text
Brazil/Spain easy
        +
US/UK/Germany/Italy intrinsically overlapping
```

rather than a generic six-way problem.

It would also tell you where AF3's additional signal lies.

For example, suppose AF3's improvement comes almost entirely from Brazil/Spain. Then the next experiment should investigate cultural/language representations, not generic encoder capacity.

---

# 11. Another diagnostic I strongly recommend: market-label integrity

Because Discogs release country means **commercial market**, I would explicitly inspect:

### Multiple-market versions

If the same recording/master has releases in both:

```text
US
UK
Germany
Italy
...
```

then the acoustic content may be nearly identical while the label differs.

That creates an **upper bound** on audio-only market classification.

Check whether identical or near-identical masters occur under multiple release-market labels.

You can do this using:

```text
audio fingerprint / chroma similarity / CLAP similarity
```

between samples in different market classes.

If you find many near-duplicate recordings across labels, a 47–60% Task-2 result could actually be quite impressive because the target variable is partly external to the audio.

This is also consistent with the Discogs-VI provenance: its core dataset contains version relationships across recordings, so the underlying metadata naturally contains many related/derivative versions. ([Zenodo][19])

**Confidence: high that this check is worth doing.**

---

# 12. What I would *not* prioritize

I would currently deprioritize:

**More MERT classifier sweeps.** You've already explored layers, PCA, SVM/logistic/MLP and full fine-tuning extensively.

**Hierarchical decade classification for Task 1.** Your own result already indicates better ordinal structure but lower raw accuracy; there is no obvious reason to spend more compute there.

**Demucs source separation for Task 2.** Your mixture result already strongly dominates vocals/accompaniment-only.

**Large architectural changes to the classifier head.** The bottleneck increasingly looks like the representation/target rather than a final linear-vs-MLP issue.

**More TTA for Task 2.** Your observed non-benefit is consistent with a label that is not strongly localized in time.

---

# 13. My proposed experimental sequence

Given your approximately two usable H200s, I would structure the next phase like this:

### Experiment 1 — MuQ frozen probe

```text
OpenMuQ/MuQ-large-msd-iter
→ layer sweep
→ mean pooling
→ LogReg + SVM
```

Run on **both tasks**.

This is the cleanest representation-replacement experiment.

### Experiment 2 — MuQ-MuLan

```text
OpenMuQ/MuQ-MuLan-large
→ audio embedding
→ linear probe
```

especially Task 2.

### Experiment 3 — Music Flamingo features

```text
nvidia/music-flamingo-2601-hf
→ get_audio_features()
→ mean/attention pooling
→ linear probe
```

Do this **before fine-tuning**.

### Experiment 4 — AF3/Music-Flamingo distillation

```text
student = MERT or MuQ
teacher = AF3/Music Flamingo

loss =
  0.5 * hard-label CE
+ 0.5 * teacher KL
```

### Experiment 5 — MERT PEFT

```text
top 6–8 layers
+ LLRD
or
LoRA r=8
```

with mild augmentation.

### Experiment 6 — CLaMP3

```text
sander-wood/clamp3
→ audio embedding
→ linear probe
```

especially Task 2.

### Experiment 7 — market-pair diagnostics / duplicate analysis

This is the experiment most likely to explain *why* Task 2 tops out.

---

# 14. Overall interpretation

### Task 1

Your **52.3% / 85.6%** ensemble result is consistent with the literature showing that decade has measurable, distributed acoustic signatures. The fact that your multi-crop/TTA and MERT fine-tuning helped is also coherent with that interpretation.

I would try **MuQ + supervised contrastive learning** next.

### Task 2

Your **47.1% / 75.5%** supervised result should not automatically be interpreted as “MERT is bad at geography.”

The target is not geography in the conventional MIR sense. It is a **commercial release-market variable**. ([Discogs][1])

The AF3 result strongly suggests that additional information exists in a representation capable of broader music/cultural reasoning. The new Music Flamingo and CLaMP3 models are particularly interesting because they explicitly incorporate richer music-text/cultural signals. ([Hugging Face][10])

So my working hypothesis would be:

> **Task 2 is representation-limited and label-noisy, rather than primarily capacity-limited.**

That is why I would switch from:

**“How do I fine-tune MERT harder?”**

to:

**“What information does AF3 have that MERT discards, and can I transfer it?”**

That is also the hypothesis most directly testable with your existing results.

**First executable experiment: MuQ frozen probing; should I give you the exact `MuQ-large-msd-iter` extraction script or the AF3/Music-Flamingo feature-extraction script first?**

[1]: https://support.discogs.com/hc/de/articles/360005006754-Datenbank-Richtlinien-7-Land?utm_source=chatgpt.com "Datenbank-Richtlinien 7. Land – Discogs"
[2]: https://arxiv.org/abs/2407.05368 "Music Era Recognition Using Supervised Contrastive Learning and Artist Information"
[3]: https://www.researchgate.net/publication/412164281_Measuring_Cross-Cultural_Style_Diffusion_Through_Era_Classification_US_and_Korean_Popular_Music?utm_source=chatgpt.com "(PDF) Measuring Cross-Cultural Style Diffusion Through Era Classification: US and Korean Popular Music"
[4]: https://www.nature.com/articles/srep00521?utm_source=chatgpt.com "Measuring the Evolution of Contemporary Western Popular Music | Scientific Reports"
[5]: https://doi.org/10.1098/rsos.171274?utm_source=chatgpt.com "Musical trends and predictability of success in contemporary songs in and out of the top charts | Royal Society Open Science | The Royal Society"
[6]: https://ismir2009.ismir.net/proceedings/OS9-5.pdf?utm_source=chatgpt.com "10th International Society for Music Information Retrieval Conference (ISMIR 2009)"
[7]: https://journal2.uad.ac.id/index.php/biste/article/view/13400?utm_source=chatgpt.com "Geographic-Origin Music Classification from Numerical Audio Features: Integrating Unsupervised Clustering with Supervised Models | Buletin Ilmiah Sarjana Teknik Elektro"
[8]: https://arxiv.org/abs/2501.01108?utm_source=chatgpt.com "MuQ: Self-Supervised Music Representation Learning with Mel Residual Vector Quantization"
[9]: https://github.com/tencent-ailab/MuQ?utm_source=chatgpt.com "GitHub - tencent-ailab/MuQ: Official repository of the paper \"MuQ: Self-Supervised Music Representation Learning with Mel Residual Vector Quantization\". · GitHub"
[10]: https://huggingface.co/nvidia/music-flamingo-2601-hf?hardware=mi210&utm_source=chatgpt.com "nvidia/music-flamingo-2601-hf · Hugging Face"
[11]: https://huggingface.co/docs/transformers/model_doc/musicflamingo?utm_source=chatgpt.com "Music Flamingo · Hugging Face"
[12]: https://sanderwood.github.io/clamp3/ "CLaMP 3 Home Page"
[13]: https://huggingface.co/sander-wood/clamp3/blame/50feb15f7e8a597b631f9ebc8782ed6c03f439f6/README.md?utm_source=chatgpt.com "README.md · sander-wood/clamp3 at 50feb15f7e8a597b631f9ebc8782ed6c03f439f6"
[14]: https://huggingface.co/nvidia/audio-flamingo-next-hf?utm_source=chatgpt.com "nvidia/audio-flamingo-next-hf · Hugging Face"
[15]: https://doi.org/10.1109/tpami.2025.3644853?utm_source=chatgpt.com "CMKD: CNN/Transformer-Based Cross-Model Knowledge Distillation for Audio Classification"
[16]: https://deepwiki.com/facebookresearch/AudioMAE/4.2.3-fine-tuning-for-other-audio-tasks?utm_source=chatgpt.com "Fine-tuning for Other Audio Tasks | facebookresearch/AudioMAE | DeepWiki"
[17]: https://www.sciencedirect.com/science/article/pii/S0950705125019318?utm_source=chatgpt.com "Parameter-efficient target-specialized audio spectrogram transformers via selective cross-domain knowledge distillation - ScienceDirect"
[18]: https://github.com/MTG/discogs-vi-dataset?utm_source=chatgpt.com "GitHub - MTG/discogs-vi-dataset: Discogs-VI dataset and code · GitHub"
[19]: https://zenodo.org/records/13983028?utm_source=chatgpt.com "Discogs-VI: A musical version identification dataset based on public editorial metadata | Zenodo"
