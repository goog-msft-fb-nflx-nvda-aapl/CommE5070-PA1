## Bottom line

The literature points to **three distinct opportunities**, but they are not equally promising:

1. **Task 2:** Replace the current arithmetic probability sum with **OOF-trained, low-dimensional logit stacking / geometric pooling plus temperature calibration**. This is the cleanest way to exploit the complementary information you have without burning the 102-sample validation set. It may improve the point estimate; it cannot magically create statistical power when the paired error pattern is unchanged.

2. **Task 1:** The most genuinely different ordinal method worth testing is **cumulative-link / threshold ordinal classification such as CORAL or CORN**, preferably as a very small head on the strongest frozen representation. This is materially different from your year-regression and hierarchical-classifier experiments because it learns **one ordered latent score plus ordered thresholds**, rather than fitting six nominal class scores or a continuous year target. ([arXiv][1])

3. **LoRA:** Your result is quite plausibly explained by **adapter placement + optimization regime**, not by LoRA being intrinsically unsuitable. In particular, your “top 8 of 24 layers, attention projections only, r=8” setup is exactly the kind of restricted LoRA configuration that newer controlled work identifies as problematic. Recent work finds substantial advantages from applying LoRA to MLP/all weight matrices, using a substantially higher LoRA learning rate than FullFT, and avoiding large batches. ([Thinking Machines Lab][2])

For new models, there are **three concrete post-mid-2025 candidates** that are actually released: **Music Flamingo 2601**, **MuFun**, and **CultureMERT**. Of those, Music Flamingo is the most direct successor to your AF3 experiment; MuFun is the most architecturally independent; CultureMERT is the most directly motivated by your cross-cultural Task 2. ([NVIDIA][3])

---

# 1. Task 2: closing the fusion-vs-AF3 gap

### The important statistical point

There is no calibration or significance-test trick that can make a **+6.9 percentage-point accuracy difference on the same 102 cases** statistically decisive unless the new method changes enough individual predictions to change the paired-discordance structure.

That is because McNemar is already exploiting the paired nature of the comparison. Your fusion and AF3 are evaluated on exactly the same clips. A different confidence-interval construction cannot manufacture information that is absent from those 102 paired outcomes.

The practical implication is:

> **Optimize the fusion for better predictions, not for a different p-value.**

Temperature scaling, for example, does not normally change a model's own argmax predictions; its value here is that it can change the **relative influence of AF3 and the probe when the two are fused**. Temperature scaling uses only one parameter, which is one reason it is attractive in small-data settings. ([スプリンガー][4])

### What I would test instead of another fixed probability weight

Use **OOF logit stacking with a deliberately tiny meta-model**.

For each training clip, obtain out-of-fold predictions from:

* AF3 zero-shot
* your best trained probe

Then fit a constrained fusion of the form

$$
s_c =
\alpha\,\frac{z^{(\mathrm{probe})}_c}{T_p}
+
(1-\alpha)\,\frac{z^{(\mathrm{AF3})}_c}{T_a}
$$

and predict with `argmax(s)`.

That gives you only **three learned quantities**:

* \(0 \le \alpha \le 1\)
* AF3 temperature \(T_a\)
* probe temperature \(T_p\)

This is much safer than fitting a full multiclass stacking classifier on only 102 validation examples.

There is a useful interpretation here: this is a **log-linear pool / geometric opinion pool** rather than an arithmetic probability pool. If the two experts are giving probabilities that have substantially different sharpness, combining their log scores can behave better than averaging their probabilities.

Stacking itself is well established; the important methodological point is that the meta-learner must see **out-of-fold base predictions**, not predictions made by models that were trained on those same examples. ([PubMed][5])

### Why I prefer this to full stacking

A conventional multinomial stacker can easily become too flexible for your setting.

You have 798 training samples. A six-class meta-model with separate class-specific weights for both experts can consume a nontrivial fraction of the available statistical information, especially once you account for the fact that the two base predictions are strongly correlated.

The most defensible sequence is therefore:

**OOF temperature calibration → one-weight logit fusion → evaluate once on the fixed validation split.**

A richer variant is to add **six class intercepts**:

$$
s_c =
\alpha z^{(P)}_c/T_p +
(1-\alpha)z^{(A)}_c/T_a + b_c
$$

but I would treat that as a second experiment, not the first one.

### What about Dirichlet calibration?

Dirichlet calibration is real and well established. It generalizes temperature scaling by applying a learned multiclass transformation to the log probabilities, and has demonstrated better calibration than simple temperature scaling across several datasets. ([NeurIPS Papers][6])

For your problem, however, I would **not** make it the first choice. Its flexibility is useful when you have a substantial calibration set; your validation set is only 102, so fitting such a flexible calibration map directly on validation would make the subsequent accuracy estimate optimistic.

Use OOF training predictions if you experiment with it.

### A subtle but important alternative: conditional gating

Your Task 2 error analysis suggests the two systems may have **different strengths by regime**: US/UK/Germany are confusable, while Brazil is cleanly separated.

That makes a **gated fusion** more interesting than a global weight:

$$
\alpha(x)=\sigma(a+b\,d(x))
$$

where \(d(x)\) can be something extremely low-dimensional such as:

* AF3 entropy
* probe entropy
* absolute confidence difference
* whether the two models disagree

Then

$$
p(x)=\alpha(x)p_\mathrm{probe}(x)
+[1-\alpha(x)]p_\mathrm{AF3}(x).
$$

But I would only do this **after** the 3-parameter logit fusion. With 102 validation clips, a learned gate can very easily become a fancy overfit.

### What will *not* solve the statistical problem

I would not change to 5×2 cross-validation simply to obtain a more favorable significance test. Those tests are designed for repeated train/test resampling; your assignment's fixed artist-disjoint validation split defines the evaluation population. Replacing it with repeated resplits answers a different question.

Likewise, a Bayesian version of McNemar can produce a posterior probability rather than a p-value, and there is literature on doing this, but it does not create new information. ([Wiley Online Library][7])

---

# 2. Task 1: an ordinal method that is actually different from your existing experiments

Your observation that errors are ordinal is exactly the situation where a **cumulative-link ordinal classifier** is attractive.

The key distinction is:

### What you already tried

* regression to year
* coarse → fine hierarchy
* ordinary multiclass classification
* probably nominal probability outputs

### What CORAL/CORN do

They posit a latent scalar \(f(x)\) and learn ordered thresholds:

$$
P(y > k\mid x)=\sigma(f(x)-\theta_k)
$$

for \(k=1,\dots,K-1\).

The thresholds are constrained to preserve the ordering. Consequently, you do not learn six independent class scores.

This is a stronger inductive bias than your standard classifier but **not the same objective as year regression**.

CORAL and CORN are specifically designed to enforce rank consistency, and implementations are available in PyTorch. ([GitHub][8])

A related deep ordinal paper based on cumulative-link models explicitly reports improvements over nominal classification on ordinal problems, including distance-aware losses. ([サイエンスダイレクト][9])

### Why this is particularly appropriate for your dataset size

Suppose your frozen embedding is \(d\)-dimensional.

A linear six-way classifier has roughly:

$$
6d + 6
$$

parameters.

A proportional-odds / CORAL-style linear ordinal head has roughly:

$$
d + 5
$$

parameters.

That is a major reduction in the supervised head's degrees of freedom while preserving the exact ordinal structure.

With ~1,000 training clips, this is much more attractive than adding another flexible nonlinear classifier.

### I would use CORAL/CORN on the *best frozen representation*

I would not start by fine-tuning MERT.

The experiment with the cleanest attribution is:

> **Best frozen MERT representation → linear CORAL/CORN head → fixed validation evaluation.**

That answers a very specific question:

> Does the failure mode come from using a nominal six-way decision surface when the target has a latent ordered structure?

### A second ordinal idea worth knowing

There is newer work on **approximately unimodal ordinal likelihoods**, specifically motivated by the observation that many ordinal tasks are close to unimodal but not perfectly unimodal. The 2025 paper explicitly tries to avoid the bias introduced by forcing every conditional class distribution to be strictly unimodal. ([arXiv][10])

That is conceptually interesting for your Task 1 because your errors are concentrated around neighboring decades.

But I would still test CORAL/CORN first. It is much simpler and has much cleaner behavior to explain in a coursework report.

### One thing I would *not* do

Do not interpret “small validation set” as a reason to optimize directly against the validation metric until it crosses some desired significance threshold. That would turn your fixed validation set into a model-selection set.

The useful way to exploit small-data structure is through a **stronger inductive bias learned from training data**.

---

# 3. Why your LoRA result can be much worse than the Huang/Ding-Lerch result

This is the part where the recent evidence is unusually relevant.

The 2024 paper you cited, **Parameter-Efficient Transfer Learning for Music Foundation Models**, is arXiv:2411.19371. It tested MERT and MusicFM with probing, FullFT, Adapter, Prompt, Prefix, BitFit, SSF, and LoRA. ([arXiv][11])

For MERT, its GTZAN results were:

| Method  | GTZAN genre accuracy |
| ------- | -------------------: |
| Full FT |                63.8% |
| Probe   |                75.6% |
| Adapter |                72.8% |
| Prompt  |                74.5% |
| Prefix  |                70.0% |
| BitFit  |                73.8% |
| SSF     |                74.8% |
| LoRA    |                74.7% |

So their LoRA did **not** actually “beat” frozen probing on MERT. It was roughly tied with the other PEFT methods and about one percentage point below probing. ([arXiv][12])

That matters because your result:

> frozen probe ~47–50% vs LoRA ~38%

is qualitatively worse.

### The biggest difference: adapter placement

Your LoRA:

> rank 8, attention projections only, top 8/24 layers.

Newer controlled work directly attacks this design choice.

The September 2025 **LoRA Without Regret** study found that even in small-data settings, LoRA did better when applied to **all weight matrices**, with especially important contributions from MLP layers. Attention-only LoRA underperformed even when the researchers increased attention-only rank enough to roughly match parameter counts. ([Thinking Machines Lab][2])

Their conclusion is especially relevant to your setup:

> attention-only is not simply “LoRA with fewer parameters”; it can be a qualitatively worse adaptation location.

So your 0.17% trainable-parameter figure is not necessarily a virtue. You may simply be constraining the update to the wrong subspace.

### Second difference: learning-rate scale

The same 2025 study found that the optimal learning rate for LoRA was approximately **10× the optimal FullFT learning rate** in its supervised experiments. ([Thinking Machines Lab][2])

The authors also found that the optimal LR was fairly insensitive to rank over a wide range once rank was above very small values. ([Thinking Machines Lab][2])

This gives a very concrete explanation for a LoRA run that plateaus below a frozen probe:

> a FullFT-derived learning rate is not necessarily a sensible LoRA learning rate.

### Third difference: batch size

That paper also found a specific LoRA optimization pathology:

* LoRA became more sensitive to large batch size than FullFT.
* The loss gap grew as batch size increased.
* Smaller batches reduced the problem. ([Thinking Machines Lab][2])

So a small-data audio experiment with a relatively large effective batch can be in a bad regime even when the training curve looks perfectly stable.

### Fourth difference: short-run dynamics and initialization

**ALLoRA** identifies three proposed LoRA failure modes particularly relevant to short training:

* dropout may be a poor regularizer in short training episodes,
* zero initialization of one LoRA factor slows early learning,
* the conventional scaling factor creates unfavorable cross-layer interactions. ([arXiv][13])

That paper proposes a dropout-free, scaling-free adaptive-LR variant and reports gains over ordinary LoRA in its LLM experiments.

This is not direct evidence that ALLoRA will improve MERT on your assignment, but it is strong mechanistic evidence against the assumption that “ordinary LoRA at r=8” is a uniquely well-conditioned recipe for small datasets.

### Fifth difference: LoRA rank

Your rank-8 choice is not absurd; it is a standard configuration. But recent controlled work swept ranks as high as 512 and found that low ranks can become **capacity-limited**, especially after more training steps. ([thinkingmachines.ai][2])

So there are actually two plausible explanations for your result:

1. **under-capacity:** r=8 is too restrictive for the subset of MERT you are adapting;
2. **wrong subspace:** attention-only misses useful information that lives in MLP weights.

The literature gives stronger evidence for #2.

### There is also LoRA+

LoRA+ shows that using the same learning rate for the \(A\) and \(B\) factors can be suboptimal and that asymmetric learning rates can improve performance and training speed. The paper reports 1–2 percentage-point gains across its experiments and releases code. ([arXiv][14])

Again, it is not an audio-specific result, so I would treat it as a **second-order optimization intervention**, not the first thing to test.

### My interpretation of your LoRA failure

I would therefore revise the diagnosis to:

> **Your experiment is not evidence that “MERT + LoRA fails at 1k clips.” It is evidence that your particular low-capacity, attention-only, top-layer LoRA parameterization is a poor transfer mechanism for this dataset.**

That is a much narrower and more defensible conclusion.

---

# 4. New models actually worth testing

## A. Music Flamingo 2601 — highest-priority zero-shot replacement

This is the most direct new candidate.

**Paper:** arXiv:2511.10289, *Music Flamingo: Scaling Music Understanding in Audio Language Models*. ([Hugging Face][15])

**HF:** `nvidia/music-flamingo-2601-hf` ([Hugging Face][16])

**Code:** NVIDIA `audio-flamingo` repository. ([GitHub][17])

Music Flamingo is specifically trained for music understanding, including music reasoning, structure, timbre, cultural context, and long-form music. The released checkpoint is supported through Hugging Face Transformers and handles up to 20 minutes of audio. ([GitHub][18])

### Why it is relevant

You already have AF3.

Music Flamingo is **built on the AF3 backbone**, but receives substantial music-specific adaptation. ([GitHub][17])

Therefore:

* it is **not** an independent representation source in the way MERT vs MuQ are;
* it is an excellent **AF3 replacement**;
* it may produce better zero-shot answers on specifically musical concepts.

For Task 2 especially, the “cultural context” specialization makes it worth testing.

I would **not** automatically fuse AF3 and Music Flamingo. Their errors may be highly correlated because they share the AF3 backbone.

---

## B. MuFun — strongest new independent architecture

**Paper:** arXiv:2508.01178, *Advancing the Foundation Model for Music Understanding*. ([Hugging Face][19])

Released models include:

* `Yi3852/MuFun-Base`
* `Yi3852/MuFun-Instruct`

The HF base model is 9B parameters, Apache-2.0, and has released weights. ([Hugging Face][20])

**Code:** `laitselec/MuFun`. The released model processes audio plus text using a multimodal LLM architecture. ([Hugging Face][19])

### Why it is interesting for your problem

MuFun explicitly targets holistic music understanding and jointly models instrumental and lyrical information. ([Hugging Face][19])

That gives it a different inductive bias from:

* MERT
* MuQ
* AF3/Qwen2-Audio

So it is potentially the **best new source for a genuinely independent fusion signal**.

For your evaluation, I would use it **zero-shot**, not frozen-probe initially, because the released object is an audio-text-to-text model rather than a conventional embedding encoder. The HF model is also not deployed through an inference provider, so local inference is the practical route. ([Hugging Face][20])

---

## C. CultureMERT — particularly interesting for Task 2

**Paper:** *CultureMERT: Continual Pre-Training for Cross-Cultural Music Representation Learning*, ISMIR 2025. ([ISMIR 2025][21])

**HF:** `ntua-slp/CultureMERT-95M` and `ntua-slp/CultureMERT-TA-95M`. ([Hugging Face][22])

The model is based on MERT-v1-95M and was continually pretrained on 650 hours spanning Greek, Turkish, and Indian musical traditions. The authors report improvements on culturally diverse tagging benchmarks while retaining performance on Western-centric datasets. ([ISMIR 2025][21])

### Why it is relevant

Your Task 2 is fundamentally cross-cultural.

The broader 2025 foundation-model evaluation found that performance degrades as the target musical culture becomes more distant from the model's training distribution, and specifically reported that **Qwen2-Audio was strongest overall while MERT-330M could be worse than MERT-95M** on the tested world-music datasets. ([ResearchGate][23])

That makes CultureMERT a very sensible diagnostic:

> Does continued exposure to culturally diverse music improve the recoverability of your release-market signal?

I would expect it to be more interesting for **Task 2 than Task 1**.

---

## What I would *not* prioritize

**MULE** is worth knowing because the ISMIR 2025 cross-cultural evaluation reviewers specifically mentioned convolutional models such as MULE as a useful architecture expansion. ([ISMIR 2025][24])

But MULE is a **2022** model, not a post-mid-2025 model, so it does not answer your “newly relevant since mid-2025” question. It is however genuinely accessible: Pandora released weights and code, and there is a PyTorch/Hugging Face port. ([GitHub][25])

So I would classify MULE as a **cheap control**, not a new-foundation-model priority.

---

# 5. What does the literature say about the ceiling?

Here the most useful evidence is actually extremely recent.

## The 22k-track paper is real, and your comparison is directionally valid

The August 2026 paper:

**arXiv:2608.10980 — *Measuring Cross-Cultural Style Diffusion Through Era Classification: US and Korean Popular Music*** ([arXiv][26])

used:

* 31,092 unique Billboard chart tracks,
* audio for 22,002 of them,
* artist-aware splits,
* six decade classes,
* 30-second audio crops,
* test-time averaging,
* multiple CNN architectures. ([ResearchGate][27])

Its in-domain Billboard test results were:

| Architecture | Macro accuracy |
| ------------ | -------------: |
| CNN          |          67.0% |
| FCN          |          69.6% |
| SCNN         |      **71.2%** |
| SCNNR        |          68.1% |
| Musicnn      |          70.1% |
| CRNN         |          69.0% |

Across 18 runs they report **69.0 ± 2.0% macro accuracy** and **75.3 ± 1.5% micro accuracy**. ([ResearchGate][28])

That is a very useful external reference because the task has:

* the same number of decades,
* similar era prediction,
* artist-disjoint evaluation,
* 30-second audio segments.

The major difference is data scale and corpus definition.

The paper's training partition is roughly **17× your Task-1 training size**, not merely “a bit larger.”

So your observation that their 69–71% results come from a dataset around 20× larger is essentially correct.

## What this does *not* establish

It does **not** establish that the ceiling for your 1,026-sample Discogs dataset is, say, 60%.

The corpora are different:

* Billboard chart music is a highly selected population;
* your labels are Discogs-VI editorial release metadata;
* your train/test split is artist-disjoint;
* your 1980s market task is qualitatively different from era prediction.

So there is no defensible literature-derived scalar such as:

> “The ceiling at n=1,000 is 61.3%.”

The literature simply does not support that level of precision.

### What it *does* support

It supports three less exciting but more reliable conclusions.

**First:** decade classification has a strong real audio signal. The 2026 study's confusion matrices are concentrated near the diagonal rather than behaving like random six-class classification. ([ResearchGate][28])

**Second:** data volume matters substantially. A 17k-training example system reaching ~70% while your ~1k system reaches ~55% is entirely compatible with a strong data-scaling effect rather than evidence that your pipeline is defective.

**Third:** the signal is not uniformly strong across decades. In that paper the 2010s were much harder than several earlier decades; for example, per-class accuracy varied from roughly 29–51% for the 2010s depending on architecture. ([ResearchGate][28])

That is important for your case because a near-saturated six-way overall accuracy can still conceal substantial class-specific uncertainty.

---

# 6. Your validation-set size is itself a major limitation

For Task 1, 132 examples means that:

$$
1\text{ validation example} = 0.758\text{ percentage points}.
$$

So:

* 73/132 = **55.3%**
* 69/132 = **52.3%**

and a seemingly meaningful-looking improvement of 3 percentage points is only about **4 additional correct clips**.

Using a Wilson 95% interval:

* 73/132 → about **46.8–63.5%**
* 69/132 → about **43.8–60.6%**

For Task 2:

* 67/102 = **65.7%**
* 60/102 = **58.8%**

with approximate Wilson 95% intervals of:

* 67/102 → **56.1–74.2%**
* 60/102 → **49.1–67.9%**

Those intervals are not a substitute for your paired McNemar analysis, but they illustrate the amount of sampling noise you are dealing with.

The 2024 music-foundation-model study makes essentially the same methodological warning: on small MIR evaluation sets, bootstrap-based significance can simply fail to resolve modest performance differences. ([arXiv][12])

---

# 7. My recommended experiment order

Given everything you have already done, I would **not** spend the next round on another generic architecture.

### First experiment: Task 2

**OOF-calibrated 3-parameter logit fusion**

Use the 798 training examples to obtain OOF AF3 + probe scores, fit:

$$
s_c=
\alpha z_{probe,c}/T_p+
(1-\alpha)z_{AF3,c}/T_{AF3}
$$

then freeze it and evaluate once on the 102-example validation set.

This is the cleanest new fusion experiment because it directly addresses the strongest unresolved question while keeping the meta-model extremely small.

### Second experiment: Task 1

**Linear CORAL/CORN on the best frozen MERT representation.**

This is the clearest test of whether the observed ordinal geometry can be exploited more effectively than your existing year-regression/hierarchical approaches. ([GitHub][8])

### Third experiment: LoRA

Do **one recovery run**, not another broad sweep:

* higher rank than 8,
* MLP + attention rather than attention-only,
* substantially higher LR than your FullFT LR,
* modest batch size,
* no extra waveform augmentation initially.

That is the configuration most strongly motivated by the 2025 LoRA evidence. ([Thinking Machines Lab][2])

### Fourth: new foundation model

For a single new model, I would test **MuFun-Instruct** because it is the most independent new representation source. For a single new zero-shot replacement of AF3, test **Music Flamingo 2601**. For a Task-2-specific representation probe, test **CultureMERT-95M**. ([Hugging Face][20])

---

## Overall research conclusion

The evidence does **not** look like “you simply haven't found the magic architecture yet.”

It looks more like:

* **Task 1:** there is a real but noisy ordinal signal; a constrained ordinal head is still a logically distinct and worthwhile test, but the 132-example validation set makes small gains intrinsically hard to certify.
* **Task 2:** the fact that fusion beats your probe strongly but only modestly beats AF3 is consistent with **partially overlapping expert errors**. Calibration-aware logit fusion is the most principled remaining low-capacity intervention; there is no statistical procedure that can turn a +7-ish example gap into convincing evidence without changing the predictions.
* **LoRA:** your particular adapter parameterization is unusually restrictive relative to what newer PEFT evidence now recommends. Attention-only/top-layer r=8 is the part I would distrust, not LoRA as a concept.
* **Ceiling:** the newest comparable era-classification study supports **~70% as achievable with much more data**, but it does not justify a precise ~1k-sample ceiling. Your current 55% Task-1 result is consistent with being in a data-limited regime rather than conclusively being at the representational ceiling.
* **Task 2:** I found no directly comparable published benchmark for artist-disjoint six-way **release-market classification** across US/UK/Brazil/Spain/Germany/Italy in 1980s music. That absence is itself important: there is presently much weaker external evidence for what “normal” performance should be on this task.

### Checkable primary sources

* **Era classification:** arXiv **2608.10980**; GitHub `malerlab/billboard-melon-era`. ([arXiv][26])
* **Music PEFT:** arXiv **2411.19371**; GitHub `suncerock/peft-music`. ([arXiv][11])
* **LoRA+:** arXiv **2402.12354**; GitHub `nikhil-ghosh-berkeley/loraplus`. ([arXiv][14])
* **LoRA failure/optimization study:** *LoRA Without Regret*, Thinking Machines Lab, September 2025. ([thinkingmachines.ai][2])
* **ALLoRA:** arXiv **2410.09692**. ([arXiv][13])
* **Ordinal CORAL/CORN:** arXiv **1905.13392**; GitHub `Raschka-research-group/coral-pytorch`. ([arXiv][1])
* **Ordinal approximate-unimodality:** arXiv **2510.00122**. ([arXiv][10])
* **Dirichlet calibration:** arXiv **1910.12656**. ([arXiv][29])
* **Music Flamingo:** arXiv **2511.10289**; HF `nvidia/music-flamingo-2601-hf`; NVIDIA `audio-flamingo` GitHub. ([Hugging Face][15])
* **MuFun:** arXiv **2508.01178**; HF `Yi3852/MuFun-Base`, `Yi3852/MuFun-Instruct`; GitHub `laitselec/MuFun`. ([Hugging Face][19])
* **CultureMERT:** ISMIR 2025; HF `ntua-slp/CultureMERT-95M`, `ntua-slp/CultureMERT-TA-95M`. ([ISMIR 2025][21])

[1]: https://arxiv.org/abs/1905.13392?utm_source=chatgpt.com "Cumulative link models for deep ordinal classification"
[2]: https://thinkingmachines.ai/blog/lora/ "LoRA Without Regret - Thinking Machines Lab"
[3]: https://research.nvidia.com/labs/adlr/MF/?utm_source=chatgpt.com "Music Flamingo: Scaling Music Understanding in Audio Language Models - NVIDIA ADLR"
[4]: https://link.springer.com/article/10.1007/s10994-023-06336-7?utm_source=chatgpt.com "Classifier calibration: a survey on how to assess and improve predicted class probabilities | Machine Learning | Springer Nature Link"
[5]: https://pubmed.ncbi.nlm.nih.gov/29637384/?utm_source=chatgpt.com "Stacked generalization: an introduction to super learning - PubMed"
[6]: https://papers.nips.cc/paper_files/paper/2019/hash/8ca01ea920679a0fe3728441494041b9-Abstract.html?utm_source=chatgpt.com "Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with Dirichlet calibration"
[7]: https://onlinelibrary.wiley.com/doi/full/10.1002/sim.6875?utm_source=chatgpt.com "Improving and extending the McNemar test using the Bayesian method - Ogura - 2016 - Statistics in Medicine - Wiley Online Library"
[8]: https://github.com/Raschka-research-group/coral-pytorch?utm_source=chatgpt.com "GitHub - Raschka-research-group/coral-pytorch: CORAL and CORN implementations for ordinal regression with deep neural networks. · GitHub"
[9]: https://www.sciencedirect.com/science/article/abs/pii/S0925231220303805?utm_source=chatgpt.com "Cumulative link models for deep ordinal classification - ScienceDirect"
[10]: https://arxiv.org/abs/2510.00122?utm_source=chatgpt.com "Approximately Unimodal Likelihood Models for Ordinal Regression"
[11]: https://arxiv.org/abs/2411.19371 "[2411.19371] Parameter-Efficient Transfer Learning for Music Foundation Models"
[12]: https://arxiv.org/html/2411.19371 "Parameter-Efficient Transfer Learning forMusic Foundation Models"
[13]: https://arxiv.org/abs/2410.09692?utm_source=chatgpt.com "ALLoRA: Adaptive Learning Rate Mitigates LoRA Fatal Flaws"
[14]: https://arxiv.org/abs/2402.12354?utm_source=chatgpt.com "LoRA+: Efficient Low Rank Adaptation of Large Models"
[15]: https://huggingface.co/papers/2511.10289?utm_source=chatgpt.com "Paper page - Music Flamingo: Scaling Music Understanding in Audio Language Models"
[16]: https://huggingface.co/nvidia/music-flamingo-2601-hf?hardware=mi210&utm_source=chatgpt.com "nvidia/music-flamingo-2601-hf · Hugging Face"
[17]: https://github.com/NVIDIA/audio-flamingo?utm_source=chatgpt.com "GitHub - NVIDIA/audio-flamingo: PyTorch implementation of Audio Flamingo: Series of Advanced Audio Understanding Language Models · GitHub"
[18]: https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/musicflamingo.md?utm_source=chatgpt.com "transformers/docs/source/en/model_doc/musicflamingo.md at main · huggingface/transformers · GitHub"
[19]: https://huggingface.co/papers/2508.01178?utm_source=chatgpt.com "Paper page - Advancing the Foundation Model for Music Understanding"
[20]: https://huggingface.co/Yi3852/MuFun-Base?utm_source=chatgpt.com "Yi3852/MuFun-Base · Hugging Face"
[21]: https://ismir2025program.ismir.net/poster_210.html?utm_source=chatgpt.com "ISMIR 2025: CultureMERT: Continual Pre-Training for Cross-Cultural Music Representation Learning"
[22]: https://huggingface.co/ntua-slp/CultureMERT-95M?utm_source=chatgpt.com "ntua-slp/CultureMERT-95M · Hugging Face"
[23]: https://www.researchgate.net/publication/392917637_Universal_Music_Representations_Evaluating_Foundation_Models_on_World_Music_Corpora?utm_source=chatgpt.com "(PDF) Universal Music Representations? Evaluating Foundation Models on World Music Corpora"
[24]: https://ismir2025program.ismir.net/poster_213.html "ISMIR 2025: Universal Music Representations? Evaluating Foundation Models on World Music Corpora"
[25]: https://github.com/PandoraMedia/music-audio-representations?utm_source=chatgpt.com "GitHub - PandoraMedia/music-audio-representations: Results and Models for Learning Audio Representations of Music Content · GitHub"
[26]: https://arxiv.org/abs/2608.10980?utm_source=chatgpt.com "Measuring Cross-Cultural Style Diffusion Through Era Classification: US and Korean Popular Music"
[27]: https://www.researchgate.net/publication/412164281_Measuring_Cross-Cultural_Style_Diffusion_Through_Era_Classification_US_and_Korean_Popular_Music?utm_source=chatgpt.com "(PDF) Measuring Cross-Cultural Style Diffusion Through Era Classification: US and Korean Popular Music"
[28]: https://www.researchgate.net/publication/412164281_Measuring_Cross-Cultural_Style_Diffusion_Through_Era_Classification_US_and_Korean_Popular_Music "(PDF) Measuring Cross-Cultural Style Diffusion Through Era Classification: US and Korean Popular Music"
[29]: https://arxiv.org/abs/1910.12656?utm_source=chatgpt.com "Beyond temperature scaling: Obtaining well-calibrated multiclass probabilities with Dirichlet calibration"
