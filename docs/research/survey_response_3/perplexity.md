## 1. Task 2 (market): fusion strategies that are more sample‑efficient than a fixed weighted sum

Yes. Two families of methods are consistently shown to improve discriminative performance *and* calibration with very small held‑out sets, and both are more flexible (and often more sample‑efficient) than a single fixed linear weight:

### (a) Stacking with a simple meta‑classifier (logistic regression / Platt‑scaled LR)

- **What to do:** treat each base model’s 6‑dim probability vector as features and train a **multiclass logistic regression** (or one‑vs‑rest LR) on your 102‑sample validation split to learn the fusion. Then evaluate on the 102‑sample test split.  
- **Why this helps at n≈100:** stacking with a linear meta‑learner is explicitly studied in small‑sample audio/medical settings and can outperform individual classifiers and naive averaging when the base models make complementary errors. Because the meta‑model has only 6×6=36 input features (and you can regularize heavily), it is far less prone to overfitting than adding more base models. [pubmed.ncbi.nlm.nih](https://pubmed.ncbi.nlm.nih.gov/41847683/)
- **Calibration bonus:** logistic regression outputs are inherently better calibrated than arbitrary weighted sums; you can add **Platt scaling** (or temperature scaling for softmax) on top if you want to optimize calibration separately. Recent work shows that **post‑hoc Platt scaling can substantially reduce ECE even with modest calibration sets**, though variance increases as the calibration set shrinks. [clawrxiv](https://clawrxiv.io/abs/2604.01103)

**Practical recipe for your n=102 val / n=102 test setup**

1. On the 102‑sample *validation* split, collect:
   - `p_trained[i] ∈ ℝ⁶` from your best frozen‑probe classifier  
   - `p_AF3[i] ∈ ℝ⁶` from Audio Flamingo 3 zero‑shot  
2. Fit a **multinomial logistic regression** with strong L2 regularization (e.g. `C=0.1` or smaller) to predict the true label from `[p_trained, p_AF3]` (12 features).  
3. Optionally, refit a **temperature scalar** or **Platt scaling** on the same 102 samples to calibrate the stacked outputs. [clawrxiv](https://clawrxiv.io/abs/2604.01103)
4. Evaluate on the 102‑sample test split and run the same paired bootstrap / McNemar tests you already use.

This is a *different* fusion strategy from your current fixed weighted sum and is explicitly designed to exploit complementarities while controlling overfit via regularization. [pubmed.ncbi.nlm.nih](https://pubmed.ncbi.nlm.nih.gov/41847683/)

### (b) Geometric‑mean / log‑odds ensembling instead of arithmetic mean

If you want something even lighter than stacking, recent calibration literature shows that **geometric mean of probabilities** (or equivalently, averaging in log‑probability / logit space) can outperform simple averaging on both discrimination and calibration, especially when base models are miscalibrated. [clawrxiv](https://clawrxiv.io/abs/2604.01103)

- **Key result:** a Monte Carlo study of ensemble aggregation found that **simple averaging degrades calibration**, while the **geometric mean consistently improves it** and often improves ranking metrics too. [clawrxiv](https://clawrxiv.io/abs/2604.01103)
- **Small‑sample relevance:** geometric mean ensembles are used in Kaggle‑style audio competitions with limited validation data and have been reported to give small but consistent gains over weighted arithmetic means. [zenn](https://zenn.dev/yuto_mo/articles/86e2db028e141c)

**How to try it**

- Replace your current fused probability:
  \[
  p_{\text{fuse}} = \alpha \, p_{\text{trained}} + (1-\alpha) \, p_{\text{AF3}}
  \]
  with a geometric mean:
  \[
  p_{\text{fuse}} \propto p_{\text{trained}}^{\alpha} \cdot p_{\text{AF3}}^{1-\alpha}
  \]
  (normalize to sum to 1).  
- Tune \(\alpha\) on your 102‑sample validation set (grid search over e.g. \(\{0.0, 0.1, \dots, 1.0\}\)), then evaluate on test.

This is a *different functional form* of fusion than your current linear weighted sum, and theory + empirical work suggest it can be more robust when models disagree and are imperfectly calibrated. [clawrxiv](https://clawrxiv.io/abs/2604.01103)

***

## 2. Task 1 (decade): techniques for small‑validation, ordinal‑ish, near‑saturated classification

Your confusion analysis (adjacent decades confuse, distant ones don’t) strongly suggests that **ordinal regression losses** and **ordinal calibration** can give you a clearer margin than plain cross‑entropy, especially when several CE‑trained configs are statistically tied.

### (a) Ordinal‑aware loss functions

Two closely related families are directly relevant:

1. **Class Distance‑Weighted Cross‑Entropy (CDW‑CE)**  
   - Adds a penalty proportional to the *distance* between predicted and true class, so mispredicting 1960s as 1980s costs more than 1960s→1970s. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12787087/table/jcm-15-00365-t003/)
   - Formally, for true class \(y\) and predicted distribution \(p\), the loss includes a term like:
     \[
     \mathcal{L}_{\text{CDW}} = -\sum_{k} w_{y,k} \log p_k,\quad w_{y,k} = 1 + \lambda \cdot |y-k|
     \]
     where \(\lambda\) controls how strongly distance matters. [arxiv](https://arxiv.org/html/2412.01246v2)
   - This directly encodes your empirical finding that “adjacent errors are less bad” and has been shown to improve both accuracy and ordinal consistency in medical ordinal tasks. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12787087/table/jcm-15-00365-t003/)

2. **Neighbor‑order / adjacent‑class penalties**  
   - Additional regularization terms that penalize non‑unimodal or non‑adjacent probability mass (e.g., high probability on 1960s and 1990s but low on 1970s/1980s). [ieeexplore.ieee](https://ieeexplore.ieee.org/iel8/6287639/10820123/10869448.pdf)
   - These encourage the network’s output to respect the global order and reduce “jumping over” intermediate decades.

**Why this matters for your situation**

- With n=132 validation samples, many CE‑trained configs will cluster around similar top‑1 accuracies but differ in *how* they err. Ordinal losses reweight those errors so that models that make “closer” mistakes are favored, often yielding a config with a **clearer separation** in your validation metric space. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12787087/table/jcm-15-00365-t003/)

### (b) Ordinal‑aware calibration

Recent work shows that standard cross‑entropy leads to **miscalibrated, non‑unimodal** outputs in ordinal tasks, and proposes **ordinal‑aware calibration losses** that enforce both calibration and unimodality. [semanticscholar](https://www.semanticscholar.org/paper/Calibration-of-ordinal-regression-networks-Kim-Chung/c30dd9fc4733fbdcddee13d6a1500bfdaba22139)

- These losses add terms that encourage the predicted probability vector to be **unimodal around the true rank** and to have confidence matching observed frequencies. [semanticscholar](https://www.semanticscholar.org/paper/Calibration-of-ordinal-regression-networks-Kim-Chung/c30dd9fc4733fbdcddee13d6a1500bfdaba22139)
- In practice, you can:
  - Train with CDW‑CE or a similar ordinal loss, then  
  - Apply a **post‑hoc ordinal calibration** step (or a custom loss that includes an ordinal calibration term) on your 132‑sample validation set. [semanticscholar](https://www.semanticscholar.org/paper/Calibration-of-ordinal-regression-networks-Kim-Chung/c30dd9fc4733fbdcddee13d6a1500bfdaba22139)

Given your near‑saturated performance and tied CIs, moving from “plain CE + argmax” to “ordinal loss + ordinal‑calibrated probabilities” is one of the few remaining axes likely to produce a **statistically cleaner winner** rather than another point in the same cluster.

***

## 3. Why LoRA on MERT underperformed frozen probing (and what to try instead)

Your observation (LoRA ≈ 38% vs frozen probe ≈ 47–50% Top‑1 on ~800–1000 clips) is actually consistent with several emerging findings about PEFT at small data scales.

### (a) Known failure modes of LoRA / PEFT on tiny datasets

Recent surveys and practical guides highlight:

- **Overfitting on tiny datasets:** adapters are cheap to train, so teams often fine‑tune on 100–1,000 examples and see severe overfitting (high train acc, low val acc). [systemoverflow](https://www.systemoverflow.com/learn/ml-llm-genai/llm-fine-tuning/failure-modes-and-edge-cases-in-peft-systems)
- **Insufficient capacity / wrong targets:** setting rank too low or targeting too few layers can cause underfitting; conversely, too much adaptation capacity on tiny data causes overfitting. [systemoverflow](https://www.systemoverflow.com/learn/ml-llm-genai/llm-fine-tuning/failure-modes-and-edge-cases-in-peft-systems)
- **Domain shift sensitivity:** PEFT assumes the base model already has strong representations for your domain; if the domain gap is large, small adapters can fail catastrophically. [systemoverflow](https://www.systemoverflow.com/learn/ml-llm-genai/llm-fine-tuning/failure-modes-and-edge-cases-in-peft-systems)

In audio specifically, a 2026 survey of PEFT for foundation models notes that **adapter‑based and LoRA methods recover substantial fractions of full fine‑tuning performance, but their relative benefit depends strongly on task, data size, and acoustic match**. Another systematic review of PEFT in speech processing emphasizes that **PEFT effectiveness is highly dependent on model size and data regime**, with smaller models sometimes benefiting more from full fine‑tuning and PEFT working best when the test distribution closely matches training. [contemporaryjournal](https://contemporaryjournal.com/index.php/14/article/view/1707)

Your MERT LoRA setup (rank 8, top 8 layers only, ~0.17% params) sits exactly in the regime where:

- The adapter is very small (good for data efficiency) but  
- The task (Discogs‑VI editorial decade/market labels) may be **farther from MERT’s pretraining distribution** (EnCodec+CQT teacher on general audio) than GTZAN genre labels are, increasing the effective domain gap. [systemoverflow](https://www.systemoverflow.com/learn/ml-llm-genai/llm-fine-tuning/failure-modes-and-edge-cases-in-peft-systems)

### (b) Hyperparameter regimes known to matter more at small scale

Recent practical guidance for LoRA/QLoRA/DoRA in 2025–2026 highlights several knobs that are especially important when data is scarce:

1. **Higher LoRA rank or broader targeting**  
   - Underfitting from “rank too low / too few layers” is a common LoRA failure mode on small data. [systemoverflow](https://www.systemoverflow.com/learn/ml-llm-genai/llm-fine-tuning/failure-modes-and-edge-cases-in-peft-systems)
   - For small datasets, some guides suggest **starting with r≈16** (not 8) and targeting **all linear layers in attention and MLP**, not just a subset. [mixpeek](https://mixpeek.com/guides/fine-tuning-with-lora-adapters)
   - A 2026 guide notes that for tone/style adaptation, **500–2,000 curated examples** can work well with LoRA, but you must ensure sufficient capacity and regularization. [futureagi](https://futureagi.com/blog/fine-tuning-llms-unlocking-peak-performance/)

2. **DoRA (Weight‑Decomposed LoRA)**  
   - DoRA decomposes weights into **magnitude and direction**, applying low‑rank updates to the directional component while learning a separate magnitude vector. [huggingface](https://huggingface.co/papers/2402.09353)
   - Empirically, DoRA often **matches or exceeds full fine‑tuning performance** with LoRA‑level parameter counts and improves training stability. [huggingface](https://huggingface.co/papers/2402.09353)
   - HuggingFace PEFT supports DoRA via `LoraConfig(use_dora=True)`. [hysenlabs](https://hysenlabs.com/en/projects/nvlabs-dora)
   - For small data, recommendations include:
     - Use **lower rank than LoRA** (e.g. if LoRA r=16, try DoRA r=8–12)  
     - Slightly **higher dropout** (0.05–0.1 vs 0–0.05)  
     - **Lower learning rate** (≈0.5–0.7× LoRA LR)  
     - Target **all linear layers** in attention and MLP. [medium](https://medium.com/@AntonioVFranco/qdora-explained-the-new-peft-standard-for-2025-5cf59afeb6ba)

3. **Regularization and schedule**  
   - Strong weight decay (0.01–0.1), adapter dropout (0.1–0.3), and early stopping are explicitly recommended for datasets under 1,000 examples to combat overfitting. [systemoverflow](https://www.systemoverflow.com/learn/ml-llm-genai/llm-fine-tuning/failure-modes-and-edge-cases-in-peft-systems)
   - Adversarial training in the embedding space has been shown to **boost LoRA/Adapter robustness in low‑data regimes** by focusing adversarial signals in the restricted PEFT subspace. [en.papernotes](https://en.papernotes.org/ACL2026/llm_efficiency/small_data_big_noise_adversarial_training_for_robust_parameter-efficient_fine-tu/)

**Concrete experiments to try on MERT**

Given your current LoRA config (r=8, top 8 layers, 15–40 epochs):

1. **DoRA on MERT**  
   - Use `LoraConfig(use_dora=True, r=8–12, lora_alpha=2×r, target_modules=all linear in attention+MLP, lora_dropout=0.05–0.1)` and a **lower LR** than your LoRA run. [medium](https://medium.com/@AntonioVFranco/qdora-explained-the-new-peft-standard-for-2025-5cf59afeb6ba)
   - This directly tests whether magnitude/direction decomposition improves sample efficiency on your ~800–1,000‑clip tasks. [huggingface](https://huggingface.co/papers/2402.09353)

2. **Higher‑rank, broader‑target LoRA + strong regularization**  
   - Try **r=16–24**, target **all attention+MLP linear layers**, with **weight decay 0.01–0.1**, **dropout 0.1–0.3**, and **early stopping** on validation loss. [systemoverflow](https://www.systemoverflow.com/learn/ml-llm-genai/llm-fine-tuning/failure-modes-and-edge-cases-in-peft-systems)
   - Compare against your current r=8, top‑8‑layers config to see if capacity/targeting was the bottleneck.

3. **Adversarial PEFT (SDBN‑style)**  
   - If you’re comfortable adding embedding‑space adversarial perturbations during PEFT, recent work shows **large gains for LoRA/Adapter in low‑data, noisy settings**. [en.papernotes](https://en.papernotes.org/ACL2026/llm_efficiency/small_data_big_noise_adversarial_training_for_robust_parameter-efficient_fine-tu/)
   - This is more experimental but directly targets the “small data, big noise” regime you’re in.

These directions are distinct from your original LoRA setup and align with 2025–2026 findings on PEFT failure modes and fixes at small scale. [systemoverflow](https://www.systemoverflow.com/learn/ml-llm-genai/llm-fine-tuning/failure-modes-and-edge-cases-in-peft-systems)

***

## 4. New music‑audio foundation models since mid‑2025 worth testing

As of late 2026, the **core music‑audio foundation models** with public weights remain dominated by:

- **MERT** (e.g. `m-a-p/MERT-v1-330M`) – EnCodec+CQT teacher pretraining [existing in your stack]  
- **MuQ** (e.g. `OpenMuQ/MuQ-large-msd-iter`, MuQ‑MuLan variants) – Mel‑RVQ pretraining on music [existing in your stack]  
- **Audio‑language models** like **Qwen2‑Audio‑7B‑Instruct** and **NVIDIA Audio Flamingo 3** – used by you in zero‑shot mode

My searches do **not** surface a clearly new, widely adopted *music‑specific* foundation model with HuggingFace weights released after mid‑2025 that is both:

- Designed for **music understanding** (not just generation), and  
- Directly comparable to MERT/MuQ for **frozen‑probe classification** on Discogs‑VI metadata.

The most notable “new” items are:

- **Stable Audio 3** with LoRA/DoRA trainers for *generation* (style adaptation via LoRA/DoRA), not for discriminative MIR tasks like decade/market classification. [fal](https://fal.ai/models/fal-ai/stable-audio-3-trainer)
- Various **speech‑oriented PEFT papers** (GC‑LoRA, DoRAN, etc.) that propose adapter architectures but do not release new *music* foundation models. [arxiv](https://www.arxiv.org/abs/2510.04331)

**Actionable takeaway:**  
There is **no obvious, accessible music‑foundation model post‑mid‑2025** that clearly supersedes MERT/MuQ for your exact tasks and is available as HF weights for frozen‑probe or zero‑shot evaluation. The most promising “new model” angle is therefore **not a new backbone**, but **new adapter architectures** (DoRA, DoRAN, GC‑LoRA) applied to your existing MERT/MuQ models. [hysenlabs](https://hysenlabs.com/en/projects/nvlabs-dora)

If you want to be exhaustive, you could:

- Search HuggingFace for tags like `music`, `audio`, `self-supervised` with dates 2025–2026 and inspect repos for classification benchmarks, but based on current literature, nothing stands out as a clear next candidate beyond MERT/MuQ for your setting.

***

## 5. Sanity check on practical ceilings for ~800–1,000‑clip music classification

Your external benchmark (~22k tracks, 67–71% macro accuracy for decade classification) and your own results (Task 1 best ≈ 55% Top‑1, Task 2 best ≈ 66% Top‑1 with fusion) suggest you are **not yet at an absolute ceiling**, but you are likely in a regime where **diminishing returns are strong**.

### What the literature implies about ceilings at your scale

- **Small‑sample audio classification studies** (e.g., Parkinson’s audio datasets with 80–252 samples) show that careful feature selection + stacking can reach **~87–90% accuracy**, but those are **binary or low‑cardinality tasks** with highly discriminative handcrafted features and very controlled recording conditions. [pubmed.ncbi.nlm.nih](https://pubmed.ncbi.nlm.nih.gov/41847683/)
- **Music decade classification** with deep models on larger datasets (thousands of tracks) typically reports **~67% overall accuracy** with macro F1 ≈ 0.66 using relatively simple architectures. Your Task 1 performance (≈ 52–55% Top‑1 on 6 classes) is below that, which is expected given: [medium](https://medium.com/@yashnavani05/predicting-musics-timeline-deep-learning-with-the-million-song-dataset-yash-navani-f4604e15cfde)
  - Much smaller training set (≈ 1k vs tens of thousands)  
  - Artist‑disjoint splits (harder generalization)  
  - 30‑second excerpts (less context than full tracks)

- **PEFT and small‑data guidance** consistently notes that below ~1,000 examples, both full fine‑tuning and PEFT can struggle to generalize unless the domain match is excellent and regularization is strong. In ASR, **~10 hours** of data is described as a “sweet spot” for LoRA; below ~1 hour, neither PEFT nor full fine‑tuning generalizes well. Your 800–1,000 clips × 30s ≈ 6.7–8.3 hours, so you’re near the lower edge of that “workable” zone. [systemoverflow](https://www.systemoverflow.com/learn/ml-llm-genai/llm-fine-tuning/failure-modes-and-edge-cases-in-peft-systems)

**Reasonable ceiling estimates for your setting**

Given:

- 6‑way classification (random = 16.7%)  
- Artist‑disjoint splits  
- 30‑second excerpts  
- Training sets of ~800–1,000 clips  

A realistic **practical ceiling** (with well‑tuned models, ordinal losses, and smart ensembling) is likely:

- **Task 1 (decade):** ~60–65% Top‑1, Top‑3 ~85–90%  
- **Task 2 (market):** ~65–70% Top‑1, Top‑3 ~85–90%

You are already at **~52–55% / 65–66% Top‑1**, so there is probably **5–10 percentage points** of headroom on Task 1 and **0–5 points** on Task 2, but **large jumps (e.g., to 75%+)** would be surprising without more data or a substantially better backbone.

This aligns with the broader pattern that **small‑dataset music classification** tends to plateau in the **60–70% Top‑1 range** for 6‑way tasks unless the dataset is much larger or the labels are easier to infer from short excerpts. [pubmed.ncbi.nlm.nih](https://pubmed.ncbi.nlm.nih.gov/41847683/)

***

## Concrete next experiments (prioritized)

**Task 2 (market)**  
1. Implement **stacking with multinomial logistic regression** on your 102‑sample validation set using `[p_trained, p_AF3]` as features, with strong L2 regularization; optionally add Platt scaling. [pubmed.ncbi.nlm.nih](https://pubmed.ncbi.nlm.nih.gov/41847683/)
2. Try **geometric‑mean fusion** with a tuned \(\alpha\) instead of your current arithmetic weighted sum. [clawrxiv](https://clawrxiv.io/abs/2604.01103)

**Task 1 (decade)**  
3. Retrain your best architectures with **CDW‑CE** (class distance‑weighted cross‑entropy) or an equivalent ordinal loss that penalizes distant misclassifications more heavily. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12787087/table/jcm-15-00365-t003/)
4. Add an **ordinal‑aware calibration** term or post‑hoc calibration step to encourage unimodal, order‑consistent probability vectors. [semanticscholar](https://www.semanticscholar.org/paper/Calibration-of-ordinal-regression-networks-Kim-Chung/c30dd9fc4733fbdcddee13d6a1500bfdaba22139)

**LoRA / PEFT on MERT**  
5. Run **DoRA** on MERT for both tasks with:
   - `r=8–12`, `lora_alpha=2×r`, `use_dora=True`  
   - Target **all linear layers** in attention + MLP  
   - Lower LR than your LoRA runs, dropout 0.05–0.1, strong weight decay. [medium](https://medium.com/@AntonioVFranco/qdora-explained-the-new-peft-standard-for-2025-5cf59afeb6ba)
6. As a control, run a **higher‑rank, broader‑target LoRA** (r=16–24, all linear layers) with aggressive regularization and early stopping to test the capacity/targeting hypothesis. [systemoverflow](https://www.systemoverflow.com/learn/ml-llm-genai/llm-fine-tuning/failure-modes-and-edge-cases-in-peft-systems)

These are all **distinct** from your previously tried configs and are directly motivated by 2024–2026 literature on small‑sample ensembling, ordinal classification, and PEFT failure modes. [pubmed.ncbi.nlm.nih](https://pubmed.ncbi.nlm.nih.gov/41847683/)

***

### Key sources

- Stacking ensembles & small‑sample audio/medical classification: [pubmed.ncbi.nlm.nih](https://pubmed.ncbi.nlm.nih.gov/41847683/)
- Ensemble aggregation & calibration (geometric mean, Platt scaling): [clawrxiv](https://clawrxiv.io/abs/2604.01103)
- Ordinal losses (CDW‑CE, neighbor‑order penalties): [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12787087/table/jcm-15-00365-t003/)
- Ordinal calibration: [semanticscholar](https://www.semanticscholar.org/paper/Calibration-of-ordinal-regression-networks-Kim-Chung/c30dd9fc4733fbdcddee13d6a1500bfdaba22139)
- PEFT surveys & failure modes (LoRA under/overfitting, domain shift): [systemoverflow](https://www.systemoverflow.com/learn/ml-llm-genai/llm-fine-tuning/failure-modes-and-edge-cases-in-peft-systems)
- DoRA / weight‑decomposed LoRA: [medium](https://medium.com/@AntonioVFranco/qdora-explained-the-new-peft-standard-for-2025-5cf59afeb6ba)
- Music decade classification benchmarks: [medium](https://medium.com/@yashnavani05/predicting-musics-timeline-deep-learning-with-the-million-song-dataset-yash-navani-f4604e15cfde)
- Small‑data PEFT guidance (10‑hour “sweet spot”, <1h struggles): [afriklang](https://afriklang.com/blog/full-fine-tuning-vs-peft-for-african-language-asr-when-10-hours-of-data-is)

If you want, I can sketch concrete PyTorch/PEFT code snippets for CDW‑CE, geometric‑mean fusion, or DoRA on MERT next.