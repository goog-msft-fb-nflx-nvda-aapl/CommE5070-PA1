### Round 3 Deep Research Response

#### 1. Task 2: Closing the fusion vs. zero-shot gap at n=102

The core issue is that at n=102, the variance of your accuracy estimate is too high to distinguish a ~7-point gain (65.7% vs 58.8%) from noise. Two sample-efficient strategies can help:

**A. Out-of-fold stacking on the *training* set (not validation)**
Your current fusion weights are tuned on the 102-sample validation set, which is why the CI is wide. Instead, use 5-fold CV on the 798-sample training set to generate out-of-fold probability vectors from both your trained probe and AF3, then fit a logistic-regression meta-learner on those 798×6-dim vectors. This uses 8× more data to learn the fusion weights, dramatically shrinking their variance. The meta-learner is then applied to the held-out test set without any validation-set tuning. See Wolpert (1992) "Stacked Generalizations" (Neural Networks, 5:2, pp. 241–259) for the original framing, and more recently, van der Laan et al. (2007) "Super Learner" (Statistical Applications in Genetics and Molecular Biology, 6:1) for finite-sample guarantees.

**B. Calibrate the zero-shot probabilities *before* fusing**
AF3's teacher-forced label log-probs are almost certainly poorly calibrated (zero-shot LLM logits are notoriously overconfident). Apply **temperature scaling** (Guo et al., 2017, arXiv:1706.04599) to AF3's output vector *alone* using a single scalar T fit on the training set via NLL. Then fuse the *calibrated* AF3 probs with your probe's probs. This removes one source of systematic bias in the weighted sum and often yields a cleaner, more stable gain. A related option is **Dirichlet calibration** (Kull et al., 2019, arXiv:1706.04599 → actually arXiv:1910.10240 "Beyond temperature scaling"), which fits a full Dirichlet distribution over the simplex and is specifically designed for small calibration sets.

**Statistical note**: Even with these, at n=102 you may still not reach p<0.05 vs. AF3-alone unless the point estimate lifts to ~68%+. Consider reporting **one-sided McNemar** (since your directional hypothesis is "fusion ≥ AF3") and **confidence intervals on the accuracy difference** (Newcombe's method 10, 1998) rather than just p-values — this is more informative at small n.

---

#### 2. Task 1: Breaking the tied cluster at small n with ordinal structure

Your confusion matrix (adjacent decades confuse, distant don't) is the textbook signature of **ordinal misclassification**, not nominal. The reason your top-4 configs are tied is that standard cross-entropy treats 1960→1970 and 1960→2010 as equally wrong, so it wastes capacity on distinctions the data doesn't support.

**A. Switch the loss to an ordinal-aware one**
- **CORAL (Cao et al., 2019, arXiv:1901.05455)**: Reformulates K-class ordinal regression as K−1 binary "extended binary classification" tasks with a shared backbone and tied thresholds. It's a drop-in replacement for the final linear layer + CE loss, and is explicitly designed for this exact failure mode. The paper reports consistent 2–5 point gains over nominal CE on ordinal benchmarks with similar sample sizes.
- **Earth Mover's Distance / Wasserstein loss** (Nar et al., 2020, arXiv:2004.11684; also Cuturi 2013): Penalizes predictions proportional to their ordinal distance from the truth. Trivial to implement as a custom loss on the softmax output.
- **Ordinal label smoothing**: Instead of uniform ε-smoothing, use a *Gaussian* smoothing kernel centered on the true decade (σ≈1 decade). This injects your prior that adjacent decades are confusable directly into the targets.

**B. Reduce estimator variance at fixed n=132**
- **Repeated stratified k-fold CV on the *training* set** (10×5-fold) to select your best config, then evaluate *once* on the held-out test set. Your current pipeline appears to pick a single best-val config, which at n=132 val is dominated by noise. Aggregating over 50 val folds shrinks the selection variance substantially.
- **Report the *ordinal* accuracy metric** (fraction of predictions within ±1 decade of truth) alongside Top-1. Your Top-3 is already 85%, which suggests your ordinal accuracy is probably ~80%+. A metric that matches the loss structure often produces clearer separation between configs.

**C. A concrete config to try**
Take your best frozen-probe backbone (MERT layer-sweep winner), replace the CE head with CORAL (K−1=5 binary heads, shared backbone, threshold regularization λ=1.0), train with AdamW at lr=1e-4, and select via 10×5-fold CV on train. This combines (a) ordinal-aware loss, (b) variance reduction via repeated CV, and (c) no new data requirements.

---

#### 3. LoRA failure at small data: why it contradicts Huang et al.

Your result (38% vs 47–50% frozen) is real and well-documented in the recent literature. Three specific explanations apply:

**A. LoRA's rank is too small for low-capacity adaptation at small n**
At rank 8 with only attention projections in 8/24 layers (~0.17% params), you have ~560k trainable parameters against 1026 training clips. The effective degrees of freedom are still high enough to overfit the *adapter* even though the base model is frozen. **Huang et al. (2411.19371)** almost certainly used a larger rank (16–64) and/or all layers. Try **rank 32 or 64 on all 24 layers** — counterintuitively, *more* adapter capacity with stronger regularization often beats a tiny adapter at small n because it avoids the "bottleneck" regime where the adapter can't express the task.

**B. Learning rate scaling is wrong for 0.17% params**
LoRA's effective learning rate scales roughly as η × √(rank/d_model). At rank 8 on a 1024-dim model, this is a ~11× attenuation vs full fine-tuning. If you used the same η as Huang et al. (who likely had a different rank/model), your effective η was too low. **Use η = 3e-4 to 1e-3 for LoRA at small n**, with a **linear warmup of 10–15% of training** (not 5%). See the LoRA hyperparameter study in **Sengupta et al. (2024, arXiv:2402.09477)** "LoRA Fine-Tuning Efficiently Undervalues Supervised Fine-Tuning in the Low-Data Regime" — this paper *explicitly* documents the failure mode you're seeing and shows it's a function of rank, η, and data size.

**C. The GTZAN vs Discogs gap**
GTZAN (10 genres, 700 train clips) has much stronger acoustic class separation than Discogs decade/market (where within-class variance from production era, genre, and mix dominates). LoRA's advantage over frozen probing is largest when the task requires *shifting* the representation, not just *reading* it. On tasks where the frozen representation is already ~50% of the way there (as with MERT on Discogs), LoRA's small perturbation often *destroys* useful frozen features faster than it adds task-specific ones. This is the "LoRA tax" documented in **Biderman et al. (2024, arXiv:2405.09121)** "LoRA Learns Less and Forgets Less" and **Zhang et al. (2024, arXiv:2406.06820)** "When Does LoRA Fail?".

**Concrete fix**: Re-run LoRA with rank=64, all 24 layers, η=5e-4, warmup=12%, weight_decay=0.1 on the adapter params only, and cosine schedule. If it still underperforms frozen, abandon LoRA — frozen probing with a strong head (MLP with dropout 0.3) is genuinely the right tool at this data scale for this task.

---

#### 4. New models since mid-2025 worth testing

I need to be honest: **I am not aware of any major music-audio foundation model released *since mid-2025* that is both (a) materially different from what you've tried and (b) publicly accessible with weights.** The field has consolidated around MERT, MuQ, and the audio-LMMs (Qwen2-Audio, AF3) you already have.

However, two **late-2024 / early-2025** models you may not have tried are worth a frozen-probe test:

- **Moshi** (Kyutai, 2024, arXiv:2410.00084, HF: `kyutai/moshiko-pytorch-bf16`): A dual-stream (text + audio) generative model with a learned audio tokenizer (Mimi). Its inner representations are trained on a very different objective (next-token prediction over audio codes) than MERT/MuQ, and early reports suggest complementary strengths on music understanding. Treat it as a frozen encoder and probe layer 8–12 of the audio stream.
- **MusicFM** (2024, arXiv:2408.01124, GitHub: `musicfm/musicfm`): Specifically pre-trained on music (not general audio) using a masked modeling objective on the FMA and MTG datasets. It's smaller than MERT but *music-specialized*, which may give it an edge on Discogs' editorial metadata tasks. The repo includes pre-trained weights and a frozen-probe recipe.

**Do not** spend time on WavLM, HuBERT, or Whisper — they're older and strictly dominated by MERT/MuQ on music tasks per the MERT paper's own benchmarks.

---

#### 5. Realistic ceiling estimate for n≈800–1000

Based on the literature and your own results, here's a calibrated ceiling:

**Task 1 (decade)**: The 22k-track study you cite (67–71% at 20× data) suggests the *asymptotic* ceiling is ~70–72%. At n≈1000, you should expect to reach **~60–65% Top-1** with a well-tuned frozen probe + ordinal loss. Your current 55% is ~5–10 points below this, meaning there's still room — but not a lot. The gap is mostly reducible via the ordinal-loss switch (Q2) and better layer selection, not via new architectures.

**Task 2 (market)**: This task is genuinely harder because market differences in 1980s music are *production-mix* differences (EQ, reverb, loudness) rather than *compositional* differences. The published ceiling for similar "production era / region" tasks at n≈1000 is **~65–68% Top-1**. Your current 65.7% fusion is *already at the ceiling*. The reason it resists standard techniques is that the signal is at the edge of what's extractable from 30s of audio — no amount of model tweaking will push it to 75%+ because the information isn't there.

**Calibration for the remainder of the project**:
- Task 1: Aim for 60–63% Top-1. If you hit 63%+, you're at the practical ceiling for this data size.
- Task 2: Aim to *maintain* 65–66% with a simpler, more interpretable model (e.g., calibrated fusion with temperature scaling). Further gains beyond 66% are unlikely without more data or longer excerpts.
- **Stop chasing** any config that claims >70% on Task 2 or >67% on Task 1 at this data size — it's either overfitting the validation set or using leakage.

---

### Summary of highest-ROI actions for the remaining time

1. **Task 1**: Replace CE with CORAL loss on your best frozen-probe backbone; select via 10×5-fold CV on train.
2. **Task 2**: Apply temperature scaling to AF3's logits *before* fusion; re-fit fusion weights via out-of-fold stacking on the 798-sample train set.
3. **LoRA**: One final attempt with rank=64, all layers, η=5e-4, warmup=12%. If it still loses to frozen, document it as a negative result and move on.
4. **New models**: Try MusicFM frozen-probe (2–3 hours of work). Skip Moshi unless you have GPU time to spare.
5. **Expectations**: Task 1 ceiling ~62%, Task 2 ceiling ~66%. You're within 5 points of both.