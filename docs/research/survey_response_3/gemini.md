## **Task 2: Statistical Significance & Sample-Efficient Fusion**

At $n \= 102$, discrete McNemar testing on Top-1 accuracy lacks statistical power because a 6.9 percentage point gain corresponds to a difference of only 7 test samples. The following sample-efficient fusion strategies and continuous testing methodologies can establish statistical significance without adding data.

### **Sample-Efficient Fusion Strategies**

* **Out-of-Fold Temperature Scaling**: Large Audio-Language Models like Audio Flamingo 3 (AF3) generate overconfident, uncalibrated class probability distributions relative to linear probes. Apply temperature scaling to AF3 logits prior to fusion:  
  $$p\_i \= \\text{softmax}\\left(\\frac{z\_i}{T}\\right)$$  
  Optimize $T$ via $K$-fold cross-validation on the training set (or internal validation folds) using negative log-likelihood. Calibrating probabilities before weighted averaging prevents AF3's raw confidence scores from overwhelming probe predictions.  
* **Logit-Space Regularized Linear Fusion**: Instead of probability-space weighted averaging ($\\alpha P\_{\\text{AF3}} \+ (1-\\alpha) P\_{\\text{probe}}$), concatenate raw logits and fit a Logistic Regression layer with strong $L\_2$ regularization ($\\text{C} \\in \[10^{-2}, 10^{-1}\]$) on out-of-fold validation predictions. This learns per-class bias corrections for market pairs prone to confusion (e.g., US vs. UK).  
* **Reciprocal Rank Fusion (RRF)**: Compute fused scores using rank positions rather than raw probabilities:  
  $$\\text{Score}(c) \= \\frac{1}{k \+ r\_{\\text{AF3}}(c)} \+ \\frac{1}{k \+ r\_{\\text{probe}}(c)}$$  
  Setting $k \\approx 60$ eliminates probability scale mismatches and produces robust ensemble ordering on small validation sets.

### **High-Power Statistical Tests**

To prove that fusion outperforms AF3 alone without increasing $n$:

* **Paired Permutation / Wilcoxon Signed-Rank Test on Log-Loss or Brier Score**: Replace binarized $0/1$ accuracy metrics with continuous scoring rules such as Mean Brier Score:  
  $$\\text{BS} \= \\frac{1}{N} \\sum\_{i=1}^N \\sum\_{k=1}^K (p\_{i,k} \- y\_{i,k})^2$$  
  Evaluating continuous prediction quality across test samples using a paired Wilcoxon signed-rank test or 10,000-sample paired bootstrap provides substantially higher statistical power than a binarized McNemar test.

## **Task 1: Breaking the Ordinal Decade Classification Tie**

With $n \= 132$, four top configurations are tied at $49\\%\\text{--}55\\%$ Top-1 accuracy due to coarse metric evaluation on ordinal errors.

### **Ordinal-Specific Loss Functions**

* **Gaussian Target Smoothing (Soft-Label Cross-Entropy)**: Replace standard one-hot decade labels with a Gaussian distribution over adjacent decades:  
  $$y\_k \\propto \\exp\\left(-\\frac{(k \- c)^2}{2\\sigma^2}\\right)$$  
  Setting $\\sigma \= 0.8\\text{--}1.0$ penalizes predicting the 1970s for a 1960s track far less than predicting the 2010s, forcing the probe to learn continuous temporal feature representations.  
* **CORAL (Consistent Rank Logits)**: Reformulate the 6-class decade classification into $K \- 1 \= 5$ binary classification tasks ($Y \> 1960\\text{s}, Y \> 1970\\text{s}, \\dots, Y \> 2000\\text{s}$) sharing a single weight vector with task-specific bias offsets ([Cao et al., 2020](https://arxiv.org/abs/1901.07884?utm_source=gemini)). This guarantees ordinal monotonicity across latent features.  
* **Earth Mover's Distance (EMD) Loss**: Train downstream classification heads using Wasserstein / EMD loss to explicitly penalize the distance between cumulative predicted probabilities and cumulative ground-truth targets.

### **Metric Resolution Improvement**

* **Evaluate Mean Absolute Error in Decades (MAE)**: Report MAE alongside Top-1:  
  $$\\text{MAE} \= \\frac{1}{N} \\sum\_{i=1}^N \\vert{}\\hat{y}\_i \- y\_i\\vert{}$$  
  Pairwise $t$-tests or bootstrap confidence intervals on MAE easily differentiate models that make off-by-one errors from those that make catastrophic off-by-several-decade errors, resolving ties that Top-1 masks.

## **Analysis of LoRA Fine-Tuning Failures on Small Audio SSL Datasets**

The underperformance of LoRA MERT fine-tuning ($38\\%$ vs. $47\\%\\text{--}50\\%$ frozen probe) stems from **representation collapse** and **hyperparameter mismatch** specific to small-scale audio SSL fine-tuning.

       \[ Base MERT-v1-330M \]  \---\> Pre-trained Latent Geometry (Stable)  
                 |  
  \+--------------+--------------+  
  |                             |  
\[ Frozen Probe \]           \[ Default LoRA \]  
  • Preserves global SSL     • Updates Q/V projections directly  
    feature manifold         • Overfits \~580K params on \~800 clips  
  • High Generalization      • Distorts internal attention space  
  • Test Top-1: 47-50%       • Test Top-1: 38% (Representation Collapse)

### **Key Failure Mechanisms**

> 1. **Latent Manifold Distortion**: Pre-trained MERT-v1-330M representations contain high-dimensional acoustic and timbral geometry across 24 self-attention layers. Updating query ($W\_q$) and value ($W\_v$) projections with only $\\sim 800$ clips causes catastrophic over-adaptation of attention distributions, distorting the pre-trained feature space.  
> 2. **Learning Rate Mismatch**: Standard LoRA learning rates ($\\sim 10^{-3}\\text{--}10^{-4}$) designed for text or large speech datasets are too high for $N \\approx 800$ audio clips, driving adapter parameters into sharp local minima.

### **Corrective Hyperparameter Regime for Small-Data Audio PEFT**

* **Target Modules**: Apply LoRA across all linear layers ($W\_q, W\_k, W\_v, W\_o, \\text{FFN}$) rather than $W\_q/W\_v$ alone, or switch to **IA3** (Infused Adapter by Inhibiting and Amplifying Inner Activations) or **BitFit** (bias-only fine-tuning). IA3 updates only activation-scaling vectors, preserving internal attention geometry while adapting to downstream tasks.  
* **Rank and Scaling**: Set rank $r \= 4$ or $r \= 8$ with $\\alpha \= r$ (scaling factor $= 1.0$), combined with high adapter dropout ($0.1\\text{--}0.2$).  
* **Learning Rate & Warmup**: Use a conservative base learning rate of $1\\times 10^{-5}$ to $3\\times 10^{-5}$ with a 10% linear warmup and weight decay of $0.01$.

## **Accessible Audio and Music Foundation Models (2024–2026)**

* **MusicFM** ([HuggingFace: minzwon/musicfm](https://github.com/minzwon/musicfm?utm_source=gemini) / [arXiv:2505.16306](https://arxiv.org/abs/2505.16306?utm_source=gemini)): Pre-trained using BEST-RQ on large-scale music audio. Its intermediate layer representations outperform MERT on acoustic and temporal feature extraction.  
* **LAION-CLAP / MS-CLAP** ([HuggingFace: laion/clap-htsat-fused](https://huggingface.co/laion/clap-htsat-fused?utm_source=gemini)): Joint audio-text contrastive encoder. Extracting frozen embeddings via clap.get\_audio\_embedding() often yields superior metadata classification (era, origin) compared to audio-only SSL models because contrastive training directly aligns audio features with textual metadata descriptions.  
* **Music Flamingo** ([ResearchGate: Music Flamingo](https://www.researchgate.net/publication/397595508_Music_Flamingo_Scaling_Music_Understanding_in_Audio_Language_Models?utm_source=gemini)): An enhanced Audio Flamingo 3 backbone fine-tuned on music theory and metadata datasets (MF-Skills / MF-Think), offering improved zero-shot text-prompt classification.  
* **MARBLE Benchmark Embeddings** ([arXiv:2306.10548](https://arxiv.org/abs/2306.10548?utm_source=gemini)): Standardized extraction tools for evaluating pre-trained music models across 14 MIR tasks.

## **Theoretical and Practical Performance Ceilings**

| Task | Metric | Realistic Ceiling (Ntrain​≈800–1000) | Key Limiting Factors |
| :---- | :---- | :---- | :---- |
| **Task 1: Decade** | Top-1 Accuracy | **58% – 63%** | Intra-era genre diversity exceeds inter-era production differences; artist-disjoint splits prevent memorization of mix engineering idiosyncrasies. |
| **Task 1: Decade** | Adjacent Accuracy ($\\pm 1$ decade) | **88% – 92%** | Gradual, continuous evolution of studio recording technology across adjacent decades. |
| **Task 2: Market** | Top-1 Accuracy | **66% – 70%** | Production standardization in the 1980s across Western markets (shared synthesizers, mixing consoles, drum machines). |

### **Task 2 Structural Bottlenecks**

> 1. **Production Standardization**: 1980s commercial pop and rock across the US, UK, Germany, Italy, and Spain relied on identical hardware (Yamaha DX7, SSL 4000 series consoles, LinnDrum, Lexicon 224 reverbs). Acoustic signatures across these regions are largely identical.  
> 2. **Separability Limits**: Brazilian 1980s releases separate cleanly due to unique rhythmic and harmonic structures (Samba-pop, MPB). US, UK, and German tracks exhibit high cross-error rates because regional differences in that era were largely linguistic or stylistic rather than acoustic.  
> 3. **Conclusion**: At **65.7% Top-1**, Task 2 is already operating near the practical acoustic ceiling for an 800-clip artist-disjoint dataset.