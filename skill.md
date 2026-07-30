# Skill: Rebuild PC-DLCMNet for Cross-Regional Wu Vowel Recognition

## 1. Project Goal

This project reconstructs the experimental pipeline for **Prompt-Conditioned Dual-Level Class Memory Network (PC-DLCMNet)** on a low-resource cross-regional Wu vowel recognition task.

The target task is:

> Given multiple source regions for training, recognize fine-grained vowel categories in an unseen target region using only a few labeled target-region samples.

The main evaluation protocol is:

* **Target-region K-shot evaluation**
* Default main setting: **K = 4**
* Each target region is treated as an unseen region.
* For each target region, sample `K` labeled support samples per vowel class.
* Use the remaining samples in the target region as query samples.
* Repeat support/query sampling 100 times.
* Report Accuracy and Macro-F1.

The proposed model is **PC-DLCMNet**, which contains:

1. **GEMA**: Global-to-Episode Class Memory Adaptation
2. **PCMR**: Prompt-Conditioned Class Memory Reading

---

## 2. Recommended Project Structure

Create the project as:

```text
pc_dlcmnet_wu/
├── configs/
│   ├── pc_dlcmnet_4shot.yaml
│   ├── ablation.yaml
│   └── baselines.yaml
├── data/
│   ├── raw_audio/
│   ├── textgrid/
│   ├── manifests/
│   │   ├── wu_vowel_segments.csv
│   │   ├── label_map.json
│   │   ├── region_map.json
│   │   └── dataset_stats.json
│   └── features/
│       ├── whisper/
│       ├── hubert/
│       └── wavlm/
├── pc_dlcmnet/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py
│   │   ├── episode_sampler.py
│   │   └── collate.py
│   ├── models/
│   │   ├── acoustic_encoder.py
│   │   ├── gema.py
│   │   ├── pcmr.py
│   │   ├── pc_dlcmnet.py
│   │   └── baselines.py
│   ├── train/
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── metrics.py
│   ├── eval/
│   │   ├── evaluator.py
│   │   ├── ablation_runner.py
│   │   └── table_writer.py
│   └── utils/
│       ├── seed.py
│       ├── io.py
│       └── logging.py
├── tools/
│   ├── build_manifest.py
│   ├── extract_features.py
│   ├── run_all_targets.py
│   ├── run_ablation.py
│   ├── run_support_weighting.py
│   └── make_latex_tables.py
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   ├── results/
│   ├── tables/
│   └── figures/
├── requirements.txt
└── README.md
```

---

## 3. Data Preparation

### 3.1 Input Data

The dataset should be organized by region, speaker, and audio segment. The final manifest should contain one row per vowel segment.

Required fields:

```text
segment_id
audio_path
region
site
speaker_id
vowel_label
ipa_label
start_time
end_time
duration
split_flag
```

A recommended `wu_vowel_segments.csv` format is:

```csv
segment_id,audio_path,region,site,speaker_id,vowel_label,ipa_label,start_time,end_time,duration
qingyang_spk01_0001,data/raw_audio/qingyang/spk01/0001.wav,Qingyang,Qingyang01,spk01,a,a,0.31,0.69,0.38
```

### 3.2 Label Space

Use a shared vowel label space:

```text
Y = {1, 2, ..., C}
```

For example:

```json
{
  "a": 0,
  "i": 1,
  "u": 2,
  "o": 3,
  "e": 4,
  "y": 5,
  "ə": 6,
  "ɛ": 7,
  "ɔ": 8
}
```

The exact labels should follow the cleaned IPA-based vowel labels used in the dataset.

### 3.3 Region Split

For each target region:

```text
target_region = one region
source_regions = all remaining regions
```

Do not use target-region query samples for training, validation, or hyperparameter selection.

For each target region, construct 100 repeated few-shot tasks:

```text
support set: K samples per class from target region
query set: remaining target-region samples
```

Default:

```text
K = 4
num_tasks = 100
```

---

## 4. Feature Extraction

Two implementation options are allowed.

### Option A: Frozen Acoustic Encoder During Training

Use frozen speech encoders such as:

```text
Whisper
HuBERT
WavLM
wav2vec 2.0
mHuBERT
MR-HuBERT
MS-HuBERT
```

For each segment:

```text
H_i = E_ac(x_i)
h_i = Pool(H_i)
```

Where:

* `H_i` is the frame-level acoustic representation.
* `h_i` is the segment-level representation.
* `Pool` is temporal average pooling over valid acoustic frames.

### Option B: Pre-extract Segment Embeddings

For faster experiments, pre-extract segment-level embeddings:

```bash
python tools/extract_features.py \
  --manifest data/manifests/wu_vowel_segments.csv \
  --encoder whisper \
  --output_dir data/features/whisper \
  --pooling mean
```

Each saved feature file should contain:

```text
segment_id
embedding vector
region
label
speaker_id
```

This is recommended for repeated 100-task evaluation.

---

## 5. Method: PC-DLCMNet

PC-DLCMNet consists of three parts:

1. Acoustic representation construction
2. GEMA: global-to-episode class memory adaptation
3. PCMR: prompt-conditioned class memory reading

---

## 6. Episode Formulation

Each episode is:

```text
E = (S, Q)
```

Where:

```text
S = support set
Q = query set
```

During training:

```text
S and Q are sampled from source regions.
```

During evaluation:

```text
S is sampled from the unseen target region.
Q is the remaining target-region samples.
```

Each sample is:

```text
(x_i, y_i)
```

Where:

```text
x_i = vowel segment
y_i = vowel class label
```

---

## 7. GEMA: Global-to-Episode Class Memory Adaptation

### 7.1 Global Class Memory

Maintain learnable global class memory slots:

```text
M_g = [m_1^g, m_2^g, ..., m_C^g] ∈ R^{C × d}
```

Where:

```text
m_c^g = global memory slot for class c
C = number of vowel classes
d = memory dimension
```

The global memory is learned from source-region episodes and is shared across episodes.

Purpose:

```text
Preserve source-region vowel-category knowledge.
```

### 7.2 Support-derived Class Candidate

For each class c in the support set:

```text
S_c = samples in support set whose label is c
```

Compute the support-derived candidate:

```text
u_c = mean embedding of samples in S_c
```

Purpose:

```text
Provide target-region or episode-specific acoustic evidence.
```

### 7.3 Gated Global-to-Episode Adaptation

Instead of directly replacing the global memory with support prototypes, compute a retention gate:

```text
g_c = sigmoid(W_g [m_c^g ; u_c] + b_g)
```

Then construct the episode-level memory:

```text
m_c^e = g_c ⊙ m_c^g + (1 - g_c) ⊙ u_c
```

Where:

```text
m_c^e = episode-level class memory for class c
⊙ = element-wise multiplication
```

Interpretation:

* If `g_c` is large, the model relies more on the global memory.
* If `g_c` is small, the model injects more support-derived evidence.
* GEMA balances stability and adaptation.

The final episode memory is:

```text
M_e = [m_1^e, m_2^e, ..., m_C^e]
```

---

## 8. PCMR: Prompt-Conditioned Class Memory Reading

PCMR performs query-dependent reading over class memories.

### 8.1 Prompt Bank

Define a learnable prompt bank:

```text
P = [p_1, p_2, ..., p_R] ∈ R^{R × d}
```

Where:

```text
R = number of prompts
```

Important:

```text
The prompt is a learnable continuous vector.
It is not a text prompt.
It is not appended to the audio sequence.
It modulates class-memory attention.
```

### 8.2 Prompt Routing

For a query embedding `h_i`, and a memory branch `η`, where:

```text
η ∈ {e, g}
e = episode memory branch
g = global memory branch
```

Compute mean memory state:

```text
m_bar^η = mean of memory slots in M^η
```

Then compute routing weights:

```text
alpha_i^η = softmax(W_r [h_i ; m_bar^η] + b_r)
```

The routed prompt is:

```text
p_i^η = sum_k alpha_{i,k}^η p_k
```

Purpose:

```text
Different queries can activate different prompt vectors.
```

### 8.3 Prompt-conditioned Memory Attention

Project query and memory slots:

```text
q_i = W_Q h_i
k_c^η = W_K m_c^η
v_c^η = W_V m_c^η
```

Transform routed prompt into class-wise attention bias:

```text
a_i^{p,η} = W_p p_i^η + b_p
```

Attention score:

```text
s_{i,c}^η = q_i^T k_c^η / sqrt(d) + a_{i,c}^{p,η}
```

Attention weight:

```text
beta_{i,c}^η = softmax over classes of s_{i,c}^η
```

Memory readout:

```text
r_i^η = sum_c beta_{i,c}^η v_c^η
```

Residual fusion:

```text
z_i^η = LN(h_i + W_O r_i^η)
```

### 8.4 Cosine Classification

For each class c:

```text
logit_{i,c}^η = cos(W_z z_i^η, W_mem m_c^η) / tau
```

Where:

```text
tau = temperature
```

Final prediction during inference uses the episode branch:

```text
prediction = argmax_c logit_{i,c}^e
```

---

## 9. Training Objective

During training, use both episode branch and global branch:

```text
L = CE(logits_e, y) + lambda_g * CE(logits_g, y)
```

Where:

```text
logits_e = logits from episode memory M_e
logits_g = logits from global memory M_g
lambda_g = weight of global auxiliary loss
```

Purpose of global auxiliary loss:

```text
Maintain discriminative structure of global class memories.
Prevent global memories from being optimized only through adapted episode memories.
```

---

## 10. Main Training Procedure

### Step 1: Build Manifest

```bash
python tools/build_manifest.py \
  --audio_root data/raw_audio \
  --textgrid_root data/textgrid \
  --output data/manifests/wu_vowel_segments.csv \
  --label_map data/manifests/label_map.json \
  --region_map data/manifests/region_map.json
```

Expected outputs:

```text
wu_vowel_segments.csv
label_map.json
region_map.json
dataset_stats.json
```

### Step 2: Extract Frozen Features

Example using Whisper:

```bash
python tools/extract_features.py \
  --manifest data/manifests/wu_vowel_segments.csv \
  --encoder whisper \
  --output_dir data/features/whisper \
  --sample_rate 16000 \
  --pooling mean
```

### Step 3: Train PC-DLCMNet on Source-region Episodes

For one target region:

```bash
python train_pc_dlcmnet.py \
  --manifest data/manifests/wu_vowel_segments.csv \
  --feature_dir data/features/whisper \
  --target_region Qingyang \
  --k_shot 4 \
  --num_classes 9 \
  --num_steps 3000 \
  --hidden_dim 256 \
  --memory_dim 128 \
  --prompt_dim 128 \
  --num_prompts 8 \
  --pcmr_layers 3 \
  --num_heads 4 \
  --ffn_dim 768 \
  --lambda_g 0.5 \
  --lr 3e-4 \
  --optimizer adamw \
  --output_dir outputs/checkpoints/qingyang
```

Training rule:

```text
Use only source regions.
Do not use target-region query samples.
Do not tune hyperparameters on target-region query samples.
```

### Step 4: Evaluate on Target-region 4-shot Tasks

```bash
python eval_pc_dlcmnet.py \
  --checkpoint outputs/checkpoints/qingyang/best.pt \
  --manifest data/manifests/wu_vowel_segments.csv \
  --feature_dir data/features/whisper \
  --target_region Qingyang \
  --k_shot 4 \
  --num_tasks 100 \
  --metric accuracy macro_f1 \
  --output outputs/results/qingyang_4shot.json
```

### Step 5: Run All Target Regions

```bash
python tools/run_all_targets.py \
  --config configs/pc_dlcmnet_4shot.yaml \
  --target_regions Qingyang Tongling Jingxian Nanling Ningguo Lishui \
  --num_tasks 100 \
  --output_dir outputs/results/main_4shot
```

---

## 11. Baselines

Implement and compare the following baselines.

### 11.1 Acoustic Feature Classifiers

```text
MFCC-SVM
Acoustic centroid
Acoustic ridge classifier
Acoustic linear SVM
```

Protocol:

* Extract MFCC or acoustic descriptors.
* Standardize using source-region statistics only.
* Train classifier on source-region data.
* Evaluate on target-region query set with K-shot support adaptation if applicable.

### 11.2 Frozen Pre-trained Speech Encoders

```text
wav2vec 2.0
HuBERT
WavLM
Whisper
```

Protocol:

* Freeze encoder.
* Mean-pool frame-level features.
* Train the same downstream classifier.
* Evaluate under the same target-region K-shot protocol.

### 11.3 Recent Speech Foundation Models

```text
mHuBERT-147
MR-HuBERT
MS-HuBERT
Qwen2-Audio
SALMONN
```

Protocol:

* Use them as frozen feature extractors if possible.
* Use the same downstream classifier for fair comparison.

### 11.4 Few-shot and Memory-based Variants

These are also used for ablation.

```text
Support prototype
Global-only memory
Global-support average
GEMA only
PC-DLCMNet
```

Definitions:

```text
Support prototype:
  Use only support-derived prototypes.

Global-only memory:
  Use only learned global class memories M_g.
  No target-region support adaptation.
  No GEMA.
  No PCMR.

Global-support average:
  Directly average global memory and support prototype.
  No gated adaptation.

GEMA only:
  Use GEMA to construct M_e.
  No PCMR.

PC-DLCMNet:
  Use both GEMA and PCMR.
```

---

## 12. Component Ablation

Use the following table logic:

```text
Method                  Support   Global   GEMA   PCMR
Support prototype       yes       no       no     no
Global-only memory      no        yes      no     no
Global-support average  yes       yes      no     no
GEMA only               yes       yes      yes    no
PC-DLCMNet              yes       yes      yes    yes
```

Expected interpretation:

1. Support prototype tests whether a few target-region samples alone are stable.
2. Global-only memory tests whether source-region class memories preserve transferable vowel knowledge.
3. Global-support average tests whether naive fusion works.
4. GEMA only tests whether gated global-to-episode adaptation works.
5. PC-DLCMNet tests whether prompt-conditioned memory reading further improves retrieval.

Run:

```bash
python tools/run_ablation.py \
  --config configs/ablation.yaml \
  --target_regions Qingyang Ningguo Lishui \
  --k_shot 4 \
  --num_tasks 100 \
  --variants support_prototype global_only global_support_average gema_only pc_dlcmnet \
  --output outputs/results/component_ablation.csv
```

---

## 13. Support Weighting Analysis

This analysis should be conducted **within the full PC-DLCMNet framework**, not as GEMA-only.

Compare:

```text
Pseudo-label weighting
Blended weighting
Label-aware weighting
```

Definitions:

```text
Pseudo-label weighting:
  Estimate support reliability from model predictions.

Blended weighting:
  Combine prediction confidence with support labels.

Label-aware weighting:
  Directly use available support labels under supervised few-shot evaluation.
```

Run:

```bash
python tools/run_support_weighting.py \
  --config configs/pc_dlcmnet_4shot.yaml \
  --target_regions Qingyang Ningguo Lishui \
  --k_shot 4 \
  --num_tasks 100 \
  --strategies pseudo_label blended label_aware \
  --output outputs/results/support_weighting.csv
```

---

## 14. K-shot Sensitivity

Evaluate different support sizes:

```text
K = 1, 2, 4, 8, 16
```

Run:

```bash
python tools/run_all_targets.py \
  --config configs/pc_dlcmnet_4shot.yaml \
  --target_regions Qingyang Tongling Jingxian Nanling Ningguo Lishui \
  --k_values 1 2 4 8 16 \
  --num_tasks 100 \
  --output_dir outputs/results/kshot
```

Expected analysis:

```text
PC-DLCMNet should be more robust in low-shot settings.
As K increases, the gap between methods may become smaller because support prototypes become more reliable.
```

---

## 15. Hyperparameter Analysis

Analyze:

```text
prompt number R
global auxiliary loss weight lambda_g
episode length
temperature tau
```

Run:

```bash
python tools/run_hyperparam_sensitivity.py \
  --config configs/pc_dlcmnet_4shot.yaml \
  --target_regions Qingyang Ningguo Lishui \
  --k_shot 4 \
  --sweep prompt_num lambda_g episode_length tau \
  --output outputs/results/hyperparam_sensitivity.csv
```

Recommended values:

```text
hidden_dim = 256
memory_dim = 128
prompt_dim = 128
pcmr_layers = 3
num_heads = 4
ffn_dim = 768
learning_rate = 3e-4
optimizer = AdamW
max_steps = 3000
```

---

## 16. Mechanism Visualization

Generate the following figures.

### 16.1 GEMA Gate Visualization

Visualize:

```text
global retention weight
support injection weight
class-wise gate values
gate values under K = 1, 3, 5
```

Expected interpretation:

```text
GEMA should retain global memory while selectively injecting support evidence.
Support injection should be class-dependent rather than monotonically increasing with K.
```

### 16.2 PCMR Attention Heatmap

Visualize:

```text
query class → class-memory slot attention
```

Expected interpretation:

```text
PCMR should show query-dependent memory reading.
Confusable vowel categories should receive stronger mutual attention.
```

### 16.3 t-SNE Representation Visualization

Compare:

```text
Whisper-only features
Support prototype
PC-DLCMNet
```

Metrics:

```text
silhouette score
class compactness
inter-class separation
```

---

## 17. Metrics

Use:

```text
Accuracy
Macro-F1
```

Accuracy:

```text
correct predictions / total query samples
```

Macro-F1:

```text
average F1 over all vowel classes
```

Report format:

```text
mean ± standard deviation over 100 target-region few-shot tasks
```

For tables:

```text
Accuracy (%)
Macro-F1 (%)
```

In LaTeX captions and table headers, always write:

```latex
Accuracy (\%)
Macro-F1 (\%)
```

Never write:

```latex
Accuracy (%)
Macro-F1 (%)
```

because `%` is a comment symbol in LaTeX.

---

## 18. Result Files

Every experiment should save:

```text
outputs/results/{experiment_name}.csv
outputs/results/{experiment_name}.json
outputs/logs/{experiment_name}.log
outputs/checkpoints/{target_region}/best.pt
```

A result JSON should contain:

```json
{
  "target_region": "Qingyang",
  "k_shot": 4,
  "num_tasks": 100,
  "method": "PC-DLCMNet",
  "accuracy_mean": 50.68,
  "accuracy_std": 2.17,
  "macro_f1_mean": null,
  "macro_f1_std": null,
  "seed": 42
}
```

---

## 19. LaTeX Table Generation

Use a script to generate tables automatically.

```bash
python tools/make_latex_tables.py \
  --input outputs/results/component_ablation.csv \
  --table component_ablation \
  --output outputs/tables/component_ablation.tex
```

Component ablation table should use:

```latex
\begin{table*}[!t]
\centering
\caption{Component ablation study of PC-DLCMNet under the 4-shot target-region setting. Accuracy (\%) is reported on three representative target regions.}
\label{tab:component_ablation}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccc|ccc}
\toprule
Method & Support & Global & GEMA & PCMR & Qingyang & Ningguo & Lishui \\
\midrule
...
\bottomrule
\end{tabular*}
\end{table*}
```

Important LaTeX checks:

```text
Use \\ at the end of table rows.
Use \% instead of %.
Do not add \\ after \toprule, \midrule, \bottomrule.
Use \cmidrule(lr){2-4}, not \cmidrule(lr) alone.
Every \begin{table} must have \end{table}.
Every \begin{tabular} must have \end{tabular}.
Every \caption{...} must be closed with }.
```

---

## 20. Reproducibility Rules

Always fix random seeds:

```python
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

Use deterministic task sampling:

```text
same seed → same support/query tasks
```

Do not allow leakage:

```text
Target-region query samples must not be used in training.
Target-region query samples must not be used for hyperparameter selection.
Feature normalization statistics for traditional baselines should be computed only from source regions.
```

---

## 21. Sanity Checks

Before running full experiments, verify:

```text
1. Every target region has enough samples for K-shot sampling.
2. Every class has at least K samples in the target region.
3. The label map is consistent across all regions.
4. Source regions and target region are disjoint.
5. Frozen encoder outputs have correct dimensions.
6. Support/query sets do not overlap.
7. Global-only memory uses no target-region support adaptation.
8. GEMA-only disables PCMR.
9. PC-DLCMNet enables both GEMA and PCMR.
10. Reported mean/std are computed over the same 100 tasks for all methods.
```

---

## 22. Minimal Execution Order

Run the project in this order:

```bash
# 1. Build manifest
python tools/build_manifest.py \
  --audio_root data/raw_audio \
  --textgrid_root data/textgrid \
  --output data/manifests/wu_vowel_segments.csv

# 2. Extract features
python tools/extract_features.py \
  --manifest data/manifests/wu_vowel_segments.csv \
  --encoder whisper \
  --output_dir data/features/whisper

# 3. Train and evaluate main model
python tools/run_all_targets.py \
  --config configs/pc_dlcmnet_4shot.yaml \
  --target_regions Qingyang Tongling Jingxian Nanling Ningguo Lishui \
  --k_values 4 \
  --num_tasks 100 \
  --output_dir outputs/results/main_4shot

# 4. Run component ablation
python tools/run_ablation.py \
  --config configs/ablation.yaml \
  --target_regions Qingyang Ningguo Lishui \
  --k_shot 4 \
  --num_tasks 100 \
  --output outputs/results/component_ablation.csv

# 5. Run support weighting analysis
python tools/run_support_weighting.py \
  --config configs/pc_dlcmnet_4shot.yaml \
  --target_regions Qingyang Ningguo Lishui \
  --k_shot 4 \
  --num_tasks 100 \
  --output outputs/results/support_weighting.csv

# 6. Run K-shot sensitivity
python tools/run_all_targets.py \
  --config configs/pc_dlcmnet_4shot.yaml \
  --target_regions Qingyang Tongling Jingxian Nanling Ningguo Lishui \
  --k_values 1 2 4 8 16 \
  --num_tasks 100 \
  --output_dir outputs/results/kshot

# 7. Generate LaTeX tables
python tools/make_latex_tables.py \
  --result_dir outputs/results \
  --output_dir outputs/tables
```

---

## 23. Expected Experimental Story

The final paper should support the following claims:

1. **Support-only prototypes are unstable.**
   Few target-region samples cannot reliably represent fine-grained vowel categories.

2. **Global-only memory is useful.**
   Source-region class memories preserve transferable vowel-category knowledge.

3. **Naive global-support averaging is insufficient.**
   Direct fusion can damage global class structure or introduce noisy support evidence.

4. **GEMA improves adaptation over naive fusion.**
   Gated global-to-episode adaptation is more controlled than direct averaging.

5. **PCMR provides the final improvement.**
   Query-dependent memory reading helps retrieve discriminative class-memory cues for each vowel segment.

6. **PC-DLCMNet is strongest under the complete setting.**
   The full model combines stable global memory, support-conditioned adaptation, and query-dependent reading.

---

## 24. Key Implementation Warning

The most important implementation distinction is:

```text
GEMA modifies class memories.
PCMR modifies how each query reads class memories.
```

Do not implement PCMR as a normal classifier head.

Do not implement prompts as text tokens.

Do not update global memories during target-region inference.

Do not use query labels or query samples for adaptation.

The correct inference flow is:

```text
target support set
→ construct support candidates
→ GEMA adapts M_g into M_e
→ each query reads M_e through PCMR
→ cosine logits
→ prediction
```
