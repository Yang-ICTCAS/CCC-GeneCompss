# CCC-GeneCompass: Cell-Cell Communication via GeneCompass

**Large Language Model for Cell-Cell Interaction Prediction in Single-Cell Transcriptomics**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[中文文档](README_zh.md)

---

## Overview

CCC-GeneCompass leverages the **GeneCompass** — a BERT-based large language model pretrained on 100M+ single-cell transcriptomes — to predict cell-cell interaction (CCI) strength. The pipeline:

1. Computes **cell-type-aggregated expression profiles** from single-cell data
2. Constructs a **gold standard** from CellChat + CellPhoneDB v5 consensus
3. Fine-tunes GeneCompass to rank cell-type interaction strength
4. Evaluates via **5-fold cross-validation** with **Spearman rank correlation ρ**

## Key Features

- **Deterministic**: Cell-type mean expression eliminates random single-cell sampling noise
- **Scientifically rigorous**: Spearman ρ + bootstrap 95% CI + permutation p-value
- **Multi-GPU**: DataParallel training on up to 4 GPUs
- **End-to-end pipeline**: Preprocessing → Gold Standard → Training → CV → Visualization
- **Modular design**: Separate scripts for each step, easy to customize

## Installation

```bash
pip install -r requirements.txt

# R dependencies for CellChat
R -e 'install.packages(c("devtools","NMF","circlize","ComplexHeatmap"))'
R -e 'devtools::install_github("jinworks/CellChat")'

# CellPhoneDB v5 database
# Download cellphonedb.zip from: https://github.com/ventolab/CellphoneDB
```

## Data Preparation

### 1. Single-Cell Data (h5ad)
Requirements: `.obs` must contain a `cell_type` column; gene symbols accessible via `feature_name` or `var_names`.

### 2. Pretrained Model
Download GeneCompass_Base checkpoint and place:
```
pretrained_models/
├── pytorch_model.bin   # ~1.1GB
└── config.json
```

### 3. Knowledge Files (for tokenization)
```
prior_knowledge/
├── human_mouse_tokens.pickle
└── public/
    └── human_gene_median_after_filter.pickle
```

### 4. CellPhoneDB v5 Database
```
CellPhoneAnalysis/v5.0.0/
├── cellphonedb.zip
├── gene_input.csv
└── protein_input.csv
```

## Pipeline

```bash
# ================================================================
# Full Pipeline for One Organ
# ================================================================
# Input:  Raw single-cell h5ad (must have cell_type column)
# Output: results/ (5-fold CV metrics + models + visualizations)
# ================================================================

ORGAN=pancreas
RAW_H5AD=/path/to/original/${ORGAN}.h5ad       # original data

# ====== Step 0: Data Preprocessing ======
#  Raw h5ad → filtered.h5ad + tokenized arrow + cell-type-aggregated
python preprocess_data.py \
    --h5ad ${RAW_H5AD} \
    --output data/${ORGAN} \
    --tokens prior_knowledge/human_mouse_tokens.pickle \
    --medians prior_knowledge/public/human_gene_median_after_filter.pickle
#  Outputs: data/${ORGAN}/filtered.h5ad              (filtered data)
#           data/${ORGAN}/single_cell_dataset/       (tokenized arrow)
#           data/${ORGAN}/cell_type_aggregated/      (used by Step 4)

# ====== Step 1: CellChat Analysis ======
#  Filtered h5ad → interaction matrix + communication probabilities
python CellChatAnalysis/h5ad_to_csv.py \
    --input data/${ORGAN}/filtered.h5ad \
    --output data/${ORGAN}/cellchat/
Rscript CellChatAnalysis/csv_to_rds.R data/${ORGAN}/cellchat/ 3 200
Rscript CellChatAnalysis/cellchat_gold_standard.R data/${ORGAN}/cellchat/ 4
#  Outputs: data/${ORGAN}/cellchat/cell_interaction_strength_matrix.csv
#           data/${ORGAN}/cellchat/cellchat_communication.csv

# ====== Step 2: CellPhoneDB v5 Analysis ======
#  Raw h5ad → statistically significant interaction matrix
CPDB_ZIP=/path/to/cellphonedb.zip
CPDB_DATA=/path/to/CellPhoneAnalysis/v5.0.0/
python run_cpdb.py \
    --h5ad ${RAW_H5AD} \
    --cpdb_db ${CPDB_ZIP} \
    --cpdb_genes ${CPDB_DATA} \
    --output data/${ORGAN}/cellphonedb/
#  Output: data/${ORGAN}/cellphonedb/significant_means.txt

# ====== Step 3: Joint Gold Standard ======
#  Weights automatically learned from source significance (w ∝ mean sig LR pairs)
python genecompass_gold_standard.py \
    --cellchat data/${ORGAN}/cellchat \
    --cpdb data/${ORGAN}/cellphonedb \
    --output data/${ORGAN}/gold_standard
#  Output: data/${ORGAN}/gold_standard/complete_labeled_interactions.csv

# ====== Step 4: 5-Fold Cross-Validation ======
#  Train GeneCompass with gold standard labels, evaluate Spearman ρ
python pipeline_cv.py \
    --proj_root . \
    --gs_path data/${ORGAN}/gold_standard/complete_labeled_interactions.csv \
    --dataset data/${ORGAN}/cell_type_aggregated \
    --output results/${ORGAN}_cv \
    --organ ${ORGAN^} --epochs 30 --batch 1 --grad_accum 4
#  Outputs: results/${ORGAN}_cv/cv_summary.json + fold{1-5}/ + visualizations

# ====== Step 5: Standalone Inference (optional) ======
python pipeline_inference.py \
    --model results/${ORGAN}_cv/fold1/best_model \
    --test_set results/${ORGAN}_cv/fold1/data_splits/test \
    --token_dict prior_knowledge/human_mouse_tokens.pickle
```

## Evaluation Metrics

### Primary: Spearman Rank Correlation ρ
Measures how well the model ranks cell-type pairs by interaction strength.

| ρ range | Interpretation |
|---------|---------------|
| ρ ≥ 0.7 | EXCELLENT |
| 0.5 ≤ ρ < 0.7 | GOOD |
| 0.3 ≤ ρ < 0.5 | MODERATE |
| ρ < 0.3 | WEAK |

Each fold reports: ρ with bootstrap 95% CI + permutation p-value.
Final: mean ± std across 5 folds.

### Secondary
- **Pearson r**: Linear correlation with gold standard
- **R²**: Explained variance
- **RMSE**: Root mean squared error

## Output Structure

```
data/{organ}/
├── filtered.h5ad                       # filtered data (Step 0)
├── single_cell_dataset/                # tokenized arrow data (Step 0)
├── cell_type_aggregated/               # cell-type-aggregated data (Step 0)
├── cellchat/                           # CellChat outputs (Step 1)
│   ├── cell_interaction_strength_matrix.csv
│   └── cellchat_communication.csv
├── cellphonedb/                        # CellPhoneDB outputs (Step 2)
│   ├── significant_means.txt
│   └── statistical_analysis_*.txt
└── gold_standard/                      # joint gold standard (Step 3)
    ├── complete_labeled_interactions.csv
    └── gold_standard_stats.json

results/{organ}_cv/                     # 5-fold CV outputs (Step 4)
├── cv_summary.json                     # mean ± std across 5 folds
├── fold{1-5}/
│   ├── metrics.json                    # per-fold ρ + CI + Pearson + R²
│   ├── best_model/                     # trained model (pytorch_model.bin)
│   ├── test_true.npy                   # true labels
│   └── test_pred.npy                   # predictions
├── interaction_heatmap.png             # 300dpi visualizations
├── interaction_network.png
├── interaction_circular.png
├── interaction_bubble.png
├── interaction_flow.png
├── autocrine_scores.png
└── true_vs_predicted.png
```

## Citation

```bibtex
@software{ccc_genecompass,
  title = {CCC-GeneCompass: Cell-Cell Communication via Large Language Model},
  year = {2025},
  note = {Based on GeneCompass: A Large-Scale Pretrained Model for Single-Cell Gene Expression}
}
```

## License

MIT
