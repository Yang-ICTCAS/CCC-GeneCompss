# CCC-GeneCompass：基于GeneCompass大模型的细胞互作分析

**利用大规模预训练语言模型进行单细胞转录组细胞间通讯预测**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[English](README.md)

---

## 概述

CCC-GeneCompass 基于 **GeneCompass**——一个在超过1亿个单细胞转录组上预训练的BERT架构大模型——预测细胞间相互作用（CCI）强度。分析流程包括：

1. 从单细胞数据计算**细胞类型聚合表达谱**
2. 基于 CellChat + CellPhoneDB v5 共识构建**金标准**
3. 微调 GeneCompass 对细胞类型相互作用强度进行排序
4. 通过**5折交叉验证**和**Spearman秩相关系数ρ**进行评估

## 核心特性

- **确定性**：细胞类型均值表达，消除单细胞随机采样噪声
- **科学严谨**：Spearman ρ + Bootstrap 95%置信区间 + 置换检验p值
- **多GPU支持**：DataParallel训练，最多支持4块GPU
- **端到端流程**：数据预处理 → 金标准构建 → 训练 → 交叉验证 → 可视化
- **模块化设计**：各步骤独立脚本，便于定制

## 安装

```bash
pip install -r requirements.txt

# CellChat 的 R 依赖
R -e 'install.packages(c("devtools","NMF","circlize","ComplexHeatmap"))'
R -e 'devtools::install_github("jinworks/CellChat")'

# CellPhoneDB v5 数据库
# 从 https://github.com/ventolab/CellphoneDB 下载 cellphonedb.zip
```

## 数据准备

### 1. 原始单细胞数据 (h5ad)
要求：`.obs` 中包含 `cell_type` 列，`.var` 中可通过 `feature_name` 或 `var_names` 获取基因符号。

### 2. 预训练模型
下载 GeneCompass_Base 模型文件，放入：
```
pretrained_models/
├── pytorch_model.bin   # ~1.1GB
└── config.json
```

### 3. 知识文件（用于token化）
```
prior_knowledge/
├── human_mouse_tokens.pickle
└── public/
    └── human_gene_median_after_filter.pickle
```

### 4. CellPhoneDB v5 数据库
```
CellPhoneAnalysis/v5.0.0/
├── cellphonedb.zip
├── gene_input.csv
└── protein_input.csv
```

## 分析流程

```bash
# ================================================================
# 单器官完整分析流程
# ================================================================
# 输入：  原始数据 h5ad（需包含 cell_type 列）
# 输出：  results/ 目录（5折交叉验证指标 + 模型 + 可视化）
# ================================================================

ORGAN=pancreas
RAW_H5AD=/path/to/original/${ORGAN}.h5ad       # 原始单细胞表达数据

# ====== Step 0: 数据预处理 ======
#  原始数据 → 过滤后数据 + token编码Arrow数据 + 细胞类型聚合数据
python preprocess_data.py \
    --h5ad ${RAW_H5AD} \
    --output data/${ORGAN} \
    --tokens prior_knowledge/human_mouse_tokens.pickle \
    --medians prior_knowledge/public/human_gene_median_after_filter.pickle
#  生成: data/${ORGAN}/filtered.h5ad              过滤后数据
#        data/${ORGAN}/single_cell_dataset/        token编码Arrow数据
#        data/${ORGAN}/cell_type_aggregated/       细胞类型聚合数据（Step 4用）

# ====== Step 1: CellChat 分析 ======
#  过滤后数据 → 细胞互作矩阵 + 通讯概率
python CellChatAnalysis/h5ad_to_csv.py \
    --input data/${ORGAN}/filtered.h5ad \
    --output data/${ORGAN}/cellchat/
Rscript CellChatAnalysis/csv_to_rds.R data/${ORGAN}/cellchat/ 3 200
Rscript CellChatAnalysis/cellchat_gold_standard.R data/${ORGAN}/cellchat/ 4
#  生成: data/${ORGAN}/cellchat/cell_interaction_strength_matrix.csv
#        data/${ORGAN}/cellchat/cellchat_communication.csv

# ====== Step 2: CellPhoneDB v5 分析 ======
#  原始数据 → 统计显著相互作用矩阵
CPDB_ZIP=/path/to/cellphonedb.zip
CPDB_DATA=/path/to/CellPhoneAnalysis/v5.0.0/
python run_cpdb.py \
    --h5ad ${RAW_H5AD} \
    --cpdb_db ${CPDB_ZIP} \
    --cpdb_genes ${CPDB_DATA} \
    --output data/${ORGAN}/cellphonedb/
#  生成: data/${ORGAN}/cellphonedb/significant_means.txt

# ====== Step 3: 构建联合金标准 ======
#  权重自动从来源显著性学习 (w ∝ 平均显著LR配对数)
python genecompass_gold_standard.py \
    --cellchat data/${ORGAN}/cellchat \
    --cpdb data/${ORGAN}/cellphonedb \
    --output data/${ORGAN}/gold_standard
#  生成: data/${ORGAN}/gold_standard/complete_labeled_interactions.csv

# ====== Step 4: 5折交叉验证 ======
#  用金标准标签训练GeneCompass，评估Spearman秩相关系数
python pipeline_cv.py \
    --proj_root . \
    --gs_path data/${ORGAN}/gold_standard/complete_labeled_interactions.csv \
    --dataset data/${ORGAN}/cell_type_aggregated \
    --output results/${ORGAN}_cv \
    --organ ${ORGAN^} --epochs 30 --batch 1 --grad_accum 4
#  生成: results/${ORGAN}_cv/cv_summary.json + fold{1-5}/ + 可视化

# ====== Step 5: 独立推理（可选） ======
python pipeline_inference.py \
    --model results/${ORGAN}_cv/fold1/best_model \
    --test_set results/${ORGAN}_cv/fold1/data_splits/test \
    --token_dict prior_knowledge/human_mouse_tokens.pickle
```

## 评估指标

### 主要指标：Spearman 秩相关系数 ρ
衡量模型对细胞类型互作强度排序与金标准共识排序的一致性。

| ρ 范围 | 等级 |
|---------|------|
| ρ ≥ 0.7 | 优秀 |
| 0.5 ≤ ρ < 0.7 | 良好 |
| 0.3 ≤ ρ < 0.5 | 中等 |
| ρ < 0.3 | 弱 |

每折报告：ρ + Bootstrap 95%置信区间 + 置换检验p值。
最终结果：5折均值 ± 标准差。

### 辅助指标
- **Pearson r**：与金标准的线性相关性
- **R²**：可解释方差
- **RMSE**：均方根误差

## 输出结构

```
data/{organ}/
├── filtered.h5ad                       # 过滤后数据 (Step 0)
├── single_cell_dataset/                # token编码Arrow数据 (Step 0)
├── cell_type_aggregated/               # 细胞类型聚合数据 (Step 0)
├── cellchat/                           # CellChat输出 (Step 1)
│   ├── cell_interaction_strength_matrix.csv
│   └── cellchat_communication.csv
├── cellphonedb/                        # CellPhoneDB输出 (Step 2)
│   ├── significant_means.txt
│   └── statistical_analysis_*.txt
└── gold_standard/                      # 联合金标准 (Step 3)
    ├── complete_labeled_interactions.csv
    └── gold_standard_stats.json

results/{organ}_cv/                     # 5折交叉验证输出 (Step 4)
├── cv_summary.json                     # 5折汇总：均值±标准差
├── fold{1-5}/
│   ├── metrics.json                    # 每折 ρ + CI + Pearson + R²
│   ├── best_model/                     # 训练模型 (pytorch_model.bin)
│   ├── test_true.npy                   # 真实标签
│   └── test_pred.npy                   # 预测值
├── interaction_heatmap.png             # 300dpi 可视化
├── interaction_network.png
├── interaction_circular.png
├── interaction_bubble.png
├── interaction_flow.png
├── autocrine_scores.png
└── true_vs_predicted.png
```

## 引用

```bibtex
@software{ccc_genecompass,
  title = {CCC-GeneCompass: Cell-Cell Communication via Large Language Model},
  year = {2025},
  note = {Based on GeneCompass: A Large-Scale Pretrained Model for Single-Cell Gene Expression}
}
```

## 许可

MIT
