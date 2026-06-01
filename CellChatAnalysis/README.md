# CellChat 细胞互作分析工具

## 概述

本目录包含 CellChat 细胞互作分析所需的工具链，从 h5ad 文件转换到最终的 CellChat 分析结果。

## 文件说明

| 文件 | 功能 |
|------|------|
| `h5ad_to_csv.py` | h5ad → 稀疏矩阵 (.mtx) + barcodes + genes + metadata |
| `csv_to_rds.R` | 稀疏矩阵 → Seurat RDS 对象 |
| `cellchat_analysis.R` | Seurat RDS → CellChat 完整分析 (interaction matrix + communication) |
| `cellchat_gold_standard.R` | 简化版 CellChat 金标准生成 (直接输出 interaction matrix) |

## 使用方法

### 步骤1: h5ad 转稀疏矩阵

```bash
python h5ad_to_csv.py \
  --input <path/to/input.h5ad> \
  --output <path/to/output_dir> \
  --celltype_col cell_type
```

**输出文件**: `sparse_matrix.mtx`, `barcodes.tsv`, `genes.tsv`, `metadata.csv`

### 步骤2: 创建 Seurat 对象

```bash
Rscript csv_to_rds.R <output_dir> [min_cells] [min_features]
```

### 步骤3: CellChat 分析

```bash
Rscript cellchat_analysis.R \
  <input_rds> <output_dir> \
  [cell_type_col] [min_cells] [workers] [nboot]
```

**输出文件**:
- `cellchat_result.rds` - CellChat 对象
- `cell_interaction_strength_matrix.csv` - **细胞互作强度矩阵**（金标准构建的核心输入）
- `cellchat_communication.csv` - 细胞通讯结果
- `cellchat_pathways.csv` - 信号通路结果
- `cell_type_info.csv` - 细胞类型信息

### 金标准生成

> **注意**：金标准由 CellChat + CellPhoneDB 交叉验证共同构建。
> CellChat 分析仅提供金标准的 CellChat 部分，最终 Consensus Score 需通过
> `building_gold_standard_database.py` 融合 CellPhoneDB 结果后计算。

```bash
Rscript cellchat_gold_standard.R \
  <output_dir> <output_dir> <cell_type_col> <workers>
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `min_cells` | 3 | 每个基因最少细胞数 |
| `min_features` | 200 | 每个细胞最少基因数 |
| `workers` | 4 | 并行 worker 数 |
| `nboot` | 10 | bootstrap 次数 |
