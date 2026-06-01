# ==============================================================================
# h5ad转换为稀疏矩阵和CSV文件（修复版）
# ==============================================================================
# 功能: 将h5ad文件转换为CellChat分析所需的格式
# 输入: h5ad文件
# 输出: 稀疏矩阵(.mtx), barcodes, genes, metadata
# 作者: GeneCompass团队
# ==============================================================================

import numpy as np
import scipy.io as sio
import pandas as pd
import anndata
import os
import argparse

# ==============================================================================
# 命令行参数解析
# ==============================================================================
parser = argparse.ArgumentParser(description='将h5ad文件转换为CellChat分析所需格式')
parser.add_argument('--input', type=str, required=True, help='输入h5ad文件路径')
parser.add_argument('--output', type=str, required=True, help='输出目录路径')
parser.add_argument('--celltype_col', type=str, default='cell_type',
                   help='细胞类型列名 (默认: cell_type)')

args = parser.parse_args()

input_h5ad = args.input
output_dir = args.output
celltype_col = args.celltype_col

print(f"===== 参数配置 =====")
print(f"输入文件: {input_h5ad}")
print(f"输出目录: {output_dir}")
print(f"细胞类型列: {celltype_col}")
print(f"====================")

# 检查输入文件
if not os.path.exists(input_h5ad):
    raise FileNotFoundError(f"输入文件不存在: {input_h5ad}")

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# ==============================================================================
# 1. 读取h5ad文件
# ==============================================================================
print("\n读取h5ad文件...")
adata = anndata.read_h5ad(input_h5ad)

print(f"数据形状: {adata.shape}")
print(f"细胞数: {adata.n_obs}, 基因数: {adata.n_vars}")

# 检查细胞类型列是否存在
if celltype_col not in adata.obs.columns:
    print(f"警告: 列 '{celltype_col}' 不在obs中")
    print(f"可用的列: {list(adata.obs.columns)}")
    # 尝试自动查找
    possible_cols = [col for col in adata.obs.columns
                    if 'type' in col.lower() or 'cluster' in col.lower()]
    if possible_cols:
        celltype_col = possible_cols[0]
        print(f"自动选择列: {celltype_col}")
    else:
        raise ValueError(f"无法确定细胞类型列")

# ==============================================================================
# 2. 基因名处理（改进版）
# ==============================================================================
print(f"\n处理基因名...")

# 检测基因标识符类型
first_gene_name = str(adata.var_names[0])
print(f"第一个基因名: {first_gene_name}")

is_ensembl = first_gene_name.startswith('ENSG')
if is_ensembl:
    print("检测到Ensembl基因标识符")
    # CellChat需要gene symbol，尝试从feature_name提取
    if 'feature_name' in adata.var.columns:
        print("从feature_name列提取gene symbol")
        import re
        raw_symbols = [str(name).strip() for name in adata.var['feature_name']]
        # 去除_ENSG后缀 (如: TSPAN6_ENSG00000000003 -> TSPAN6)
        gene_symbols = [re.sub(r'_ENSG\d+$', '', s) for s in raw_symbols]
        # 过滤空值和nan
        keep_indices = [i for i, name in enumerate(gene_symbols) if name and name.lower() != 'nan']
        gene_symbols = [gene_symbols[i] for i in keep_indices]
        adata = adata[:, keep_indices]
        print(f"  保留基因数: {len(keep_indices)}/{len(adata.var)}")
    else:
        print("警告: 无feature_name列，使用Ensembl ID（可能与CellChatDB不匹配）")
        gene_symbols = adata.var_names.tolist()
else:
    print("检测到Gene Symbol或混合格式")
    # 尝试从feature_name列获取
    if 'feature_name' in adata.var.columns:
        print("从feature_name列提取基因名")
        gene_symbols = []
        keep_indices = []

        for i, name in enumerate(adata.var['feature_name']):
            name = str(name).strip()

            # 改进：保留完整基因名，不要只取下划线前部分
            # 只过滤空值
            if name:
                gene_symbols.append(name)
                keep_indices.append(i)
            else:
                print(f"  警告: 跳过空基因名（索引 {i}）")

        print(f"  保留基因数: {len(keep_indices)}/{len(adata.var)}")
        adata = adata[:, keep_indices]
    else:
        print("使用原始基因名")
        gene_symbols = adata.var_names.tolist()
        keep_indices = list(range(len(gene_symbols)))

print(f"最终基因数: {len(gene_symbols)}")
print(f"前10个基因名: {gene_symbols[:10]}")

# ==============================================================================
# 3. 保存稀疏矩阵（基因×细胞）
# ==============================================================================
print(f"\n保存稀疏矩阵...")
output_mtx = os.path.join(output_dir, "sparse_matrix.mtx")

# 确保矩阵格式
if hasattr(adata.X, 'toarray'):
    matrix_data = adata.X.T  # 转置为基因×细胞
else:
    matrix_data = adata.X.T

sio.mmwrite(output_mtx, matrix_data, field='integer', symmetry='general')
print(f"  稀疏矩阵: {output_mtx}")

# ==============================================================================
# 4. 保存barcodes
# ==============================================================================
print(f"\n保存barcodes...")
output_barcodes = os.path.join(output_dir, "barcodes.tsv")
barcodes_df = pd.DataFrame(adata.obs.index)
barcodes_df.to_csv(output_barcodes, sep='\t', index=False, header=False)
print(f"  条形码: {output_barcodes}")

# ==============================================================================
# 5. 保存基因信息
# ==============================================================================
print(f"\n保存基因信息...")
output_genes = os.path.join(output_dir, "genes.tsv")

# 改进：保存为单列，不包含表头
genes_df = pd.DataFrame(gene_symbols, columns=['gene_symbol'])
genes_df.to_csv(output_genes, sep='\t', index=False, header=False)
print(f"  基因信息: {output_genes}")

# ==============================================================================
# 6. 保存元数据
# ==============================================================================
print(f"\n保存元数据...")
output_metadata = os.path.join(output_dir, "metadata.csv")
metadata = adata.obs.copy()
metadata.to_csv(output_metadata)
print(f"  元数据: {output_metadata}")

# 统计细胞类型分布
if celltype_col in metadata.columns:
    print(f"\n细胞类型分布:")
    celltype_counts = metadata[celltype_col].value_counts()
    for ct, count in celltype_counts.items():
        print(f"  {ct}: {count}")

# ==============================================================================
# 7. 保存数据摘要
# ==============================================================================
print(f"\n保存数据摘要...")
summary_file = os.path.join(output_dir, "conversion_summary.txt")
with open(summary_file, 'w') as f:
    f.write("===== h5ad转换摘要 =====\n")
    f.write(f"输入文件: {input_h5ad}\n")
    f.write(f"输出目录: {output_dir}\n")
    f.write(f"细胞类型列: {celltype_col}\n\n")
    f.write(f"数据统计:\n")
    f.write(f"  细胞数: {adata.n_obs}\n")
    f.write(f"  基因数: {len(gene_symbols)}\n")
    f.write(f"  基因标识符类型: {'Ensembl' if is_ensembl else 'Gene Symbol'}\n\n")

    if celltype_col in metadata.columns:
        f.write(f"细胞类型统计:\n")
        for ct, count in metadata[celltype_col].value_counts().items():
            f.write(f"  {ct}: {count}\n")

print(f"  数据摘要: {summary_file}")

print(f"\n✅ 转换完成！")
print(f"输出文件:")
print(f"  - 稀疏矩阵: {output_mtx}")
print(f"  - 条形码: {output_barcodes}")
print(f"  - 基因信息: {output_genes}")
print(f"  - 元数据: {output_metadata}")
print(f"  - 数据摘要: {summary_file}")
