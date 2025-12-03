import numpy as np
import scipy.io as sio
import pandas as pd
import anndata
import os
import re

# 输入和输出路径
input_h5ad = "/mnt/data_sdb/wangx/data/SingleCell/ori_data/TabulaSapiens/tabula_sapiens_liver/tabula_sapiens_liver.h5ad"
output_dir = "/mnt/data_sdb/wangx/data/SingleCell/processed_datasets/TabulaSapiens/tabula_sapiens_liver_sparse"

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 读取h5ad文件
adata = anndata.read_h5ad(input_h5ad)

# 只提取gene symbol部分
print("原始feature_name列类型:", type(adata.var['feature_name']))
print("前5个feature_name值:", adata.var['feature_name'].head().tolist())

# 提取gene symbols并过滤空值
gene_symbols = []
keep_indices = []  # 保留的基因索引

for i, name in enumerate(adata.var['feature_name']):
    # 简单处理：取第一个下划线前的部分作为gene symbol
    if '_' in name:
        symbol = name.split('_')[0].strip()
    else:
        symbol = name.strip()

    # 过滤空符号
    if symbol:
        gene_symbols.append(symbol)
        keep_indices.append(i)

# 过滤基因数据
adata = adata[:, keep_indices]

print(f"\n原始基因总数: {len(adata.var)}, 保留基因数: {len(keep_indices)}")
print("前10个处理后的gene symbols:", gene_symbols[:10])

# 保存稀疏矩阵（基因×细胞）
sio.mmwrite(
    os.path.join(output_dir, "sparse_matrix.mtx"),
    adata.X.T,  # 转置为基因×细胞
    field='integer',
    symmetry='general'
)

# 保存barcodes
barcodes = pd.DataFrame(adata.obs.index)
barcodes.to_csv(
    os.path.join(output_dir, "barcodes.tsv"),
    sep="\t",
    index=False,
    header=False
)

# 保存基因信息 - 只保存一列gene symbol
genes_df = pd.DataFrame(gene_symbols, columns=['gene_symbol'])
genes_df.to_csv(
    os.path.join(output_dir, "genes.tsv"),
    sep="\t",
    index=False,
    header=False
)

# 保存元数据
metadata = adata.obs
metadata.to_csv(os.path.join(output_dir, "metadata.csv"))

print(f"转换完成！文件保存在: {output_dir}")
print(f"- 稀疏矩阵: sparse_matrix.mtx")
print(f"- 细胞条形码: barcodes.tsv")
print(f"- 基因信息(gene symbol): genes.tsv")
print(f"- 元数据: metadata.csv")
print(f"总共保留了 {len(genes_df)} 个有效gene symbol的基因")