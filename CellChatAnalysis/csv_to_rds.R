library(Seurat)
library(Matrix)
library(dplyr)
library(ggplot2)
library(patchwork)

# 设置文件路径
output_dir <- "/mnt/data_sdb/wangx/data/SingleCell/processed_datasets/TabulaSapiens/tabula_sapiens_liver_sparse/"
output_rds <- file.path(output_dir, "tabula_sapiens_liver.rds")

# 读取稀疏矩阵
cat("读取稀疏矩阵...\n")
sparse_matrix <- Matrix::readMM(file.path(output_dir, "sparse_matrix.mtx"))

# 读取barcodes
cat("读取barcodes...\n")
barcodes <- readLines(file.path(output_dir, "barcodes.tsv")) 
colnames(sparse_matrix) <- barcodes

# 关键修改：读取单列基因信息 (gene symbol only)
cat("读取基因信息...\n")
# 直接读取为单列向量
gene_symbols <- readLines(file.path(output_dir, "genes.tsv"))

# ========== 简化处理流程 ==========
# 1. 过滤空基因符号
cat("过滤空基因符号...\n")
non_empty_idx <- which(gene_symbols != "" & !is.na(gene_symbols))
gene_symbols <- gene_symbols[non_empty_idx]
sparse_matrix <- sparse_matrix[non_empty_idx, ]

# 2. 使用gene symbol作为行名并确保唯一
cat("设置基因名为gene symbol并确保唯一...\n")
rownames(sparse_matrix) <- make.unique(gene_symbols)

# ========== 行名验证 ==========
cat("\n行名验证结果:\n")
cat("保留基因总数:", nrow(sparse_matrix), "\n")
cat("空行名数量:", sum(rownames(sparse_matrix) == ""), "\n")
cat("重复行名数量:", sum(duplicated(rownames(sparse_matrix))), "\n")
cat("示例行名:\n")
print(head(rownames(sparse_matrix), 10))

# 读取元数据
cat("读取元数据...\n")
metadata <- read.csv(file.path(output_dir, "metadata.csv"), row.names = 1)

# 创建Seurat对象
cat("创建Seurat对象...\n")
seurat_obj <- CreateSeuratObject(
    counts = sparse_matrix,
    meta.data = metadata,
    project = "TabulaSapiens_Liver",
    min.cells = 3,
    min.features = 200
)

# ========== 简化基因信息存储 ==========
cat("保存基因信息到Seurat对象...\n")
# 创建简化版基因数据框
gene_metadata <- data.frame(
    gene_symbol = rownames(seurat_obj),
    unique_rowname = rownames(seurat_obj),
    stringsAsFactors = FALSE
)
rownames(gene_metadata) <- rownames(seurat_obj)

# 添加到Seurat对象
seurat_obj[["RNA"]]@meta.data <- gene_metadata

# 添加质量控制指标
cat("\n=============== 质量控制 ===============\n")
mt_pattern <- "(^MT-|^MTRNR)"  # 匹配多种线粒体基因前缀

# 从基因信息中查找线粒体基因
gene_metadata <- seurat_obj[["RNA"]]@meta.data
mt_idx <- grepl(mt_pattern, gene_metadata$gene_symbol, ignore.case = TRUE)

if (sum(mt_idx) > 0) {
  cat("检测到", sum(mt_idx), "个线粒体基因\n")
  seurat_obj[["percent.mt"]] <- PercentageFeatureSet(
    seurat_obj, 
    features = rownames(gene_metadata)[mt_idx]
  )
} else {
  warning("未检测到线粒体基因")
  seurat_obj[["percent.mt"]] <- 0
}

# 可视化QC指标
cat("生成QC可视化图表...\n")
VlnPlot(seurat_obj, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), 
        pt.size = 0.1) + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  ggtitle("质量控制指标分布")

# 保存Seurat对象
cat("\n保存Seurat对象...\n")
saveRDS(seurat_obj, file = output_rds)
cat("\n✅ Tabula Sapiens Seurat对象已保存: ", output_rds, "\n")

# 验证基因信息
cat("\n=============== 最终验证 ===============\n")
cat("Seurat对象摘要:\n")
print(seurat_obj)

cat("\n基因信息统计:\n")
gene_metadata <- seurat_obj[["RNA"]]@meta.data
cat("保留的基因总数:", nrow(gene_metadata), "\n")
cat("空基因符号数量:", sum(gene_metadata$gene_symbol == ""), "\n")

cat("\n质量控制指标汇总:\n")
qc_summary <- seurat_obj@meta.data %>% 
  summarise(
    median_nFeature = median(nFeature_RNA),
    median_nCount = median(nCount_RNA),
    median_mt = median(percent.mt),
    cells_with_high_mt = sum(percent.mt > 20)
  )
print(qc_summary)

cat("\n处理完成!") 