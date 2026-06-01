# ==============================================================================
# CSV转换为Seurat RDS文件（修复版）
# ==============================================================================
# 功能: 将稀疏矩阵和元数据转换为Seurat RDS对象
# 输入: 稀疏矩阵(.mtx), barcodes, genes, metadata
# 输出: Seurat RDS文件
# 作者: GeneCompass团队
# ==============================================================================

library(Seurat)
library(Matrix)
library(dplyr)
library(ggplot2)
library(patchwork)

# ==============================================================================
# 命令行参数解析
# ==============================================================================
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
  cat("用法: Rscript csv_to_rds.R <output_dir> [options]\n")
  cat("参数:\n")
  cat("  output_dir     : 输入文件所在目录（也是输出目录）\n")
  cat("  min_cells      : 每个基因的最小细胞数 (默认: 3)\n")
  cat("  min_features   : 每个细胞的最小基因数 (默认: 200)\n")
  quit(status = 1)
}

output_dir <- args[1]
min_cells <- ifelse(length(args) >= 2, as.integer(args[2]), 3)
min_features <- ifelse(length(args) >= 3, as.integer(args[3]), 200)

cat("===== 参数配置 =====\n")
cat("输出目录:", output_dir, "\n")
cat("最小细胞数/基因:", min_cells, "\n")
cat("最小基因数/细胞:", min_features, "\n")
cat("====================\n\n")

# 检查输入文件
input_files <- list(
  sparse_matrix = file.path(output_dir, "sparse_matrix.mtx"),
  barcodes = file.path(output_dir, "barcodes.tsv"),
  genes = file.path(output_dir, "genes.tsv"),
  metadata = file.path(output_dir, "metadata.csv")
)

missing_files <- sapply(input_files, function(f) !file.exists(f))
if (any(missing_files)) {
  cat("错误: 缺少以下文件:\n")
  print(names(missing_files)[missing_files])
  quit(status = 1)
}

# ==============================================================================
# 1. 读取稀疏矩阵
# ==============================================================================
cat("读取稀疏矩阵...\n")
sparse_matrix <- Matrix::readMM(input_files$sparse_matrix)

# ==============================================================================
# 2. 读取barcodes
# ==============================================================================
cat("读取barcodes...\n")
barcodes <- readLines(input_files$barcodes)
colnames(sparse_matrix) <- barcodes

# ==============================================================================
# 3. 读取基因信息（改进版）
# ==============================================================================
cat("读取基因信息...\n")
# 读取为单列向量
gene_symbols <- readLines(input_files$genes)

# 检查基因名格式
first_gene <- gene_symbols[1]
is_ensembl <- grepl("^ENSG", first_gene)

if (is_ensembl) {
  cat("检测到Ensembl基因标识符\n")
} else {
  cat("检测到Gene Symbol格式\n")
}

# 过滤空基因符号
non_empty_idx <- which(gene_symbols != "" & !is.na(gene_symbols))
if (length(non_empty_idx) < length(gene_symbols)) {
  cat("过滤空基因符号:", length(non_empty_idx), "/", length(gene_symbols), "\n")
  gene_symbols <- gene_symbols[non_empty_idx]
  sparse_matrix <- sparse_matrix[non_empty_idx, ]
}

# 处理重复基因名
dup_count <- sum(duplicated(gene_symbols))
if (dup_count > 0) {
  cat("警告: 发现", dup_count, "个重复基因名\n")
  cat("使用make.unique()使基因名唯一\n")
}

# 使用gene symbol作为行名并确保唯一
rownames(sparse_matrix) <- make.unique(gene_symbols)

# ==============================================================================
# 4. 行名验证
# ==============================================================================
cat("\n===== 数据验证 =====\n")
cat("保留基因总数:", nrow(sparse_matrix), "\n")
cat("细胞总数:", ncol(sparse_matrix), "\n")
cat("空行名数量:", sum(rownames(sparse_matrix) == ""), "\n")
cat("重复行名数量:", sum(duplicated(rownames(sparse_matrix))), "\n")
cat("前5个基因名:\n")
print(head(rownames(sparse_matrix), 5))

# ==============================================================================
# 5. 读取元数据
# ==============================================================================
cat("\n读取元数据...\n")
metadata <- read.csv(input_files$metadata, row.names = 1)

# 检查细胞数量是否一致
if (nrow(metadata) != ncol(sparse_matrix)) {
  stop("错误: 元数据行数(", nrow(metadata), ")与矩阵列数(", ncol(sparse_matrix), ")不匹配")
}

# 检查细胞ID是否一致
if (!all(rownames(metadata) == colnames(sparse_matrix))) {
  cat("警告: 细胞ID不一致，尝试对齐...\n")
  common_cells <- intersect(rownames(metadata), colnames(sparse_matrix))
  cat("保留共同细胞:", length(common_cells), "\n")
  if (length(common_cells) < 100) {
    stop("错误: 共同细胞数量太少（", length(common_cells), "）")
  }
  metadata <- metadata[common_cells, ]
  sparse_matrix <- sparse_matrix[, common_cells]
}

# ==============================================================================
# 6. 创建Seurat对象
# ==============================================================================
cat("\n创建Seurat对象...\n")
seurat_obj <- CreateSeuratObject(
    counts = sparse_matrix,
    meta.data = metadata,
    project = "CellChat_Analysis",
    min.cells = min_cells,
    min.features = min_features
)

cat("Seurat对象创建成功\n")
print(seurat_obj)

# ==============================================================================
# 7. 添加质量控制指标（改进线粒体基因检测）
# ==============================================================================
cat("\n===== 质量控制 =====\n")

# 改进：根据基因名格式选择线粒体基因模式
if (is_ensembl) {
  # Ensembl格式的线粒体基因通常以MT-开头
  mt_pattern <- "^MT-"
} else {
  # Gene Symbol格式的线粒体基因模式
  mt_pattern <- "(^MT-|^MTRNR)"
}

# 查找线粒体基因
gene_names <- rownames(seurat_obj)
mt_idx <- grepl(mt_pattern, gene_names, ignore.case = TRUE)

if (sum(mt_idx) > 0) {
  cat("检测到", sum(mt_idx), "个线粒体基因\n")
  seurat_obj[["percent.mt"]] <- PercentageFeatureSet(
    seurat_obj,
    features = rownames(seurat_obj)[mt_idx]
  )
} else {
  warning("未检测到线粒体基因，设置percent.mt为0")
  seurat_obj[["percent.mt"]] <- 0
}

# 可视化QC指标
cat("生成QC可视化图表...\n")
tryCatch({
  vln_plot <- VlnPlot(seurat_obj, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"),
                      pt.size = 0.1) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    ggtitle("质量控制指标分布")

  # 保存QC图
  qc_plot_path <- file.path(output_dir, "QC_plot.png")
  ggsave(qc_plot_path, plot = vln_plot, width = 10, height = 8, dpi = 300)
  cat("QC图已保存:", qc_plot_path, "\n")
}, error = function(e) {
  cat("警告: QC图生成失败:", e$message, "\n")
})

# ==============================================================================
# 8. 标准化数据（为CellChat准备）
# ==============================================================================
cat("\n标准化数据...\n")
seurat_obj <- NormalizeData(seurat_obj, normalization.method = "LogNormalize", scale.factor = 10000)
cat("数据标准化完成\n")

# ==============================================================================
# 9. 保存Seurat对象
# ==============================================================================
cat("\n保存Seurat对象...\n")
output_rds <- file.path(output_dir, "seurat_obj.rds")
saveRDS(seurat_obj, file = output_rds)
cat("✅ Seurat对象已保存:", output_rds, "\n")

# ==============================================================================
# 10. 最终验证
# ==============================================================================
cat("\n===== 最终验证 =====\n")
cat("Seurat对象摘要:\n")
print(seurat_obj)

cat("\n基因信息统计:\n")
cat("保留的基因总数:", nrow(seurat_obj), "\n")

cat("\n细胞类型统计（如果有）:\n")
if ("cell_type" %in% colnames(seurat_obj@meta.data)) {
  celltype_counts <- table(seurat_obj@meta.data$cell_type)
  print(celltype_counts)
} else {
  cat("  无cell_type列\n")
  cat("  可用的元数据列:", colnames(seurat_obj@meta.data), "\n")
}

cat("\n质量控制指标汇总:\n")
qc_summary <- seurat_obj@meta.data %>%
  summarise(
    median_nFeature = median(nFeature_RNA, na.rm = TRUE),
    median_nCount = median(nCount_RNA, na.rm = TRUE),
    median_mt = median(percent.mt, na.rm = TRUE),
    cells_with_high_mt = sum(percent.mt > 20, na.rm = TRUE)
  )
print(qc_summary)

# ==============================================================================
# 11. 保存处理摘要
# ==============================================================================
summary_file <- file.path(output_dir, "seurat_conversion_summary.txt")
sink(summary_file)
cat("===== Seurat对象创建摘要 =====\n")
cat("输出目录:", output_dir, "\n")
cat("RDS文件:", output_rds, "\n\n")

cat("数据统计:\n")
cat("  细胞数:", ncol(seurat_obj), "\n")
cat("  基因数:", nrow(seurat_obj), "\n")
cat("  基因标识符类型:", ifelse(is_ensembl, "Ensembl", "Gene Symbol"), "\n\n")

cat("质量控制:\n")
cat("  中位数基因数/细胞:", qc_summary$median_nFeature, "\n")
cat("  中位数计数/细胞:", qc_summary$median_nCount, "\n")
cat("  中位数线粒体比例:", qc_summary$median_mt, "%\n")
cat("  高线粒体细胞数:", qc_summary$cells_with_high_mt, "\n\n")

cat("会话信息:\n")
sessionInfo()
sink()

cat("✅ 处理完成！\n")
cat("  - Seurat对象:", output_rds, "\n")
cat("  - QC图表:", qc_plot_path, "\n")
cat("  - 摘要文件:", summary_file, "\n")
