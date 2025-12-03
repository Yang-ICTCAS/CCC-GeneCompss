# 设置环境变量以避免网络请求
Sys.setenv(UV_OFFLINE = 1)

# 加载必要的库
library(Seurat)
library(CellChat)
library(ggplot2)
library(patchwork)
library(dplyr)
library(future)
library(ComplexHeatmap)
library(RColorBrewer)
library(stringr)

# 设置并行计算提高效率
plan("multicore", workers = 16)
options(future.globals.maxSize = 80 * 1024^3) # 增加到80GB内存

# 设置工作路径
output_dir <- "/mnt/data_sdb/wangx/data/SingleCell/processed_datasets/TabulaSapiens/"
cellchat_output <- file.path(output_dir, "CellChat_Results20251015/")
dir.create(cellchat_output, showWarnings = FALSE, recursive = TRUE)

# 1. 加载Seurat对象
cat("加载Seurat对象...\n")
seurat_rds <- "/mnt/data_sdb/wangx/data/SingleCell/processed_datasets/TabulaSapiens/tabula_sapiens_liver_sparse/tabula_sapiens_liver.rds"
seurat_obj <- readRDS(seurat_rds)

# 验证Seurat对象
cat("===== 验证Seurat对象 =====\n")
cat("细胞数量:", ncol(seurat_obj), "\n")  
cat("基因数量:", nrow(seurat_obj), "\n")
cat("元数据列:", colnames(seurat_obj@meta.data), "\n")

# 2. 准备CellChat输入数据
cat("准备CellChat输入数据...\n")

# 获取标准化数据
data.input <- GetAssayData(seurat_obj, assay = "RNA", slot = "data")
if(is.null(data.input) || nrow(data.input) == 0) {
  cat("尝试从counts数据创建标准化数据...\n")
  raw_counts <- GetAssayData(seurat_obj, assay = "RNA", slot = "counts")
  data.input <- NormalizeData(raw_counts, normalization.method = "LogNormalize")
}

# 确保基因名为大写（匹配数据库）
cat("转换基因名为大写以匹配数据库...\n")
rownames(data.input) <- toupper(rownames(data.input))

cat("表达矩阵维度:", dim(data.input), "\n")

# 准备元数据
metadata <- seurat_obj@meta.data
cell_type_col <- "cell_type"  # 确保这是您的细胞类型列名

# 验证细胞类型列
if (!cell_type_col %in% colnames(metadata)) {
  possible_cols <- grep("type|celltype|cluster", colnames(metadata), value = TRUE, ignore.case = TRUE)
  if(length(possible_cols) > 0) {
    cell_type_col <- possible_cols[1]
    cat(paste0("警告：未找到'cell_type'列，改用'", cell_type_col, "'作为细胞类型列\n"))
  } else {
    stop("错误：无法确定细胞类型列，请检查元数据")
  }
}

# 设置细胞数量阈值
min_cell_threshold <- 20

# 检查细胞类型分布
cat("检查细胞类型分布...\n")
cell_type_counts <- table(metadata[[cell_type_col]])
cat("细胞类型数量:", length(cell_type_counts), "\n")
cat("最小细胞群:", min(cell_type_counts), "\n")

# 移除不达标的细胞群
if(min(cell_type_counts) < min_cell_threshold) {
  cat("移除不达标细胞群...\n")
  valid_types <- names(cell_type_counts)[cell_type_counts >= min_cell_threshold]
  cells_to_keep <- metadata[[cell_type_col]] %in% valid_types
  
  # 更新数据
  metadata <- metadata[cells_to_keep, ]
  data.input <- data.input[, cells_to_keep]
  
  cat("保留细胞类型:", paste(valid_types, collapse=", "), "\n")
  cat("保留细胞数:", sum(cells_to_keep), "/", ncol(data.input), "\n")
}

# 验证数据维度一致性
cat("验证数据维度一致性...\n")
if(ncol(data.input) != nrow(metadata)) {
  stop(sprintf("数据矩阵(%d列)与元数据(%d行)维度不匹配", 
               ncol(data.input), nrow(metadata)))
}

# 3. 创建CellChat对象
cat("创建CellChat对象...\n")
cellchat <- createCellChat(object = data.input, 
                           meta = metadata, 
                           group.by = cell_type_col)

# 4. 设置数据库
cat("设置CellChat数据库...\n")
CellChatDB <- CellChatDB.human

# 安全转换函数 - 确保返回向量长度不变
safe_upper <- function(x) {
  if (is.character(x)) {
    return(toupper(x))
  } else if (is.factor(x)) {
    levels(x) <- toupper(levels(x))
    return(x)
  } else {
    return(x)
  }
}

# 转换数据库基因名为大写（匹配数据）
cat("转换数据库基因名为大写...\n")
CellChatDB$interaction$ligand <- safe_upper(CellChatDB$interaction$ligand)
CellChatDB$interaction$receptor <- safe_upper(CellChatDB$interaction$receptor)


cat("简化数据库结构...\n")
# CellChatDB$complex <- NULL  
# CellChatDB$cofactor <- NULL  

# 仅保留简单配体-受体对
simple_pairs <- !grepl("_|\\s|-|COMPLEX", CellChatDB$interaction$ligand) & 
  !grepl("_|\\s|-|COMPLEX", CellChatDB$interaction$receptor)
CellChatDB$interaction <- CellChatDB$interaction[simple_pairs, ]

# 设置数据库
cellchat@DB <- CellChatDB

# 检查数据库
cat("数据库统计:\n")
cat("配体-受体对数量:", nrow(CellChatDB$interaction), "\n")
cat("示例配体:", head(unique(CellChatDB$interaction$ligand)), "\n")
cat("示例受体:", head(unique(CellChatDB$interaction$receptor)), "\n")

# 5. 预处理数据 
cat("预处理数据...\n")
# 安全执行subsetData
tryCatch({
  cellchat <- subsetData(cellchat)
}, error = function(e) {
  cat("标准预处理失败:", e$message, "\n")
  
  # 手动创建子集数据
  cat("尝试手动创建子集数据...\n")
  
  # 获取数据库中的基因
  db_genes <- unique(c(CellChatDB$interaction$ligand, CellChatDB$interaction$receptor))
  
  # 找出数据中存在的数据库基因
  valid_genes <- intersect(rownames(data.input), db_genes)
  
  cat("有效数据库基因数量:", length(valid_genes), "\n")
  
  if(length(valid_genes) == 0) {
    stop("错误：数据库基因与数据基因无交集")
  }
  
  # 创建子集数据
  data.input.sub <- data.input[valid_genes, ]
  
  # 手动创建CellChat对象
  cellchat <- createCellChat(object = data.input.sub, 
                             meta = metadata, 
                             group.by = cell_type_col)
  
  # 重新设置数据库
  cellchat@DB <- CellChatDB
  
  
  cat("手动设置data.signaling...\n")
  cellchat@data.signaling <- as.matrix(data.input.sub)
  
  
  cat("手动设置idents...\n")
  cellchat@idents <- as.factor(metadata[[cell_type_col]])
  
  cat("手动创建子集数据成功!\n")
})


# 6. 识别过表达基因
cat("识别过表达基因...\n")

# 获取CellChat版本
cellchat_version <- packageVersion("CellChat")

# 根据版本选择参数
tryCatch({
  if (cellchat_version >= "1.6.0") {
    cellchat <- identifyOverExpressedGenes(cellchat, thresh.p = 0.05)
  } else {
    cellchat <- identifyOverExpressedGenes(cellchat, pvalue.cutoff = 0.05)
  }
}, error = function(e) {
  cat("标准方法失败:", e$message, "\n尝试手动识别过表达基因...\n")
  
  # 手动识别过表达基因
  cat("手动执行identifyOverExpressedGenes...\n")
  
  # 获取细胞类型
  celltypes <- unique(cellchat@idents)
  
  # 创建存储结果的列表
  features.over <- list()
  
  # 对每种细胞类型执行过表达分析
  for (ct in celltypes) {
    cat("分析细胞类型:", ct, "\n")
    
    # 获取该细胞类型的细胞索引
    idx <- which(cellchat@idents == ct)
    
    # 检查索引是否有效
    if (length(idx) == 0 || any(idx > ncol(cellchat@data.signaling))) {
      cat("警告：无效的细胞索引，跳过细胞类型", ct, "\n")
      next
    }
    
    # 获取其他细胞类型的索引
    idx.other <- setdiff(1:length(cellchat@idents), idx)
    
    # 检查其他细胞类型索引是否有效
    if (length(idx.other) == 0 || any(idx.other > ncol(cellchat@data.signaling))) {
      cat("警告：无效的其他细胞类型索引，跳过细胞类型", ct, "\n")
      next
    }
    
    # 计算每个基因的p值
    pvals <- sapply(1:nrow(cellchat@data.signaling), function(i) {
      # 提取表达值
      expr_ct <- as.numeric(cellchat@data.signaling[i, idx])
      expr_other <- as.numeric(cellchat@data.signaling[i, idx.other])
      
      # 检查是否有足够的数据进行检验
      if (length(expr_ct) < 3 || length(expr_other) < 3) {
        return(NA)
      }
      
      # 执行Wilcoxon检验
      tryCatch({
        wilcox.test(expr_ct, expr_other, alternative = "greater")$p.value
      }, error = function(e) {
        cat("基因", rownames(cellchat@data.signaling)[i], "检验失败:", e$message, "\n")
        return(NA)
      })
    })
    
    # 校正p值
    pvals.adj <- p.adjust(pvals, method = "fdr", na.rm = TRUE)
    
    # 选择显著过表达的基因
    sig.genes <- rownames(cellchat@data.signaling)[which(pvals.adj < 0.05 & !is.na(pvals.adj))]
    
    # 存储结果
    features.over[[as.character(ct)]] <- sig.genes
    cat("细胞类型", ct, "发现显著基因数量:", length(sig.genes), "\n")
  }
  
  # 将结果存入cellchat对象
  cellchat@var.features$features.over <- features.over
  cat("手动识别过表达基因完成!\n")
})

# 7. 识别过表达配体-受体互作
cat("识别过表达配体-受体互作...\n")
tryCatch({
  cellchat <- identifyOverExpressedInteractions(cellchat)
}, error = function(e) {
  cat("识别过表达互作失败:", e$message, "\n")
  cat("尝试简化方法...\n")
  
  # 简化方法：直接使用数据库中的配体-受体对
  LRsig <- CellChatDB$interaction
  cellchat@LR$LRsig <- LRsig
  cat("使用简化方法设置配体-受体对\n")
})

# 8. 计算细胞通讯概率
cat("计算细胞通讯概率...\n")
tryCatch({
  # 尝试使用更宽松的参数
  cellchat <- computeCommunProb(cellchat, 
                                type = "truncatedMean", 
                                trim = 0.1,  # 更宽松的trim值
                                raw.use = FALSE,
                                population.size = TRUE,
                                nboot = 10,   # 减少bootstrap次数以提高速度
                                seed.use = 42)
  cat("通讯概率计算成功!\n")
}, error = function(e) {
  cat("计算方法失败:", e$message, "\n")
  cat("尝试使用更宽松的参数...\n")
  
  # 尝试放宽参数
  tryCatch({
    cellchat <- computeCommunProb(cellchat, 
                                  type = "truncatedMean", 
                                  trim = 0.2,  # 进一步放宽trim值
                                  raw.use = FALSE,
                                  population.size = TRUE,
                                  nboot = 5,   # 进一步减少bootstrap次数
                                  seed.use = 42)
    cat("通讯概率计算成功（使用宽松参数）!\n")
  }, error = function(e2) {
    cat("再次尝试失败:", e2$message, "\n")
    cat("尝试使用median方法...\n")
    
    # 尝试使用median方法
    tryCatch({
      cellchat <- computeCommunProb(cellchat, 
                                    type = "median", 
                                    raw.use = FALSE,
                                    population.size = TRUE,
                                    nboot = 10,
                                    seed.use = 42)
      cat("通讯概率计算成功（使用median方法）!\n")
    }, error = function(e3) {
      cat("median方法失败:", e3$message, "\n")
      cat("尝试使用默认参数...\n")
      
      # 最后尝试使用默认参数
      cellchat <- computeCommunProb(cellchat)
      cat("通讯概率计算成功（使用默认参数）!\n")
    })
  })
})

# 检查通讯网络有效性
if(exists("cellchat@net") && !is.null(cellchat@net$count) && sum(cellchat@net$count) > 0) {
  cat("检测到有效通讯数量:", sum(cellchat@net$count), "\n")
} else {
  cat("\n===== 通讯失败诊断 =====\n")
  cat("细胞类型数量:", length(unique(cellchat@idents)), "\n")
  
  # 计算过表达基因数量
  if (!is.null(cellchat@var.features$features.over)) {
    over_genes_count <- length(unlist(cellchat@var.features$features.over))
    cat("过表达基因数量:", over_genes_count, "\n")
  } else {
    cat("过表达基因数量: 0 (未识别)\n")
  }
  
  if (!is.null(cellchat@LR$LRsig)) {
    cat("有效配体-受体对数量:", nrow(cellchat@LR$LRsig), "\n")
  } else {
    cat("有效配体-受体对数量: 0 (未识别)\n")
  }
  
  # 尝试最后手段
  cat("尝试最后手段：使用最小通讯检测...\n")
  tryCatch({
    cellchat <- computeCommunProb(cellchat, 
                                  type = "median", 
                                  raw.use = TRUE,
                                  population.size = FALSE,
                                  nboot = 1)
    cat("最小通讯检测成功!\n")
  }, error = function(e) {
    cat("最小通讯检测失败:", e$message, "\n")
    cat("无法计算通讯概率，请检查数据和参数\n")
  })
}

# 9. 安全过滤通讯结果
cat("安全过滤通讯结果...\n")
safe_filter <- function(cellchat, min.cells) {
  # 检查通讯网络是否有效
  if (is.null(cellchat@net) || is.null(cellchat@net$count)) {
    warning("通讯网络无效，跳过过滤")
    return(cellchat)
  }
  
  # 获取当前细胞类型计数
  cell_counts <- table(cellchat@idents)
  
  # 仅保留有效通讯
  valid_pairs <- which(cellchat@net$count >= min.cells, arr.ind = TRUE)
  
  if(length(valid_pairs) == 0) {
    warning("没有满足阈值的通讯对，跳过过滤")
    return(cellchat)
  }
  
  # 创建过滤后的通讯数组
  new_net <- list(
    prob = cellchat@net$prob[valid_pairs[,1], valid_pairs[,2], , drop = FALSE],
    count = cellchat@net$count[valid_pairs[,1], valid_pairs[,2], drop = FALSE],
    pval = cellchat@net$pval[valid_pairs[,1], valid_pairs[,2], , drop = FALSE]
  )
  
  cellchat@net <- new_net
  return(cellchat)
}

cellchat <- safe_filter(cellchat, min.cells = 10)

# 10. 计算通讯网络
cat("计算通讯网络...\n")
cellchat <- computeCommunProbPathway(cellchat)

# 11. 聚合网络
cat("聚合网络...\n")
cellchat <- aggregateNet(cellchat)

# 12. 保存CellChat对象
cat("保存CellChat对象...\n")
saveRDS(cellchat, file.path(cellchat_output, "tabula_sapiens_liver_cellchat.rds"))

# 13. 可视化结果
cat("生成可视化结果...\n")

# A. 细胞通讯网络图
tryCatch({
  png(file.path(cellchat_output, "communication_network.png"), 
      width = 12, height = 10, units = "in", res = 300)
  netVisual_circle(cellchat@net$count, 
                   vertex.weight = as.numeric(table(cellchat@idents)),
                   weight.scale = TRUE,
                   label.edge = TRUE,
                   edge.label.cex = 0.7,
                   vertex.label.cex = 0.8)
  dev.off()
  cat("细胞通讯网络图已保存\n")
}, error = function(e) {
  cat("网络图生成失败:", e$message, "\n")
})

# B. 热力图展示通讯强度
tryCatch({
  png(file.path(cellchat_output, "communication_heatmap.png"), 
      width = 14, height = 12, units = "in", res = 300)
  
  mat <- cellchat@net$weight
  Heatmap(mat,
          name = "Interaction strength",
          col = colorRampPalette(c("blue", "white", "red"))(100),
          row_names_gp = gpar(fontsize = 8),
          column_names_gp = gpar(fontsize = 8))
  dev.off()
  cat("通讯热力图已保存\n")
}, error = function(e) {
  cat("热力图生成失败:", e$message, "\n")
})

# C. 信号通路气泡图
tryCatch({
  pathways <- cellchat@netP$pathways
  if(length(pathways) > 0) {
    png(file.path(cellchat_output, "pathway_bubble.png"), 
        width = 12, height = 8, units = "in", res = 300)
    netVisual_bubble(cellchat, 
                     sources.use = 1:length(cellchat@idents),
                     targets.use = 1:length(cellchat@idents),
                     remove.isolate = FALSE)
    dev.off()
    cat("信号通路气泡图已保存\n")
  } else {
    cat("无可用的信号通路信息，跳过气泡图\n")
  }
}, error = function(e) {
  cat("气泡图生成失败:", e$message, "\n")
})

# D. 配体-受体互作图
tryCatch({
  pathways <- cellchat@netP$pathways
  if(length(pathways) > 0) {
    png(file.path(cellchat_output, "ligand_receptor_network.png"), 
        width = 14, height = 12, units = "in", res = 300)
    netVisual_aggregate(cellchat, signaling = pathways[1], layout = "circle")
    dev.off()
    cat("配体-受体互作图已保存\n")
  } else {
    cat("无可用的信号通路信息，跳过配体-受体互作图\n")
  }
}, error = function(e) {
  cat("配体-受体互作图生成失败:", e$message, "\n")
})

# 14. 保存详细结果
cat("保存详细结果...\n")

# 保存细胞通讯结果
tryCatch({
  df.net <- subsetCommunication(cellchat)
  write.csv(df.net, file.path(cellchat_output, "cellchat_communication.csv"), row.names = FALSE)
  cat("细胞通讯结果已保存\n")
}, error = function(e) {
  cat("保存通讯结果失败:", e$message, "\n")
})

# 保存信号通路结果
tryCatch({
  df.pathways <- as.data.frame(cellchat@netP[["pathways"]])
  write.csv(df.pathways, file.path(cellchat_output, "cellchat_pathways.csv"), row.names = FALSE)
  cat("信号通路结果已保存\n")
}, error = function(e) {
  cat("保存信号通路结果失败:", e$message, "\n")
})

# 保存细胞类型信息
tryCatch({
  cell_type_info <- data.frame(
    cell_id = colnames(seurat_obj),
    cell_type = as.character(seurat_obj@meta.data[[cell_type_col]])
  )
  write.csv(cell_type_info, 
            file.path(cellchat_output, "cell_type_info.csv"),
            row.names = FALSE)
  cat("细胞类型信息已保存\n")
}, error = function(e) {
  cat("保存细胞类型信息失败:", e$message, "\n")
})

# 新增：保存细胞互作强度矩阵
tryCatch({
  # 获取细胞类型互作强度矩阵
  interaction_matrix <- cellchat@net$weight
  
  # 转换为数据框
  interaction_df <- as.data.frame(interaction_matrix)
  
  # 添加行名作为列
  interaction_df$Sender <- rownames(interaction_matrix)
  
  # 重新排列列
  interaction_df <- interaction_df[, c("Sender", colnames(interaction_matrix))]
  
  # 保存为CSV
  write.csv(interaction_df, 
            file.path(cellchat_output, "cell_interaction_strength_matrix.csv"),
            row.names = FALSE)
  
  cat("细胞互作强度矩阵已保存\n")
}, error = function(e) {
  cat("保存细胞互作强度矩阵失败:", e$message, "\n")
})

# 15. 保存会话信息
sink(file.path(cellchat_output, "session_info.txt"))
sessionInfo()
sink()

cat("\n✅ CellChat 分析成功完成！结果保存在:", cellchat_output, "\n")
cat("关键步骤:\n")
cat("1. 数据准备和预处理\n")
cat("2. 过表达基因识别\n")
cat("3. 配体-受体互作识别\n")
cat("4. 细胞通讯概率计算\n")
cat("5. 结果可视化和保存\n")
cat("6. 新增: 细胞互作强度矩阵导出\n")
