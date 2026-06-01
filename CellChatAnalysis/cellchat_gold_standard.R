#!/usr/bin/env Rscript
library(CellChat); library(Seurat); library(dplyr); library(future)
args=commandArgs(trailingOnly=TRUE)
DD=args[1]; workers=ifelse(length(args)>=2,as.integer(args[2]),4)

cat("Loading Seurat...\n")
seurat=readRDS(file.path(DD,"seurat_obj.rds"))
Idents(seurat)=seurat@meta.data$cell_type
cc=table(Idents(seurat)); valid=names(cc[cc>=10])
seurat=subset(seurat,idents=valid)
cat("Cell types:",length(valid),"\n")

plan("multicore",workers=workers); options(future.globals.maxSize=8000*1024^2)
cellchat=createCellChat(object=seurat,group.by="ident",assay="RNA")
cellchat@DB=subsetDB(CellChatDB.human,search=c("Secreted Signaling","Cell-Cell Contact","ECM-Receptor"))

cat("Preprocessing...\n")
cellchat=subsetData(cellchat)
cellchat=identifyOverExpressedGenes(cellchat)
cellchat=identifyOverExpressedInteractions(cellchat)

cat("Computing communication...\n")
cellchat=computeCommunProb(cellchat,type="triMean",nboot=10)
cellchat=filterCommunication(cellchat,min.cells=10)
cellchat=computeCommunProbPathway(cellchat)
cellchat=aggregateNet(cellchat)

saveRDS(cellchat,file.path(DD,"cellchat_result.rds"))

m=as.data.frame(cellchat@net$weight); m$Sender=rownames(m)
write.csv(m,file.path(DD,"cell_interaction_strength_matrix.csv"),row.names=FALSE)
df=subsetCommunication(cellchat)
if(nrow(df)>0) write.csv(df,file.path(DD,"cellchat_communication.csv"),row.names=FALSE)
cat("DONE!\n")
