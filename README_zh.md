# CCC-GeneCompss

​	这是一个基于单细胞生命基础大模型[GeneCompass](https://github.com/xCompass-AI/GeneCompass)的细胞通信分析工具，目标是替代传统细胞通信分析工具[CellChat](https://github.com/jinworks/CellChat?tab=readme-ov-file)、[CellPhoneDB](https://github.com/ventolab/CellphoneDB)等，以深度学习技术和大模型方法进行细胞间互作及细胞通信的分析和研究。



## 1. 数据预处理

### 1.1 H5AD to RDS

​	在基于CellChat进行细胞互作关系分析时，对于.h5ad和.csv格式的单细胞转录组Counts矩阵数据，为提高细胞互作分析成功率，可以将其转换为R语言支持的.rds格式，我们给出基于Python的.h5ad格式数据转换为.csv的脚本`./CellChatAnalysis/h5ad_to_csv.py`及基于R的.csv数据转换为.rds数据的脚本`./CellChatAnalysis/csv_to_rds.R`。

### 1.2 单细胞转录组数据质控与标准化

参考[GeneCompass](https://github.com/xCompass-AI/GeneCompass)数据预处理方法，我们给出了`./preprocess/filter.py`和`./preprocess/normalized.py`来实现通过基因总数和异常值统计过滤双细胞、死细胞、剔除破碎细胞、离群细胞以及剔除线粒体基因、血红蛋白基因，保留蛋白质编码基因的单细胞转录组数据质控功能，并对数据进行标准化和Tokens编码，将单细胞转录组数据编码为[GeneCompass](https://github.com/xCompass-AI/GeneCompass)可加载的Tokens形式。



## 2. 构建细胞互作关系金标准

整合传统细胞通信分析工具[CellChat](https://github.com/jinworks/CellChat?tab=readme-ov-file)、[CellPhoneDB](https://github.com/ventolab/CellphoneDB)以二者共识分数的形式构建细胞互作金标准，将CellChat的互作强度矩阵与CellPhoneDB的结果通过发送和接收进行合并，对CellChat互作强度、CellPhoneDB平均互作强度、CellPhoneDB最大互作强度的分数进行MinMax归一化（0-1范围），计算三个归一化分数的平均值作为共识分数：
$$
共识分数 = (𝑁𝑜𝑟𝑚_𝐶𝑒𝑙𝑙𝐶ℎ𝑎𝑡 + 𝑁𝑜𝑟𝑚_𝐶𝑃𝐷𝐵_𝑀𝑒𝑎𝑛 + 𝑁𝑜𝑟𝑚_𝐶𝑃𝐷𝐵_𝑀𝑎𝑥) / 3
$$

### 2.1 基于CellChat的细胞互作分析

#### 系统要求

**R版本:** ==4.3.3

**操作系统:** Windows/Linux

**内存:** ≥ 16GB（推荐40GB+用于大型单细胞数据集）

#### 环境配置

**安装CRAN包**

```R
install.packages(c("Seurat", "ggplot2", "patchwork", "dplyr", 
                   "future", "RColorBrewer", "stringr"))
```

**安装Bioconductor包**

```R
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("ComplexHeatmap")
```

**安装CellChat**

```R
install.packages("devtools")
devtools::install_github("sqjin/CellChat")
```

基于**1.1 H5AD to RDS**处理后的rds格式单细胞转录组数据，替换./CellChatAnalysis/CellChatAnalysis.R脚本下输出路径及数据集路径，逐步运行即可完成基于CellChat的细胞互作分析得到细胞互作分析矩阵及可视化结果。

### 2.2 基于CellPhoneDB的细胞互作分析

#### 系统要求

**Python版本:** ==3.12.0

**操作系统:** Windows/Linux

**内存:** ≥ 16GB（推荐40GB+用于大型单细胞数据集）

#### 环境配置

```bash
cd ./CellPhoneAnalysis
conda create -n cpdb python==3.12.0
conda activate cpdb
pip install -r requirements.txt 
```

#### 微环境分析准备

```bash
python prepare_microenvs_h5ad.py
```

#### 差异表达基因准备

```bash
python prepare_DEGs_h5ad.py
```

#### 运行CellPhoneDB进行细胞互作分析

```bash
python CellPhoneAnalysis.py
```

### 2.3 金标准构建

生成金标准作为标签，以用于微调生命基础大模型[GeneCompass](https://github.com/xCompass-AI/GeneCompass)

```bash
cd path/to/CCC-GeneCompss
python building_gold_standard_database.py
```



## 3. 生成Embeddings

​	生命基础大模型[GeneCompass](https://github.com/xCompass-AI/GeneCompass)通过将单细胞转录组数据转换为高维向量表示的Embeddings进行细胞通信分析，将**1.2 单细胞转录组数据质控与标准化**过程中经过标准化和Tokens编码的单细胞转录组数据转换成Embeddings。

```bash
cd path/to/CCC-GeneCompss
python generate_embeddings.py
```

生成Emeddings需要占用大量的计算资源，并且耗费较长时间，为方便验证，我们提供了已生成的Embeddings示例：https://pan.baidu.com/s/1X97G7PdJRHXYn5vako9RnQ?pwd=1uyh ，提取码: 1uyh 


## 4. 细胞互作分析

#### 环境配置	

```bash
cd path/to/CCC-GeneCompss
conda create -n ccc python==3.12.0
conda activate ccc
pip install -r requirements.txt 
```

如果遇到`transformers==4.30.0`安装报错，可以执行以下步骤：

```bash
conda install -c conda-forge tokenizers=0.13.3
pip install transformers==4.30.0
```

conda install 

结合CellChat和CellPhoneDB共识金标准以及生成的Embeddings对生命基础大模型[GeneCompass](https://github.com/xCompass-AI/GeneCompass)进行细胞互作分析下游任务微调，得到细胞互作矩阵及可视化结果。

```bash
cd path/to/CCC-GeneCompss
python cell_cell_interaction.py
```

为方便验证，我们提供了已处理好的normalized数据：[tabula_sapiens_liver](https://pan.baidu.com/s/1RsTlTB4aTlwlk5cHtIQtuA?pwd=b8d8#list/path=%2F)，提取码: b8d8

我们还提供了一份已生成的金标准标签数据[complete_labeled_interactions.csv](https://pan.baidu.com/s/1tcELkJexk3LwN6frNNykbA?pwd=jmc6)，提取码: jmc6 

[GeneCompass](https://github.com/xCompass-AI/GeneCompass)预训练模型可以通过下面的链接获取：

将pretrained_model目录置于主路径下（`./pretrained_models/GeneCompass_Small`，`./pretrained_models/GeneCompass_Base`）

| Model             | Description                         | Download                                           |
| ----------------- | ----------------------------------- | -------------------------------------------------- |
| GeneCompass_Small | Pretrained on 6-layer GeneCompass.  | [Link](https://www.scidb.cn/en/anonymous/SUZOdk1y) |
| GeneCompass_Base  | Pretrained on 12-layer GeneCompass. | [Link](https://www.scidb.cn/en/anonymous/SUZOdk1y) |
