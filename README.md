# CCC-GeneCompss
​	This is a cell-cell communication analysis tool based on the single-cell foundation model [GeneCompass](https://github.com/xCompass-AI/GeneCompass). It aims to replace traditional cell communication analysis tools like [CellChat](https://github.com/jinworks/CellChat?tab=readme-ov-file) and [CellPhoneDB](https://github.com/ventolab/CellphoneDB) by utilizing deep learning technology and large model methods for the analysis and research of cell-cell interactions and communication.



## 1. Data Preprocessing

### 1.1 H5AD to RDS

​	When performing cell-cell interaction analysis using [CellChat](https://github.com/jinworks/CellChat?tab=readme-ov-file) , for single-cell Counts matrix data in `.h5ad` or `.csv` format, converting it to the `.rds` format supported by R can improve the success rate of the analysis. We provide a Python script `./CellChatAnalysis/h5ad_to_csv.py`to convert `.h5ad` format data to `.csv`, and an R script `./CellChatAnalysis/csv_to_rds.R` to convert `.csv` data to `.rds` format.

### 1.2 Single-Cell Data Quality Control and Normalization

​	Referencing the data preprocessing method of [GeneCompass](https://github.com/xCompass-AI/GeneCompass), we provide `./preprocess/filter.py` and `./preprocess/normalized.py` to implement quality control for single-cell data. This includes filtering doublets, dead cells, removing broken cells, outlier cells, mitochondrial genes, and hemoglobin genes based on total gene counts and outlier statistics, retaining only protein-coding genes. The data is then normalized and token-encoded, converting the single-cell data into the Tokens form loadable by  [GeneCompass](https://github.com/xCompass-AI/GeneCompass).



## 2. Constructing a Gold Standard for Cell-Cell Interactions

Integrate traditional cell communication analysis tools  [CellChat](https://github.com/jinworks/CellChat?tab=readme-ov-file) and [CellPhoneDB](https://github.com/ventolab/CellphoneDB)  to build a gold standard for cell-cell interactions using a consensus score derived from both. The  [CellChat](https://github.com/jinworks/CellChat?tab=readme-ov-file)  interaction strength matrix and  [CellPhoneDB](https://github.com/ventolab/CellphoneDB)  results are merged based on sender and receiver roles. The interaction strength from   [CellChat](https://github.com/jinworks/CellChat?tab=readme-ov-file), and the mean and maximum interaction strengths from  [CellPhoneDB](https://github.com/ventolab/CellphoneDB)  are MinMax normalized (to the range 0-1). The average of these three normalized scores is calculated as the consensus score:

$$
Consensus Score= (𝑁𝑜𝑟𝑚_𝐶𝑒𝑙𝑙𝐶ℎ𝑎𝑡 + 𝑁𝑜𝑟𝑚_𝐶𝑃𝐷𝐵_𝑀𝑒𝑎𝑛 + 𝑁𝑜𝑟𝑚_𝐶𝑃𝐷𝐵_𝑀𝑎𝑥) / 3
$$

### 2.1 Cell-Cell Interaction Analysis Based on CellChat

#### System Requirements

**R Version:** ==4.3.3

**Operating System:** Windows/Linux

**Memory:** ≥ 16GB (recommended 40GB+ for large single-cell datasets)

#### **Environment Setup**

**Install CRAN Packages**

```R
install.packages(c("Seurat", "ggplot2", "patchwork", "dplyr", 
                   "future", "RColorBrewer", "stringr"))
```

**Install Bioconductor Packages**

```R
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("ComplexHeatmap")
```

**Install CellChat**

```R
install.packages("devtools")
devtools::install_github("sqjin/CellChat")
```

### 2.2 Cell-Cell Interaction Analysis Based on CellPhoneDB

#### System Requirements

**Python Version:** ==3.12.0

**Operating System:** Windows/Linux

**Memory:** ≥ 16GB (recommended 40GB+ for large single-cell datasets)

#### Environment Setup

```bash
cd ./CellPhoneAnalysis
conda create -n cpdb python==3.12.0
conda activate cpdb
pip install -r requirements.txt
```

#### Microenvironment Analysis Preparation

```bash
python prepare_microenvs_h5ad.py
```

#### Differential Expression Gene Preparation

```bash
python prepare_DEGs_h5ad.py
```

#### Run CellPhoneDB for Cell-Cell Interaction Analysis

```bash
python CellPhoneAnalysis.py
```

### 2.3 Gold Standard Construction

Generate the gold standard as labels for fine-tuning the foundation model [GeneCompass](https://github.com/xCompass-AI/GeneCompass).

```bash
cd path/to/CCC-GeneCompass
python building_gold_standard_database.py
```



## 3. Generating Embeddings

Thefoundation model [GeneCompass](https://github.com/xCompass-AI/GeneCompass) performs cell-cell communication analysis by converting the single-cell data into high-dimensional vector representations called Embeddings. This step converts the normalized and token-encoded single-cell data from the **1.2 Single-Cell Transcriptomics Data Quality Control and Normalization** process into Embeddings.

```bash
cd path/to/CCC-GeneCompass
python generate_embeddings.py
```

Generating Embeddings requires significant computational resources and time. For convenient verification, we provide an example of pre-generated Embeddings: https://pan.baidu.com/s/1X97G7PdJRHXYn5vako9RnQ?pwd=1uyh (Extraction code: 1uyh)



## 4. Cell-Cell Interaction Analysis

#### Environment Setup

```bash
cd path/to/CCC-GeneCompass
conda create -n ccc python==3.12.0
conda activate ccc
pip install -r requirements.txt
```

If you encounter an error installing `transformers==4.30.0`, you can execute the following steps:

```bash
conda install -c conda-forge tokenizers=0.13.3
pip install transformers==4.30.0
```

Fine-tune the foundation model  [GeneCompass](https://github.com/xCompass-AI/GeneCompass)  for the downstream task of cell-cell interaction analysis using the [CellChat](https://github.com/jinworks/CellChat?tab=readme-ov-file) and [CellPhoneDB](https://github.com/ventolab/CellphoneDB)  consensus gold standard and the generated Embeddings, to obtain the cell-cell interaction matrix and visualization results.

```bash
cd path/to/CCC-GeneCompass
python cell_cell_interaction.py
```

For convenient verification, we provide pre-processed normalized data: [tabula_sapiens_liver](https://pan.baidu.com/s/1RsTlTB4aTlwlk5cHtIQtuA?pwd=b8d8#list/path=%2F)，(Extraction code: b8d8)

We also provide a pre-generated gold standard label data file: [complete_labeled_interactions.csv](https://pan.baidu.com/s/1tcELkJexk3LwN6frNNykbA?pwd=jmc6) (Extraction code: jmc6)

Pretrained models of GeneCompass on 100 million single-cell transcriptomes from humans and mice. Put pretrained_model dir under main path.('./pretrained_models/GeneCompass_Small', './pretrained_models/GeneCompass_Base')

| Model             | Description                         | Download                                           |
| ----------------- | ----------------------------------- | -------------------------------------------------- |
| GeneCompass_Small | Pretrained on 6-layer GeneCompass.  | [Link](https://www.scidb.cn/en/anonymous/SUZOdk1y) |
| GeneCompass_Base  | Pretrained on 12-layer GeneCompass. | [Link](https://www.scidb.cn/en/anonymous/SUZOdk1y) |
