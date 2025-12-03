# coding: utf-8
# Preprocess program 1: filter cells
import anndata
import pickle
import os
import pandas as pd
from scipy.stats import zscore
import numpy as np
import scanpy as sc
import warnings

warnings.filterwarnings("ignore")


def get_gene_list(species: str, file_list: list[str]):
    take_list = []
    protein_list = []
    miRNA_list = []
    mitochondria_list = []
    for i in file_list:
        if i.split('/')[-1].split('_')[0] == species:
            take_list.append(i)
    for i in take_list:
        if i.split('.')[-1] == "txt":
            with open(i, "r") as f:
                name_temp = f.name.split('/')[-1].split('_')[1]
                if name_temp == 'protein':
                    protein_list.extend(line.split()[0] for line in f)
                elif name_temp == 'miRNA.txt':
                    miRNA_list.extend(line.split()[0] for line in f)
        elif i.split('.')[-1] == "xlsx":
            df = pd.read_excel(i)
            mitochondria_list = df.iloc[:, 1].tolist()
    return protein_list, miRNA_list, mitochondria_list


def gene_id_filter(adata: anndata, gene_list: list[str]) -> anndata:
    mask = adata.var.index.isin(gene_list)
    return adata[:, mask]


def id_name_match(protein_list, miRNA_list, dict1):
    p_l = [dict1.get(i, 'delete') for i in protein_list]
    m_l = [dict1.get(i, 'delete') for i in miRNA_list]
    return p_l, m_l


def id_name_match1(name_list, dict1_reverse):
    # Use the reversed dictionary to map Ensembl IDs to gene symbols
    gene_symbols = [dict1_reverse.get(i, 'delete') for i in name_list]
    return gene_symbols


def normal_filter(adata: anndata, mito_list: list[str]) -> anndata:
    # Ensure adata.X is a dense matrix for sum operation
    if isinstance(adata.X, np.ndarray):
        total_counts = adata.X.sum(axis=1)
    else:
        total_counts = adata.X.sum(axis=1).A1

    idx = total_counts > 0
    adata = adata[idx, :]
    total_counts = total_counts[idx]

    gene_name = adata.var.gene_symbols if 'gene_symbols' in adata.var else adata.var.feature_name

    gene_name = gene_name.tolist()
    print("gene_name: ", gene_name)
    print("mito_list: ", mito_list)
    index = [element in mito_list for element in gene_name]
    print("index: ", index)
    mito_adata = adata[:, index]
    if isinstance(mito_adata.X, np.ndarray):
        mito_counts = mito_adata.X.sum(axis=1)
    else:
        mito_counts = mito_adata.X.sum(axis=1).A1
    mito_percentage = mito_counts / total_counts

    total_counts_zscore = zscore(total_counts)
    mito_percentage_zscore = zscore(mito_percentage)

    keep_cells = ((total_counts_zscore > -3) & (total_counts_zscore < 3) &
                  (mito_percentage_zscore > -3) & (mito_percentage_zscore < 3))
    return adata[keep_cells, :]


def gene_number_filter(adata, gene_list: list[str]):
    indices = adata.var.index.isin(gene_list)
    f_adata = adata[:, indices]
    if isinstance(f_adata.X, np.ndarray):
        data = f_adata.X
    else:
        data = f_adata.X.toarray()
    mask = np.count_nonzero(data, axis=1) > 6
    return adata[mask]


# Configuration settings
f_list = [
    "./prior_knowledge/public/mouse_protein_coding.txt",
    "./prior_knowledge/public/human_protein_coding.txt",
    "/public/monkey_mulatta_protein_coding.txt",
    "/public/mouse_miRNA.txt",
    "/public/human_miRNA.txt",
    "/public/monkey_mulatta_miRNA.txt",
    "/public/human_mitochondria.xlsx",
    "/mouse_mitochondria.xlsx",
    "/monkey_mulatta_MT.xlsx"
]
species_str = 'human'
gene_id_name_path = './prior_knowledge/public/Gene_id_name_dict1.pickle'
gene_id_path = './prior_knowledge/human_mouse_tokens.pickle'
dir_path = '/mnt/data_sdb/wangx/data/SingleCell/ori_data/TabulaSapiens/tabula_sapiens_liver/'
out_path = '/mnt/data_sdb/wangx/data/SingleCell/filtered_data/TabulaSapiens/tabula_sapiens_liver/'

try:
    with open(gene_id_name_path, 'rb') as f:
        dict1 = pickle.load(f)
    with open(gene_id_path, 'rb') as f:
        gene_id = pickle.load(f)

    # Reverse the dictionary to map Ensembl IDs to gene symbols
    dict1_reverse = {v: k for k, v in dict1.items()}

    # Print first few items of dict1 to verify its contents
    print(list(dict1.items())[:10])

    protein_list, miRNA_list, mitochondria_list = get_gene_list(species=species_str, file_list=f_list)
    protein_list, miRNA_list = id_name_match(protein_list, miRNA_list, dict1)

    for filename in os.listdir(dir_path):
        if filename.endswith('.h5ad'):
            print(f"Processing {filename}")
            adata = anndata.read_h5ad(os.path.join(dir_path, filename))
            print(f"Oiginal adata: {adata.shape}")

            # Translate IDs to names
            name_list = adata.var.index.to_list()
            print(f"First few gene IDs in name_list: {name_list[:10]}")

            gene_symbols = id_name_match1(name_list, dict1_reverse)
            adata.var['gene_symbols'] = gene_symbols

            # Filter out genes marked as 'delete'
            adata = adata[:, ~(adata.var['gene_symbols'] == "delete")]

            # Debugging: Check the number of genes before filtering
            print(f"Before filtering: {adata.shape}")

            # Filter cells
            adata = normal_filter(adata, mitochondria_list)
            adata = gene_id_filter(adata, gene_id)
            adata = gene_number_filter(adata, protein_list + miRNA_list)
            print("Filtered adata: ", adata.X.shape)

            # Save the processed data
            outpath = os.path.join(out_path, filename)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            adata.write_h5ad(outpath)
            print(f"Saved processed data to {outpath}")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    print('Folder processing complete!')