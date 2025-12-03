# coding: utf-8
#脚本4，预处理，输出
import anndata
import scanpy as sc
import numpy as np
import pickle
import tqdm
from datasets import Dataset,Features,Sequence,Value
import os
import scipy.sparse as sp
# with open("/data/share/ict01/public/tongyuan_token.pickle","rb") as f:
#     token_data = pickle.load(f)
# with open("/data/home/ict01/public/tongyuan_h&m_token.pickle","rb") as f:
#     token_data = pickle.load(f)
def tokenize_cell(gene_vector, gene_list,token_dict):
    """
    Convert normalized gene expression vector to tokenized rank value encoding.
    """
    gene_vector = sp.csr_matrix(gene_vector)
    gene_vector = gene_vector.toarray().flatten()
    nonzero_mask = np.nonzero(gene_vector)[0]
    sorted_indices = np.argsort(-gene_vector[nonzero_mask])
    gene_list = np.array(gene_list)[nonzero_mask][sorted_indices]
    f_token = [token_dict[gene] for gene in gene_list]
    value = gene_vector[nonzero_mask][sorted_indices]
    return f_token, value.tolist()
    # nonzero_mask = np.nonzero(gene_vector)[0]
    # sorted_indices = np.argsort(-gene_vector[nonzero_mask])
    # gene_list = np.array(gene_list)[nonzero_mask][sorted_indices]
    # f_token = [token_dict[gene] for gene in gene_list]
    # value = gene_vector[nonzero_mask][sorted_indices]
    # return f_token, value.tolist()
def Normalized(adata,dict_path,Parmerters):
    data = adata.X
    # 将稀疏矩阵转换为 numpy.matrix 对象
    data = sp.csr_matrix(data)
    matrix_a = data.toarray()
    # matrix = np.matrix(dense_mat)
    # matrix_a = np.array(matrix)
    with open(dict_path, 'rb') as f:
        dict_gene = pickle.load(f)
    # list_t存储非零中值
    gene_list_t = adata.var.index.to_list()
    gene_nonzero_t = []
    for i in gene_list_t:
        if i in dict_gene:
            gene_nonzero_t.append(dict_gene[i])
        else:
            gene_nonzero_t.append(1)
    gene_nonzero_t = np.array(gene_nonzero_t)

    # 计算每个细胞的非零值的和
    per_cell_nonzero_sum = np.sum(matrix_a, axis=1)
    nonzero_count = np.count_nonzero(matrix_a, axis=1)
    per_cell_nonzero_sum[nonzero_count == 0] = 0
    #subview_norm_array = np.nan_to_num(matrix_a[:, :].T / per_cell_nonzero_sum * Parmerters / gene_nonzero_t[:, None])
    subview_norm_array = np.nan_to_num(matrix_a[:, :].T / gene_nonzero_t[:, None])
    subview_norm_array = np.array(subview_norm_array.T)
    adata.X = subview_norm_array
    return adata

def log1p(adata):
    sc.pp.log1p(adata, base=2)
    return adata

def rank_value(adata,gene_token_path):
    with open(gene_token_path, 'rb') as f:
        token = pickle.load(f)
    # input_ids = np.zeros((len(adata.X), 2048))
    input_ids = np.zeros((adata.X.shape[0], 2048)) # Initialize input_ids as a 2D array filled with zeros
    values = np.zeros((adata.X.shape[0], 2048))  # Initialize values as a 2D array filled with zeros
    length = []

    gene_id = adata.var.index.to_list()
    for index, i in enumerate(tqdm.tqdm(adata.X)):
        tokenizen, value = tokenize_cell(i, gene_id, token)
        # 处理2048截断
        if len(tokenizen) > 2048:
            input_ids[index] = tokenizen[:2048]
            values[index] = value[:2048]
            length.append(2048)
        else:
            input_ids[index, :len(tokenizen)] = tokenizen
            values[index, :len(value)] = value
            input_ids[index, len(tokenizen):] = 0  # Fill remaining elements with zeros
            values[index, len(value):] = 0  # Fill remaining elements with zeros
            length.append(len(tokenizen))
    return input_ids,length,values

def transfor_out(specices_str,length,input_ids,values):
    if specices_str == 'human':
        specices_int = 0
    elif specices_str =='mouse':
        specices_int = 1
    specices_int = [specices_int] * adata.X.shape[0]
    specices_int = [[x] for x in specices_int]
    length = [[x] for x in length]

    cell_type = adata.obs['cell_type'].astype(str).tolist()
    print("目录： ", adata.obs.columns)
    print("细胞类型： ", adata.obs['cell_type'].astype(str).tolist())

    data_out = {'input_ids': input_ids,'values':values,'length': length,'species': specices_int, 'cell_type': cell_type}
    features = Features({
        'input_ids': Sequence(feature=Value(dtype='int32')),
        'values': Sequence(feature=Value(dtype='float32')),
        'length': Sequence(feature=Value(dtype='int16')),
        'species':Sequence(feature=Value(dtype='int16')),
        'cell_type': Value(dtype='string')
    })
    dataset = Dataset.from_dict(data_out, features=features)
    return dataset

def save_disk(patch_str,dataset,length):
    dataset.save_to_disk(patch_str)
    sorted_list = sorted(length)
    out_path = patch_str + '/sorted_length.pickle'
    with open(out_path, 'wb') as f:
        pickle.dump(sorted_list, f)

dict_path = './prior_knowledge/public/human_gene_median_after_filter.pickle' #中值字典路径
gene_token_path = './prior_knowledge/human_mouse_tokens.pickle'
specices_str = 'human'  #or mouse
dir_path = '/mnt/data_sdb/wangx/data/SingleCell/filtered_data/TabulaSapiens/tabula_sapiens_liver/' #存放filter细胞数据集的路径
out_path = '/mnt/data_sdb/wangx/data/SingleCell/normalized_data/TabulaSapiens/tabula_sapiens_liver/'
patch_id = 1

for filename in os.listdir(dir_path):
    if filename.endswith('.h5ad'):
        print("Started:")
        print(filename)
        # with open('/data/home/ict01/dzx_project/monkey/monkey.pickle', 'rb') as file:
        #     adata = pickle.load(file)
        adata = anndata.read_h5ad(os.path.join(dir_path, filename))
        print(adata.X.shape)
        #1.Normalized
        print("1.Normalized")
        adata = Normalized(adata,dict_path,Parmerters=1e4)
        #2.log1p
        print("2.log1p")
        adata = log1p(adata)
        #3.Rank
        print("3.rank value")
        input_ids,length,values = rank_value(adata,gene_token_path)

        #4.输出Hungface_dataset
        print("4.dataset transfor")
        datasets = transfor_out(specices_str,length,input_ids,values)
        #5.存储
        patch = filename.split('.')[0]
        # patch = 'patch'+ str(patch_id)

        path = out_path+patch
        if not os.path.exists(path):
            os.makedirs(path)
        print("5.save disk to : {}".format(patch))
        save_disk(path,datasets,length)
        # patch_id = patch_id+1
print("处理完毕!")

