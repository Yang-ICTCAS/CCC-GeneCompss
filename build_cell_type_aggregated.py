#!/usr/bin/env python3
"""
Build Cell-Type Cell-Type-Aggregated Dataset
=====================================
Replaces single-cell random sampling with deterministic per-cell-type mean expression.
Each cell type → one cell-type-aggregated profile → zero sampling noise, fully reproducible.
"""
import sys,os,pickle,argparse
import numpy as np,anndata
from tqdm import tqdm
from datasets import Dataset,Features,Sequence,Value

def build_cell_type_aggregated(h5ad_path,output_path,token_dict_path,median_dict_path,max_len=2048):
    """Compute cell-type mean expression → normalize → tokenize → save as Dataset."""
    print(f"Loading {h5ad_path}...")
    adata=anndata.read_h5ad(h5ad_path)

    # Load token dict and gene median
    with open(token_dict_path,'rb')as f:td=pickle.load(f)
    with open(median_dict_path,'rb')as f:gmed=pickle.load(f)

    # Get cell types
    if'cell_type'not in adata.obs.columns:
        for c in['cell_type','cell_ontology_class']:
            if c in adata.obs.columns:adata.obs['cell_type']=adata.obs[c].astype(str);break
    cell_types=sorted(set(adata.obs['cell_type']))
    print(f"Cell types: {len(cell_types)}: {cell_types}")

    # Get Ensembl IDs and expression matrix
    ens_ids=list(adata.var.index)
    if hasattr(adata.X,'toarray'):X=adata.X.toarray()
    else:X=np.array(adata.X)

    # Compute per-cell-type mean expression (cell-type-aggregated)
    print("Computing cell-type cell-type-aggregated means...")
    pb_profiles={}
    for ct in cell_types:
        mask=(adata.obs['cell_type']==ct).values
        pb_profiles[ct]=X[mask].mean(axis=0)

    # Normalize & tokenize each cell-type-aggregated (same pipeline as single-cell)
    mv=np.array([gmed.get(g,1)for g in ens_ids])
    ids=np.zeros((len(cell_types),max_len),dtype=np.int32)
    values=np.zeros((len(cell_types),max_len),dtype=np.float32)
    lengths=[]
    species=[[0]]*len(cell_types)

    for i,ct in enumerate(tqdm(cell_types,desc='Tokenize cell-type-aggregated')):
        expr=pb_profiles[ct]
        # Normalize by gene median
        norm=expr/mv
        norm=np.nan_to_num(norm)
        norm=np.log2(norm+1)
        # Sort by expression (descending)
        nz=np.nonzero(norm)[0]
        si=np.argsort(-norm[nz])
        sorted_genes=np.array(ens_ids)[nz][si]
        sorted_vals=norm[nz][si]
        # Tokenize
        tk=np.array([td.get(g,0)for g in sorted_genes],dtype=np.int32)
        al=min(len(tk),max_len)
        ids[i,:al]=tk[:al]
        values[i,:al]=sorted_vals[:al]
        lengths.append([al])

    # Build Dataset (one row per cell type)
    dd={'input_ids':ids.tolist(),'values':values.tolist(),
        'length':lengths,'species':species,'cell_type':cell_types}
    feat=Features({'input_ids':Sequence(Value('int32')),'values':Sequence(Value('float32')),
        'length':Sequence(Value('int16')),'species':Sequence(Value('int16')),
        'cell_type':Value('string')})
    ds=Dataset.from_dict(dd,features=feat)
    ds.save_to_disk(output_path)
    print(f"CellTypeAggregated dataset saved: {output_path} ({len(ds)} cell types)")
    return ds

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--h5ad',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--tokens',required=True,help='human_mouse_tokens.pickle')
    p.add_argument('--medians',required=True,help='human_gene_median_after_filter.pickle')
    args=p.parse_args()
    build_cell_type_aggregated(args.h5ad,args.output,args.tokens,args.medians)
