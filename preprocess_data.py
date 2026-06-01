#!/usr/bin/env python3
"""
Preprocess h5ad → filtered h5ad + tokenized Dataset + cell-type-aggregated Dataset.
====================================================================================
Usage:
  python preprocess_data.py --h5ad /path/to/organ.h5ad --output data/kidney \
      --tokens prior_knowledge/human_mouse_tokens.pickle \
      --medians prior_knowledge/public/human_gene_median_after_filter.pickle
"""
import sys,os,pickle,argparse
import numpy as np,anndata
from tqdm import tqdm
from datasets import Dataset,Features,Sequence,Value

def preprocess(h5ad_path,output_dir,tokens_path,medians_path,max_len=2048,cell_type_col=None):
    os.makedirs(output_dir,exist_ok=True)

    print(f"Loading {h5ad_path}...");adata=anndata.read_h5ad(h5ad_path)
    ens_ids=list(adata.var.index)

    # Identify cell_type column
    if cell_type_col and cell_type_col in adata.obs.columns:
        pass
    else:
        for c in['cell_type','cell_ontology_class']:
            if c in adata.obs.columns:adata.obs['cell_type']=adata.obs[c].astype(str);break
    if'cell_type'not in adata.obs.columns:
        raise ValueError("No cell_type column found in h5ad.obs")

    # Load token dictionary
    with open(tokens_path,'rb')as f:gtk=pickle.load(f)
    # Filter genes by token dict
    kg=[i for i,g in enumerate(ens_ids)if g in gtk]
    ens_ids=[ens_ids[i]for i in kg]
    adata=adata[:,kg]
    X=adata.X.toarray()if hasattr(adata.X,'toarray')else np.array(adata.X)
    # Filter cells with >6 expressed genes
    fg=np.count_nonzero(X,axis=1)>6;adata=adata[fg];X=X[fg]
    print(f"Filtered: {adata.shape}, {adata.obs['cell_type'].nunique()} types")

    # Save filtered h5ad
    filtered_path=f"{output_dir}/filtered.h5ad"
    adata.write_h5ad(filtered_path)
    print(f"Saved: {filtered_path}")

    # Normalize & tokenize
    with open(medians_path,'rb')as f:gmed=pickle.load(f)
    mv=np.array([gmed.get(g,1)for g in ens_ids]);Xn=np.nan_to_num(X/mv);Xn=np.log2(Xn+1)
    ids=np.zeros((Xn.shape[0],max_len),dtype=np.int32);vs=np.zeros((Xn.shape[0],max_len),dtype=np.float32);ls=[]
    for i in tqdm(range(Xn.shape[0]),desc='Tokenize'):
        r=Xn[i];nz=np.nonzero(r)[0];si=np.argsort(-r[nz])
        sg=np.array(ens_ids)[nz][si];sv=r[nz][si]
        tk=np.array([gtk.get(g,0)for g in sg],dtype=np.int32)
        al=min(len(tk),max_len);ids[i,:al]=tk[:al];vs[i,:al]=sv[:al];ls.append([al])
    sp=[[0]]*Xn.shape[0];ct=adata.obs['cell_type'].tolist()
    dd={'input_ids':ids.tolist(),'values':vs.tolist(),'length':ls,'species':sp,'cell_type':ct}
    feat=Features({'input_ids':Sequence(Value('int32')),'values':Sequence(Value('float32')),
        'length':Sequence(Value('int16')),'species':Sequence(Value('int16')),'cell_type':Value('string')})
    ds_path=f"{output_dir}/single_cell_dataset"
    Dataset.from_dict(dd,features=feat).save_to_disk(ds_path)
    print(f"Tokenized dataset: {ds_path} ({len(ct)} cells, {len(set(ct))} types)")

    # Build cell-type-aggregated dataset
    print("Building cell-type-aggregated profiles...")
    ct_pb={}
    for ct_name in sorted(set(adata.obs['cell_type'])):
        mask=(adata.obs['cell_type']==ct_name).values
        ct_pb[ct_name]=X[mask].mean(axis=0)

    pb_ids=np.zeros((len(ct_pb),max_len),dtype=np.int32)
    pb_vals=np.zeros((len(ct_pb),max_len),dtype=np.float32)
    pb_lens=[];pb_cts=sorted(ct_pb.keys())
    for i,ct_name in enumerate(tqdm(pb_cts,desc='Tokenize pseudo-bulk')):
        expr=ct_pb[ct_name];norm=expr/mv;norm=np.nan_to_num(norm);norm=np.log2(norm+1)
        nz=np.nonzero(norm)[0];si=np.argsort(-norm[nz])
        sg=np.array(ens_ids)[nz][si];sv=norm[nz][si]
        tk=np.array([gtk.get(g,0)for g in sg],dtype=np.int32)
        al=min(len(tk),max_len)
        pb_ids[i,:al]=tk[:al];pb_vals[i,:al]=sv[:al];pb_lens.append([al])

    pb_dd={'input_ids':pb_ids.tolist(),'values':pb_vals.tolist(),
           'length':pb_lens,'species':[[0]]*len(pb_cts),'cell_type':pb_cts}
    pb_path=f"{output_dir}/cell_type_aggregated"
    Dataset.from_dict(pb_dd,features=feat).save_to_disk(pb_path)
    print(f"Cell-type-aggregated dataset: {pb_path} ({len(pb_cts)} types)")

    return filtered_path,ds_path,pb_path


if __name__=='__main__':
    p=argparse.ArgumentParser(description='Preprocess single-cell h5ad for CCC-GeneCompass')
    p.add_argument('--h5ad',required=True,help='Input h5ad file')
    p.add_argument('--output',required=True,help='Output directory')
    p.add_argument('--tokens',required=True,help='human_mouse_tokens.pickle path')
    p.add_argument('--medians',required=True,help='human_gene_median_after_filter.pickle path')
    p.add_argument('--cell_type',default=None,help='Cell type column name (auto-detect if not set)')
    p.add_argument('--max_len',type=int,default=2048,help='Max sequence length')
    args=p.parse_args()
    preprocess(args.h5ad,args.output,args.tokens,args.medians,args.max_len,args.cell_type)
