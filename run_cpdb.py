#!/usr/bin/env python3
"""
CellPhoneDB v5 Analysis — Prepare inputs and run statistical analysis.
======================================================================
Usage:
  python run_cpdb.py --h5ad /path/to/organ.h5ad --cpdb_db /path/to/cellphonedb.zip \
      --cpdb_genes /path/to/CellPhoneAnalysis/v5.0.0/ --output data/{organ}/cellphonedb/
"""
import sys,os,shutil,argparse
import numpy as np,pandas as pd,anndata
from cellphonedb.src.core.methods import cpdb_statistical_analysis_method


def prepare_cpdb_inputs(h5ad_path,cpdb_data_dir,output_dir,max_cells_per_type=300,threads=8):
    """Prepare counts.txt, meta.txt, microenvs.txt, degs.txt for CPDB v5."""
    os.makedirs(output_dir,exist_ok=True)

    # --- Load CPDB database genes ---
    gi=pd.read_csv(f"{cpdb_data_dir}/gene_input.csv")
    db_genes=set(gi['gene_name'].dropna())
    pi_path=f"{cpdb_data_dir}/protein_input.csv"
    if os.path.exists(pi_path):
        pi=pd.read_csv(pi_path)
        if'uniprot'in pi.columns:
            db_genes|=set(pi['uniprot'].dropna().str.split('_').str[0])
    print(f"CPDB database genes: {len(db_genes)}")

    # --- Load h5ad & filter to CPDB genes ---
    print(f"Loading {h5ad_path}...")
    adata=anndata.read_h5ad(h5ad_path)
    # Extract gene symbols (Tabula Sapiens format: GENE_ENSG -> GENE)
    gene_syms=[str(g).split('_')[0] for g in adata.var['feature_name']]
    # Set cell_type
    for c in['cell_type','cell_ontology_class']:
        if c in adata.obs.columns:
            adata.obs['cell_type']=adata.obs[c].astype(str)
            break

    # Filter to CPDB genes
    kg=[i for i,g in enumerate(gene_syms) if g in db_genes]
    adata=adata[:,kg]
    gene_syms_f=[gene_syms[i] for i in kg]
    X=adata.X.toarray() if hasattr(adata.X,'toarray') else np.array(adata.X)
    print(f"Filtered: {len(gene_syms_f)} genes × {X.shape[0]} cells")

    # --- Subsample cells (max N per type) ---
    cc=adata.obs['cell_type'].value_counts()
    valid=cc[cc>=5].index.tolist()
    print(f"Valid cell types (≥5 cells): {len(valid)}")
    idx=[]
    for ct in valid:
        indices=adata.obs[adata.obs['cell_type']==ct].index
        n=max(1,min(max_cells_per_type,len(indices)))
        idx.extend(np.random.choice(indices,n,replace=False))
    sub=adata[idx]
    Xs=sub.X.toarray() if hasattr(sub.X,'toarray') else np.array(sub.X)
    Xs_int=(Xs*10).astype(int)  # Scale to integers for CPDB
    print(f"Subsampled: {Xs_int.shape} (max {max_cells_per_type} cells/type)")

    # --- Write counts.txt (genes × cells, tab-separated) ---
    counts_df=pd.DataFrame(Xs_int.T,index=gene_syms_f,columns=sub.obs.index)
    counts_df.to_csv(f"{output_dir}/counts.txt",sep='\t')
    print(f"Written: counts.txt ({counts_df.shape[0]} genes × {counts_df.shape[1]} cells)")

    # --- Write meta.txt (cell → cell_type) ---
    meta_df=pd.DataFrame({'cell_type':sub.obs['cell_type'].values},index=sub.obs.index)
    meta_df.to_csv(f"{output_dir}/meta.txt",sep='\t')
    print(f"Written: meta.txt ({meta_df.shape[0]} cells × {meta_df['cell_type'].nunique()} types)")

    # --- Write microenvs.txt (cell_type → all) ---
    with open(f"{output_dir}/microenvs.txt",'w') as f:
        for ct in valid:
            f.write(f"{ct}\tall\n")
    print("Written: microenvs.txt")

    # --- Generate DEGs (differential expression per cell type) ---
    degs=[]
    for ct in valid:
        mask=(sub.obs['cell_type']==ct).values
        other=~mask
        fc=np.log2(Xs[mask].mean(0)+1)-np.log2(Xs[other].mean(0)+1)
        pct=(Xs[mask]>0).mean(0)
        # Select genes with log2FC > 0.5 and expressed in >10% of cells
        hit=np.where((fc>0.5)&(pct>0.1))[0][:300]
        for i in hit:
            degs.append({
                'gene':gene_syms_f[i],'cluster':ct,
                'log2fc':round(float(fc[i]),3),'pvalue':0.01,
                'pct.1':round(float(pct[i]),3),'pct.2':0.3
            })
    degs_df=pd.DataFrame(degs)
    degs_df.to_csv(f"{output_dir}/degs.txt",sep='\t',index=False)
    print(f"Written: degs.txt ({len(degs)} DEGs)")

    # --- Write active_tfs.txt (empty, in cluster format) ---
    pd.DataFrame(columns=['gene','cluster']).to_csv(f"{output_dir}/active_tfs.txt",sep='\t',index=False)
    print("Written: active_tfs.txt")
    print(f"\nAll CPDB inputs prepared in: {output_dir}/")
    return valid


def run_cpdb(cpdb_db_path,output_dir,threads=8):
    """Run CellPhoneDB v5 statistical analysis."""
    print(f"\nRunning CellPhoneDB v5 statistical analysis ({threads} threads)...")
    cpdb_statistical_analysis_method.call(
        cpdb_file_path=cpdb_db_path,
        meta_file_path=f"{output_dir}/meta.txt",
        counts_file_path=f"{output_dir}/counts.txt",
        counts_data='gene_name',
        output_path=output_dir,
        threshold=0.1,
        result_precision=3,
        score_interactions=True,
        threads=threads)
    # Rename output for downstream pipeline compatibility
    for f_ in os.listdir(output_dir):
        if'statistical_analysis_significant_means'in f_:
            shutil.copy(f"{output_dir}/{f_}",f"{output_dir}/significant_means.txt")
            print(f"  Copied: {f_} → significant_means.txt")
    print("CellPhoneDB analysis complete.")


if __name__=='__main__':
    p=argparse.ArgumentParser(description='CellPhoneDB v5 Gold Standard')
    p.add_argument('--h5ad',required=True,help='Original h5ad file (raw, pre-filtering)')
    p.add_argument('--cpdb_db',required=True,help='Path to cellphonedb.zip')
    p.add_argument('--cpdb_genes',required=True,help='Dir containing gene_input.csv, protein_input.csv')
    p.add_argument('--output',required=True,help='Output directory for CPDB results')
    p.add_argument('--max_cells',type=int,default=300,help='Max cells per type for subsampling')
    p.add_argument('--threads',type=int,default=8,help='CPU threads')
    p.add_argument('--skip_prepare',action='store_true',help='Skip input preparation (already done)')
    args=p.parse_args()

    if not args.skip_prepare:
        prepare_cpdb_inputs(args.h5ad,args.cpdb_genes,args.output,args.max_cells,args.threads)
    run_cpdb(args.cpdb_db,args.output,args.threads)
