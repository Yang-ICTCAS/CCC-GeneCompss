#!/usr/bin/env python3
"""
GeneCompass CCI Pipeline — Standalone Inference on Fixed Test Set
===================================================================
Loads a trained model and the saved test set, predicts, and reports
Spearman ρ consistency metrics. No collator dependency.
"""
import sys,os,pickle,json,warnings,logging,argparse
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score
from scipy.stats import spearmanr,pearsonr
from datasets import load_from_disk
import torch
from tqdm import tqdm

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
logger=logging.getLogger(__name__)

def evaluate_model(model_path,test_dataset_dir,output_dir=None,token_dict_path=None,proj_root=None,batch_size=4):
    """Load trained model + fixed test set, compute Spearman ρ."""

    if proj_root and proj_root not in sys.path:
        sys.path.insert(0,proj_root)

    # ---- Load model ----
    from genecompass import BertForSequenceClassification
    from genecompass.utils import load_prior_embedding

    know={}
    if token_dict_path:
        try:
            out=load_prior_embedding(token_dictionary_or_path=token_dict_path)
            know=dict(zip(['promoter','co_exp','gene_family','peca_grn','homologous_gene_human2mouse'],out))
        except:pass

    logger.info(f"Loading model from {model_path}")
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=BertForSequenceClassification.from_pretrained(model_path,num_labels=1,
        output_attentions=False,output_hidden_states=False,knowledges=know)
    model.to(device);model.eval()

    # ---- Load test set ----
    logger.info(f"Loading test set from {test_dataset_dir}")
    tds=load_from_disk(test_dataset_dir)
    n=len(tds)
    logger.info(f"Test samples: {n}")

    # ---- Predict (simple manual batching, no collator) ----
    preds=[]
    with torch.no_grad():
        for i in tqdm(range(0,n,batch_size),desc='Predicting'):
            end=min(i+batch_size,n)
            ids=torch.tensor([tds[j]['input_ids'] for j in range(i,end)]).long().to(device)
            vs=torch.tensor([tds[j]['values'] for j in range(i,end)]).float().to(device)
            out=model(input_ids=ids,values=vs)
            preds.extend(out.logits.cpu().numpy().flatten().tolist())

    pv=np.array(preds)
    tv=np.array(tds['label'])

    # ---- Metrics ----
    sp_rho,_=spearmanr(tv,pv)

    # Bootstrap 95% CI (paired resampling)
    n_boot=2000;np.random.seed(42)
    boot_rhos=[]
    for _ in range(n_boot):
        idx=np.random.choice(n,n,replace=True)
        rho_b,_=spearmanr(tv[idx],pv[idx])
        boot_rhos.append(rho_b)
    sp_ci_low=np.percentile(boot_rhos,2.5)
    sp_ci_high=np.percentile(boot_rhos,97.5)

    # Permutation p-value
    n_perm=5000
    perm_rhos=[spearmanr(tv,np.random.permutation(pv))[0] for _ in range(n_perm)]
    perm_pval=(np.sum(np.abs(perm_rhos)>=np.abs(sp_rho))+1)/(n_perm+1)

    # Secondary regression metrics
    pr=pearsonr(tv,pv)[0] if n>2 else np.corrcoef(tv,pv)[0,1]
    r2=r2_score(tv,pv) if n>1 else 0.
    rmse=np.sqrt(mean_squared_error(tv,pv))

    # Quality rating
    if sp_rho>=0.7:quality="EXCELLENT"
    elif sp_rho>=0.5:quality="GOOD"
    elif sp_rho>=0.3:quality="MODERATE"
    else:quality="WEAK"

    logger.info(f"  Spearman ρ={sp_rho:.4f} (95%CI: [{sp_ci_low:.4f}, {sp_ci_high:.4f}], p_perm={perm_pval:.4f}) → {quality}")
    logger.info(f"  Pearson  r={pr:.4f} R²={r2:.4f} RMSE={rmse:.4f}")

    results={
        'model_path':model_path,'test_set':test_dataset_dir,'n_test_samples':n,
        'consistency_primary':{
            'metric':'Spearman_rank_correlation','rho':round(float(sp_rho),4),
            'ci_95_low':round(float(sp_ci_low),4),'ci_95_high':round(float(sp_ci_high),4),
            'p_value_permutation':round(float(perm_pval),4),'interpretation':quality
        },
        'regression_secondary':{
            'pearson_r':round(float(pr),4),'r_squared':round(float(r2),4),
            'rmse':round(float(rmse),4)
        }
    }

    if output_dir:
        os.makedirs(output_dir,exist_ok=True)
        json.dump(results,open(f"{output_dir}/inference_metrics.json",'w'),indent=2)
        import pandas as pd
        pd.DataFrame({'true':tv,'predicted':pv}).to_csv(
            f"{output_dir}/predictions.csv",index=False)
        logger.info(f"Results saved: {output_dir}/inference_metrics.json, predictions.csv")

    return results


if __name__=='__main__':
    p=argparse.ArgumentParser(description='GeneCompass CCI Inference on Fixed Test Set')
    p.add_argument('--model',required=True,help='Path to trained model directory')
    p.add_argument('--test_set',required=True,help='Path to saved test dataset')
    p.add_argument('--output',default=None,help='Output directory for results')
    p.add_argument('--token_dict',default=None,help='Token dictionary path')
    p.add_argument('--proj_root',default='.',help='Project root')
    p.add_argument('--batch',type=int,default=4,help='Inference batch size')
    args=p.parse_args()
    evaluate_model(args.model,args.test_set,args.output,args.token_dict,args.proj_root,args.batch)
