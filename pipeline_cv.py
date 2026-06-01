#!/usr/bin/env python3
"""
GeneCompass CCI Pipeline — 5-Fold Cross-Validation
=====================================================================
Each fold: train → predict → save model + metrics.
Final: mean ± std of Spearman ρ, Pearson r, R², RMSE across 5 folds.
"""
import sys,os,pickle,json,warnings,logging,argparse
import numpy as np,pandas as pd,torch
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import KFold
from scipy.stats import spearmanr,pearsonr
from tqdm import tqdm
from datasets import Dataset,load_from_disk
from transformers import Trainer,TrainingArguments
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
logger=logging.getLogger(__name__)

class GeneCompassCV:
    def __init__(self,proj_root):
        self.proj_root=proj_root
        if proj_root not in sys.path:sys.path.insert(0,proj_root)
        self.model_path=f"{proj_root}/pretrained_models/GeneCompass_Base"
        self.token_dict_path=f"{proj_root}/prior_knowledge/human_mouse_tokens.pickle"
        self._load_knowledge()
        self._load_lr_genes()

    def _load_knowledge(self):
        from genecompass.utils import load_prior_embedding
        self.know={}
        try:
            out=load_prior_embedding(token_dictionary_or_path=self.token_dict_path)
            self.know=dict(zip(['promoter','co_exp','gene_family','peca_grn','homologous_gene_human2mouse'],out))
        except:pass

    def _load_lr_genes(self):
        self.lr_genes=set()
        cc_comm=f"{self.proj_root}/CellChatAnalysis/cellchat_output/cellchat_communication.csv"
        for p in[cc_comm]:
            if os.path.exists(p):
                df=pd.read_csv(p)
                for c in['ligand','receptor']:
                    if c in df.columns:
                        for g in df[c].dropna().unique():
                            if isinstance(g,str) and g.isalpha() and len(g)>1:self.lr_genes.add(g.upper())
        cpdb_genes=f"{self.proj_root}/CellPhoneAnalysis/v5.0.0/gene_input.csv"
        if os.path.exists(cpdb_genes):
            df=pd.read_csv(cpdb_genes)
            if'gene_name'in df.columns:
                for g in df['gene_name'].dropna():
                    if isinstance(g,str) and g.isalpha() and len(g)>1:self.lr_genes.add(g.upper())
        logger.info(f"LR genes loaded: {len(self.lr_genes)}")

    def _build_all_pairs(self,gs_df,dataset,score_col='Consensus_Score',max_len=2048):
        with open(self.token_dict_path,'rb')as f:td=pickle.load(f)
        ct2idx={}
        for idx,ct in enumerate(dataset['cell_type']):
            ct2idx[str(ct)]=int(idx)
        cls_t,sep_t,pad_t=td.get('<cls>',1),td.get('<sep>',2),td.get('<pad>',0)
        seqs,labs,pair_ids=[],[],[]
        for _,row in tqdm(gs_df.iterrows(),total=len(gs_df),desc='Building pairs'):
            s,r=row['Sender'],row['Receiver']
            if s not in ct2idx or r not in ct2idx:continue
            si,ri=ct2idx[s],ct2idx[r]
            sd,rd=dataset[si],dataset[ri]
            i1=list(sd['input_ids']);v1=list(sd['values'])
            i2=list(rd['input_ids']);v2=list(rd['values'])
            av=max_len-3;tot=len(i1)+len(i2)
            if tot>av:a1=max(1,int(av*len(i1)/tot));a2=max(1,av-a1)
            else:a1,a2=len(i1),len(i2)
            i1,v1=i1[:a1],v1[:a1];i2,v2=i2[:a2],v2[:a2]
            pi=[cls_t]+i1+[sep_t]+i2+[sep_t];pv=[0.0]+v1+[0.0]+v2+[0.0]
            if len(pi)<max_len:pi+=[pad_t]*(max_len-len(pi));pv+=[0.0]*(max_len-len(pv))
            else:pi,pv=pi[:max_len],pv[:max_len]
            seqs.append({'input_ids':pi,'values':pv})
            labs.append(float(row[score_col]))
            pair_ids.append(f"{s}_{r}")
        dd={'input_ids':[s_['input_ids']for s_ in seqs],
            'values':[s_['values']for s_ in seqs],
            'species':[0]*len(seqs),'label':labs}
        return Dataset.from_dict(dd),pair_ids

    def _bootstrap_ci(self,tv,pv,seed=42,n_boot=1000):
        np.random.seed(seed)
        rhos=[]
        for _ in range(n_boot):
            idx=np.random.choice(len(tv),len(tv),replace=True)
            rho_b,_=spearmanr(tv[idx],pv[idx])
            rhos.append(rho_b)
        return float(np.percentile(rhos,2.5)),float(np.percentile(rhos,97.5))

    def cross_validate(self,gs_path,dataset_path,output_dir,organ="Organ",
                       epochs=30,batch_size=1,grad_accum=4,lr=5e-5,n_folds=5):
        os.makedirs(output_dir,exist_ok=True)
        from genecompass import BertForSequenceClassification,DataCollatorForCellClassification
        gs=pd.read_csv(gs_path);ds=load_from_disk(dataset_path)
        n_types=gs['Sender'].nunique()
        logger.info(f"{organ}: {len(gs)} GS pairs, {len(ds)} cell-type profiles, {n_types} types")

        full_ds,pair_ids=self._build_all_pairs(gs,ds,'Consensus_Score')
        logger.info(f"Total pairs for {n_folds}-fold CV: {len(full_ds)}")

        kf=KFold(n_splits=n_folds,shuffle=True,random_state=42)
        all_fold_metrics=[]

        for fold_idx,(train_idx,test_idx) in enumerate(kf.split(full_ds)):
            logger.info(f"\n{'='*50}")
            logger.info(f"FOLD {fold_idx+1}/{n_folds}: Train={len(train_idx)} Test={len(test_idx)}")
            logger.info(f"{'='*50}")

            trd=full_ds.select(train_idx.tolist())
            tds=full_ds.select(test_idx.tolist())

            torch.cuda.empty_cache()
            model=BertForSequenceClassification.from_pretrained(self.model_path,num_labels=1,
                output_attentions=False,output_hidden_states=False,knowledges=self.know)

            ta=TrainingArguments(
                output_dir=f"{output_dir}/fold{fold_idx+1}/ckpt",
                num_train_epochs=epochs,per_device_train_batch_size=batch_size,fp16=True,
                learning_rate=lr,lr_scheduler_type='linear',warmup_steps=50,weight_decay=0.001,
                evaluation_strategy='no',save_strategy='no',
                logging_dir=f"{output_dir}/fold{fold_idx+1}/logs",logging_steps=50,
                report_to=[],remove_unused_columns=True,
                gradient_accumulation_steps=grad_accum,dataloader_num_workers=0)

            trainer=Trainer(model=model,args=ta,
                data_collator=DataCollatorForCellClassification(),train_dataset=trd)
            logger.info(f"Training fold {fold_idx+1} ({epochs} epochs, batch={batch_size}, grad_accum={grad_accum})")
            trainer.train()

            # Predict on test fold
            preds=trainer.predict(tds)
            tv=preds.label_ids.flatten();pv=preds.predictions.flatten()
            sp_rho,_=spearmanr(tv,pv)
            pr=pearsonr(tv,pv)[0]if len(tv)>2 else 0.
            r2=r2_score(tv,pv)if len(tv)>1 else 0.
            rmse=np.sqrt(mean_squared_error(tv,pv))
            ci_low,ci_high=self._bootstrap_ci(tv,pv,seed=42+fold_idx)

            logger.info(f"Fold {fold_idx+1}: ρ={sp_rho:.4f} CI[{ci_low:.4f},{ci_high:.4f}] r={pr:.4f} R²={r2:.4f}")

            # Save fold model + predictions
            fold_dir=f"{output_dir}/fold{fold_idx+1}"
            trainer.save_model(f"{fold_dir}/best_model")
            np.save(f"{fold_dir}/test_true.npy",tv)
            np.save(f"{fold_dir}/test_pred.npy",pv)

            fold_metrics={
                'fold':fold_idx+1,'spearman_rho':float(sp_rho),
                'ci_95_low':ci_low,'ci_95_high':ci_high,
                'pearson_r':float(pr),'r2':float(r2),'rmse':float(rmse),
                'train_n':len(trd),'test_n':len(tds)
            }
            all_fold_metrics.append(fold_metrics)
            with open(f"{fold_dir}/metrics.json",'w')as f:json.dump(fold_metrics,f,indent=2)

            torch.cuda.empty_cache()

        # ---- Aggregate across folds ----
        rhos=[m['spearman_rho']for m in all_fold_metrics]
        prs=[m['pearson_r']for m in all_fold_metrics]
        r2s=[m['r2']for m in all_fold_metrics]
        rmses=[m['rmse']for m in all_fold_metrics]

        mean_rho=np.mean(rhos);std_rho=np.std(rhos,ddof=1)
        mean_pr=np.mean(prs);std_pr=np.std(prs,ddof=1)
        mean_r2=np.mean(r2s);std_r2=np.std(r2s,ddof=1)
        mean_rmse=np.mean(rmses);std_rmse=np.std(rmses,ddof=1)

        if mean_rho>=0.7:quality="EXCELLENT"
        elif mean_rho>=0.5:quality="GOOD"
        elif mean_rho>=0.3:quality="MODERATE"
        else:quality="WEAK"

        logger.info(f"\n{'='*60}")
        logger.info(f"5-FOLD CV RESULTS — {organ}")
        logger.info(f"  Spearman ρ = {mean_rho:.4f} ± {std_rho:.4f}  [{quality}]")
        logger.info(f"  Pearson  r = {mean_pr:.4f} ± {std_pr:.4f}")
        logger.info(f"  R²         = {mean_r2:.4f} ± {std_r2:.4f}")
        logger.info(f"  RMSE       = {mean_rmse:.4f} ± {std_rmse:.4f}")
        logger.info(f"  Per-fold ρ: {[round(r,4) for r in rhos]}")
        logger.info(f"{'='*60}")

        summary={
            'organ':organ,'n_folds':n_folds,'n_pairs':len(full_ds),
            'n_cell_types':n_types,
            'aggregate':{
                'spearman_rho_mean':round(float(mean_rho),4),
                'spearman_rho_std':round(float(std_rho),4),
                'pearson_r_mean':round(float(mean_pr),4),
                'pearson_r_std':round(float(std_pr),4),
                'r2_mean':round(float(mean_r2),4),
                'r2_std':round(float(std_r2),4),
                'rmse_mean':round(float(mean_rmse),4),
                'rmse_std':round(float(std_rmse),4),
                'quality':quality
            },
            'folds':all_fold_metrics
        }
        with open(f"{output_dir}/cv_summary.json",'w')as f:json.dump(summary,f,indent=2)
        logger.info(f"CV summary saved: {output_dir}/cv_summary.json")

        # ---- Generate visualizations using fold1 model ----
        try:
            from genecompass_inference import GeneCompassPredictor,CellChatVisualizer
            fold1_model=f"{output_dir}/fold1/best_model"
            if os.path.exists(fold1_model):
                logger.info("Generating 300dpi visualizations...")
                mat,cts=GeneCompassPredictor(
                    fold1_model,self.token_dict_path,self.proj_root
                ).analyze(dataset_path,output_dir,organ,True,cell_type_aggregated=True)
                gs=pd.read_csv(gs_path)
                CellChatVisualizer(organ).generate_all(mat,output_dir,cts,gs)
                logger.info("Visualizations done.")
        except Exception as e:
            logger.warning(f"Visualization skipped (non-critical): {e}")

        return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description='GeneCompass CCI 5-Fold CV')
    p.add_argument('--proj_root',required=True)
    p.add_argument('--gs_path',required=True)
    p.add_argument('--dataset',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--organ',default='Organ')
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--batch',type=int,default=1)
    p.add_argument('--grad_accum',type=int,default=4)
    p.add_argument('--lr',type=float,default=5e-5)
    p.add_argument('--folds',type=int,default=5)
    args=p.parse_args()
    cv=GeneCompassCV(args.proj_root)
    cv.cross_validate(args.gs_path,args.dataset,args.output,args.organ,
                      args.epochs,args.batch,args.grad_accum,args.lr,args.folds)
