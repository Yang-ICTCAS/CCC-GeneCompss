#!/usr/bin/env python3
"""
GeneCompass CCI Pipeline — Gold Standard Builder
=================================================
Weights learned from source significance: w ∝ mean(N_SigLR)
CellChat + CellPhoneDB v5 consensus → ranked interaction pairs.
"""
import os,sys,json,warnings,logging,pickle,glob
import numpy as np,pandas as pd
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
logger=logging.getLogger(__name__)

class GoldStandardBuilder:

    def __init__(self,cellchat_dir,cpdb_dir):
        self.cc_dir=cellchat_dir;self.cpdb_dir=cpdb_dir
        self.weights=None  # learned during build()
        self.scaler=MinMaxScaler()

    def _load_cellchat_scores(self):
        cc=pd.read_csv(f"{self.cc_dir}/cell_interaction_strength_matrix.csv")
        cc_l=cc.melt(id_vars='Sender',value_vars=[c for c in cc.columns if c!='Sender'],
                     var_name='Receiver',value_name='CellChat_Score')
        cc_l['CellChat_Score']=pd.to_numeric(cc_l['CellChat_Score'],errors='coerce').fillna(0)
        return cc_l

    def _load_cellchat_significance(self):
        comm_path=f"{self.cc_dir}/cellchat_communication.csv"
        if not os.path.exists(comm_path):return None
        comm=pd.read_csv(comm_path)
        sig=comm[comm['prob']>0].groupby(['source','target']).size().reset_index(name='N_Significant_LR')
        sig.columns=['Sender','Receiver','N_CC_SigLR']
        return sig

    def _load_cpdb_scores(self):
        cp=pd.read_csv(f"{self.cpdb_dir}/significant_means.txt",sep='\t')
        non_p=['id_cp_interaction','interacting_pair','gene_a','gene_b','partner_a','partner_b']
        pcols=[c for c in cp.columns if c not in non_p]or[c for c in cp.columns if'|'in c]
        cp_l=pd.melt(cp,id_vars=['gene_a','gene_b']if'gene_a'in cp.columns else non_p,
                     value_vars=pcols,var_name='CellPair',value_name='CPDB_Score')
        cp_l['CPDB_Score']=pd.to_numeric(cp_l['CPDB_Score'],errors='coerce')
        cp_l=cp_l.dropna(subset=['CPDB_Score']);cp_l=cp_l[cp_l['CPDB_Score']>0]
        cp_l[['Sender','Receiver']]=cp_l['CellPair'].str.split(r'\|',expand=True)
        return cp_l.groupby(['Sender','Receiver']).agg(
            CPDB_Mean=('CPDB_Score','mean'),CPDB_Max=('CPDB_Score','max'),
            CPDB_Num=('CPDB_Score','count')).reset_index()

    def _load_cpdb_significance(self):
        pv_files=glob.glob(f"{self.cpdb_dir}/statistical_analysis_pvalues_*.txt")
        if not pv_files:
            pv_files=glob.glob(f"{self.cpdb_dir}/significant_means.txt")
            if not pv_files:return None
        pv_path=pv_files[0]
        pv=pd.read_csv(pv_path,sep='\t')
        non_p=['id_cp_interaction','interacting_pair','partner_a','partner_b',
               'gene_a','gene_b','secreted','receptor_a','receptor_b','annotation_strategy']
        pcols=[c for c in pv.columns if c not in non_p]
        if not pcols:pcols=[c for c in pv.columns if'|'in c]
        if not pcols:
            logger.warning("No cell-pair columns in CPDB p-values; using significant_means as proxy")
            return self._load_cpdb_scores()[['Sender','Receiver','CPDB_Num']].rename(
                columns={'CPDB_Num':'N_CPDB_SigLR'})
        pv_l=pd.melt(pv,id_vars=non_p if all(x in pv.columns for x in non_p) else ['gene_a','gene_b'],
                     value_vars=pcols,var_name='CellPair',value_name='pvalue')
        pv_l['pvalue']=pd.to_numeric(pv_l['pvalue'],errors='coerce')
        sig=pv_l[pv_l['pvalue']<0.05].groupby('CellPair').size().reset_index(name='N_Sig_LR')
        sig[['Sender','Receiver']]=sig['CellPair'].str.split(r'\|',expand=True)
        return sig.groupby(['Sender','Receiver'])['N_Sig_LR'].sum().reset_index().rename(
            columns={'N_Sig_LR':'N_CPDB_SigLR'})

    def _learn_weights(self,combined):
        """
        Learn weights from source significance. w ∝ mean(N_SigLR).
        No GeneCompass involvement — no circularity.
        """
        cc_conf=combined['N_CC_SigLR'].mean()
        cpdb_conf=combined['N_CPDB_SigLR'].mean()
        if cc_conf<=0 and cpdb_conf<=0:
            self.weights={'CellChat':1/3,'CPDB_Mean':1/3,'CPDB_Max':1/3}
        else:
            w_cc=cc_conf if cc_conf>0 else 1e-6
            w_cpm=cpdb_conf if cpdb_conf>0 else 1e-6
            total=w_cc+w_cpm+w_cpm  # CPDB_Mean and CPDB_Max share same confidence
            self.weights={
                'CellChat':round(float(w_cc)/total,4),
                'CPDB_Mean':round(float(w_cpm)/total,4),
                'CPDB_Max':round(float(w_cpm)/total,4)
            }
        logger.info(f"Weights: {self.weights} (CC sig={cc_conf:.1f} LR/pair, CPDB sig={cpdb_conf:.1f} LR/pair)")

    def build(self,output_dir,include_self=True,threshold_quantile=0.7):
        os.makedirs(output_dir,exist_ok=True)
        cc_l=self._load_cellchat_scores()
        cpdb_a=self._load_cpdb_scores()
        cc_sig=self._load_cellchat_significance()
        cpdb_sig=self._load_cpdb_significance()

        types=sorted(set(cc_l['Sender'].unique())|set(cc_l['Receiver'].unique()))
        aps=pd.DataFrame([(s,r)for s in types for r in types if s!=r or include_self],
                          columns=['Sender','Receiver'])

        combined=aps.merge(cc_l,on=['Sender','Receiver'],how='left').merge(cpdb_a,on=['Sender','Receiver'],how='left')
        for c in['CellChat_Score','CPDB_Mean','CPDB_Max']:combined[c]=combined[c].fillna(0)

        if cc_sig is not None:
            combined=combined.merge(cc_sig,on=['Sender','Receiver'],how='left')
            combined['N_CC_SigLR']=combined['N_CC_SigLR'].fillna(0).astype(int)
        else:combined['N_CC_SigLR']=0
        if cpdb_sig is not None:
            combined=combined.merge(cpdb_sig,on=['Sender','Receiver'],how='left')
            combined['N_CPDB_SigLR']=combined['N_CPDB_SigLR'].fillna(0).astype(int)
        else:combined['N_CPDB_SigLR']=0

        # Significance flags (informational)
        combined['CC_Significant']=(combined['N_CC_SigLR']>=1).astype(int)
        combined['CPDB_Significant']=(combined['N_CPDB_SigLR']>=1).astype(int)

        # Learn weights from source significance
        self._learn_weights(combined)

        # Consensus Score
        combined['N_CC']=self.scaler.fit_transform(combined[['CellChat_Score']]).ravel()
        combined['N_CPM']=self.scaler.fit_transform(combined[['CPDB_Mean']]).ravel()
        combined['N_CPH']=self.scaler.fit_transform(combined[['CPDB_Max']]).ravel()
        tw=sum(self.weights.values())
        combined['Consensus_Score']=(self.weights['CellChat']*combined['N_CC']+
            self.weights['CPDB_Mean']*combined['N_CPM']+self.weights['CPDB_Max']*combined['N_CPH'])/tw
        combined['Consensus_Score']=self.scaler.fit_transform(combined[['Consensus_Score']]).ravel()

        th=combined['Consensus_Score'].quantile(threshold_quantile)
        combined['Gold_Standard_Label']=(combined['Consensus_Score']>=th).astype(int)
        combined['Threshold_Quantile']=threshold_quantile
        combined['Threshold_Value']=th

        combined['Pair_ID']=combined['Sender']+'_'+combined['Receiver']
        pos=int(combined['Gold_Standard_Label'].sum())
        n_self=int((combined['Sender']==combined['Receiver']).sum())
        logger.info(f"GS: {len(combined)} pairs, {pos} positive ({pos/len(combined)*100:.1f}%), {n_self} self")

        combined.to_csv(os.path.join(output_dir,'complete_labeled_interactions.csv'),index=False)
        combined[combined['Gold_Standard_Label']==1].to_csv(
            os.path.join(output_dir,'gold_standard_interactions.csv'),index=False)

        stats={
            'total_pairs':len(combined),'positive_pairs':pos,
            'positive_pct':round(pos/len(combined)*100,1),
            'self_interactions':n_self,'cell_types':len(types),
            'weights':{k:round(float(v),4) for k,v in self.weights.items()},
            'cc_avg_sig_lr':round(float(combined['N_CC_SigLR'].mean()),1),
            'cpdb_avg_sig_lr':round(float(combined['N_CPDB_SigLR'].mean()),1),
        }
        json.dump(stats,open(os.path.join(output_dir,'gold_standard_stats.json'),'w'),indent=2)
        return combined


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description='Gold Standard Builder')
    p.add_argument('--cellchat',required=True,help='CellChat output dir')
    p.add_argument('--cpdb',required=True,help='CellPhoneDB output dir')
    p.add_argument('--output',required=True,help='Gold standard output dir')
    args=p.parse_args()
    builder=GoldStandardBuilder(args.cellchat,args.cpdb)
    builder.build(args.output)
