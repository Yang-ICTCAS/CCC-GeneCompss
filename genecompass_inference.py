#!/usr/bin/env python3
"""
GeneCompass CCI Pipeline (Modular) — Inference & Visualization
"""
import sys,os,pickle,warnings,logging,argparse
import numpy as np,pandas as pd,torch
from datasets import load_from_disk
from tqdm import tqdm
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; import seaborn as sns
from matplotlib.patches import FancyArrowPatch,Circle,Arc,ConnectionStyle
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
logger=logging.getLogger(__name__)
DPI=300

class GeneCompassPredictor:
    """Module: Model loading & inference"""
    def __init__(self,model_path,token_dict_path,proj_root=None):
        self.model_path=model_path;self.token_dict_path=token_dict_path
        self.device=torch.device('cuda'if torch.cuda.is_available()else'cpu')
        if proj_root and proj_root not in sys.path:sys.path.insert(0,proj_root)
        self._load()
    def _load(self):
        from genecompass import BertForSequenceClassification
        from genecompass.utils import load_prior_embedding
        with open(self.token_dict_path,'rb')as f:self.token_dict=pickle.load(f)
        kw={}
        try:
            out=load_prior_embedding(token_dictionary_or_path=self.token_dict_path)
            kw=dict(zip(['promoter','co_exp','gene_family','peca_grn','homologous_gene_human2mouse'],out))
        except:pass
        logger.info(f"Loading model: {self.model_path}")
        self.model=BertForSequenceClassification.from_pretrained(self.model_path,knowledges=kw)
        self.model.to(self.device);self.model.eval()
        logger.info(f"Model on {self.device}")
    def _build_seq(self,ds,si,ri,ml=2048):
        ct,sp,pt=self.token_dict.get('<cls>',1),self.token_dict.get('<sep>',2),self.token_dict.get('<pad>',0)
        sd,rd=ds[si],ds[ri];i1=list(sd['input_ids']);v1=list(sd['values']);i2=list(rd['input_ids']);v2=list(rd['values'])
        av=ml-3;tot=len(i1)+len(i2)
        if tot>av:a1=max(1,int(av*len(i1)/tot));a2=max(1,av-a1)
        else:a1,a2=len(i1),len(i2)
        i1,v1=i1[:a1],v1[:a1];i2,v2=i2[:a2],v2[:a2]
        pi=[ct]+i1+[sp]+i2+[sp];pv=[0.0]+v1+[0.0]+v2+[0.0]
        # token_type_ids: 0=sender genes, 1=receiver genes (standard BERT sentence-pair convention)
        tt=[0]+[0]*len(i1)+[0]+[1]*len(i2)+[1]
        if len(pi)<ml:pi+=[pt]*(ml-len(pi));pv+=[0.0]*(ml-len(pv));tt+=[0]*(ml-len(tt))
        else:pi,pv,tt=pi[:ml],pv[:ml],tt[:ml]
        return{'input_ids':pi,'values':pv,'token_type_ids':tt}
    def analyze(self,dataset_path,output_dir,organ="Organ",include_self=True,bs=4,cell_type_aggregated=True):
        os.makedirs(output_dir,exist_ok=True)
        ds=load_from_disk(dataset_path)
        cts=sorted(set(ds['cell_type']));logger.info(f"Types: {len(cts)} (cell_type_aggregated={cell_type_aggregated})")
        # Map cell_type → dataset row index (direct for cell_type_aggregated, pool for single-cell)
        c2i={}
        for idx,ct in enumerate(ds['cell_type']):
            c2i.setdefault(str(ct),[]).append(int(idx))
        aps=[(s,r)for s in cts for r in cts if s!=r or include_self]
        logger.info(f"Pairs: {len(aps)} (self={include_self})")
        sc=[]
        with torch.no_grad():
            for i in tqdm(range(0,len(aps),bs),desc='Predict'):
                batch=[p for p in aps[i:i+bs]]
                if cell_type_aggregated:
                    # Deterministic: direct mapping (no random sampling)
                    bse=[self._build_seq(ds,c2i[str(s)][0],c2i[str(r)][0]) for s,r in batch if str(s) in c2i and str(r) in c2i]
                else:
                    bse=[self._build_seq(ds,int(np.random.choice(c2i[str(s)])),int(np.random.choice(c2i[str(r)]))) for s,r in batch if str(s) in c2i and str(r) in c2i]
                if not bse:continue
                ids=torch.tensor([s_['input_ids']for s_ in bse]).long().to(self.device)
                vs=torch.tensor([s_['values']for s_ in bse]).float().to(self.device)
                tt=torch.tensor([s_['token_type_ids']for s_ in bse]).long().to(self.device)
                sc.extend(self.model(input_ids=ids,values=vs,token_type_ids=tt).logits.cpu().numpy().flatten().tolist())
        df=pd.DataFrame({'Sender':[p[0]for p in aps],'Receiver':[p[1]for p in aps],'Score':sc[:len(aps)]})
        mat=df.pivot_table(index='Sender',columns='Receiver',values='Score',fill_value=0.)
        mat=mat.reindex(index=cts,columns=cts,fill_value=0.)
        mat.to_csv(os.path.join(output_dir,'interaction_matrix.csv'))
        df.to_csv(os.path.join(output_dir,'detailed_predictions.csv'),index=False)
        logger.info(f"Matrix: {mat.shape}")
        return mat,cts


class CellChatVisualizer:
    """Module: CellChat-professional visualization (300dpi, all interactions)"""
    def __init__(self,organ="Organ"):self.organ=organ
    def _save(self,fig,od,nm):p=os.path.join(od,nm);fig.savefig(p,dpi=DPI,bbox_inches='tight');plt.close(fig);logger.info(f"  {nm}")

    def generate_all(self,matrix,output_dir,cell_types=None,gs_df=None):
        """7 visualization types, CellChat-style"""
        if cell_types is None:cell_types=list(matrix.index)
        os.makedirs(output_dir,exist_ok=True)
        n=len(cell_types);mabs=np.abs(matrix.values).max()or 1

        # ── 1. Heatmap (CellChat style: diverging colormap, clean labels) ──
        fig,ax=plt.subplots(figsize=(max(16,n*0.8),max(14,n*0.7)))
        mask=np.eye(n,dtype=bool)
        sns.heatmap(matrix,cmap='RdBu_r',center=0,annot=True,fmt='.2f',linewidths=.5,
                    xticklabels=True,yticklabels=True,annot_kws={'size':8},mask=None,
                    cbar_kws={'label':'Interaction Score','shrink':0.8},ax=ax)
        ax.set_title(f'{self.organ} Cell-Cell Interaction Matrix',fontsize=16,fontweight='bold',pad=15)
        ax.set_xticklabels(ax.get_xticklabels(),rotation=45,ha='right',fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(),fontsize=9)
        self._save(fig,output_dir,'interaction_heatmap.png')

        # ── 2. Network (CellChat circular style: nodes on circle, precise curves, edge weights) ──
        fig,ax=plt.subplots(figsize=(max(18,n*1.5),max(18,n*1.5)))
        ax.set_xlim(-1.3,1.3);ax.set_ylim(-1.3,1.3);ax.set_aspect('equal');ax.axis('off')
        angles=np.linspace(0,2*np.pi,n+1)[:n];r=1.0
        pos={ct:(r*np.cos(a),r*np.sin(a))for ct,a in zip(cell_types,angles)}
        # Nodes
        ns=[800+1200*abs(matrix.loc[ct,ct])/mabs for ct in cell_types]
        for ct,sz,ang in zip(cell_types,ns,angles):
            color=plt.cm.RdBu_r(0.3+0.4*(1+matrix.loc[ct,ct]/mabs)if mabs else 0.5)
            circle=Circle(pos[ct],0.06,facecolor=color,edgecolor='black',linewidth=1.5,zorder=10)
            ax.add_patch(circle)
            lr=1.12;lx,lr_=(r+0.05)*np.cos(ang),(r+0.05)*np.sin(ang)
            ha='center';va='center'
            if ang<np.pi/2 or ang>3*np.pi/2:ha='left'
            elif np.pi/2<ang<3*np.pi/2:ha='right'
            if 0<ang<np.pi:va='bottom'
            else:va='top'
            ax.annotate(ct,pos[ct],textcoords='offset points',xytext=(15*np.cos(ang),15*np.sin(ang)),
                        fontsize=7,fontweight='bold',ha=ha,va=va,zorder=11)
        # Edges with curves (CellChat arc style)
        for si,sr in enumerate(cell_types):
            for ri,rc in enumerate(cell_types):
                sc=matrix.iloc[si,ri]
                if abs(sc)<0.003:continue
                color='#e74c3c'if sc>0 else'#3498db'
                lw=max(0.5,abs(sc)/mabs*4)
                alpha=min(0.6,0.15+abs(sc)/mabs*0.5)
                t1=angles[si];t2=angles[ri]
                if si==ri:  # Self loop
                    arc=Arc(pos[sr],0.15,0.15,angle=0,theta1=0,theta2=360,color=color,lw=lw,alpha=alpha,zorder=5)
                    ax.add_patch(arc)
                else:
                    rad=0.2 if abs(t1-t2)<np.pi else-0.2
                    arrow=FancyArrowPatch(pos[sr],pos[rc],connectionstyle=f'arc3,rad={rad}',
                                          arrowstyle='->',mutation_scale=12+abs(sc)/mabs*10,
                                          color=color,lw=lw,alpha=alpha,zorder=5)
                    ax.add_patch(arrow)
                # Edge label at midpoint
                mx,my=(pos[sr][0]+pos[rc][0])/2,(pos[sr][1]+pos[rc][1])/2
                rad_=0.2 if abs(t1-t2)<np.pi else-0.2
                dx,dy=pos[rc][0]-pos[sr][0],pos[rc][1]-pos[sr][1]
                mx+=rad_*(-dy)*0.3;my+=rad_*(dx)*0.3
                ax.text(mx,my,f'{sc:.2f}',fontsize=4,color='gray',ha='center',va='center',alpha=0.7,zorder=6)
        ax.set_title(f'{self.organ} Cell-Cell Interaction Network',fontsize=16,fontweight='bold',pad=15)
        self._save(fig,output_dir,'interaction_network.png')

        # ── 3. Bubble chart ──
        fig,ax=plt.subplots(figsize=(26,14))
        bbl=matrix.stack().reset_index();bbl.columns=['Sender','Receiver','Score']
        bbl=bbl[bbl['Score']!=0].copy();bbl['Abs']=np.abs(bbl['Score']);bbl['Label']=bbl['Sender']+' → '+bbl['Receiver']
        bbl=bbl.sort_values('Abs',ascending=False)
        clr=[('#e74c3c'if s>0 else'#3498db')for s in bbl['Score']]
        ax.scatter(range(len(bbl)),bbl['Score'],s=bbl['Abs']*80+15,c=clr,alpha=0.7,edgecolors='gray',linewidth=0.2)
        ax.axhline(y=0,color='gray',linestyle='--',alpha=0.4)
        if len(bbl)>0:
            st=max(1,len(bbl)//30)
            ax.set_xticks(range(0,len(bbl),st));ax.set_xticklabels([bbl['Label'].iloc[i]for i in range(0,len(bbl),st)],rotation=90,ha='center',fontsize=5)
        ax.set_ylabel('Interaction Score',fontsize=12);ax.set_xlabel('Cell Type Pair',fontsize=12)
        ax.set_title(f'{self.organ} Cell-Cell Interactions ({len(bbl)} pairs)',fontsize=16,fontweight='bold')
        self._save(fig,output_dir,'interaction_bubble.png')

        # ── 4. Flow ──
        fig,axes=plt.subplots(1,2,figsize=(22,12))
        og=matrix.sum(axis=1).sort_values(ascending=False)
        axes[0].barh(range(len(og)),og.values,color='steelblue',edgecolor='black',lw=.5)
        axes[0].set_yticks(range(len(og)));axes[0].set_yticklabels(og.index,size=9);axes[0].invert_yaxis()
        axes[0].set_title('Total Outgoing Strength (Sender)',fontsize=13);axes[0].set_xlabel('Cumulative Score')
        ig=matrix.sum(axis=0).sort_values(ascending=False)
        axes[1].barh(range(len(ig)),ig.values,color='coral',edgecolor='black',lw=.5)
        axes[1].set_yticks(range(len(ig)));axes[1].set_yticklabels(ig.index,size=9);axes[1].invert_yaxis()
        axes[1].set_title('Total Incoming Strength (Receiver)',fontsize=13);axes[1].set_xlabel('Cumulative Score')
        self._save(fig,output_dir,'interaction_flow.png')

        # ── 5. Autocrine ──
        fig,ax=plt.subplots(figsize=(max(14,n*0.7),7))
        scd={ct:matrix.loc[ct,ct]for ct in cell_types}
        cl_=list(scd.keys());vl_=list(scd.values())
        cb_=['#2ecc71'if v>0 else'#e74c3c'for v in vl_]
        ax.bar(cl_,vl_,color=cb_,alpha=.8,edgecolor='black',lw=1)
        ax.axhline(y=0,color='gray',ls='--',lw=1.5)
        ax.set_xticks(range(len(cl_)));ax.set_xticklabels(cl_,rotation=45,ha='right',size=9)
        for i,(ct,v)in enumerate(zip(cl_,vl_)):ax.text(i,v+0.005 if v>=0 else v-0.02,f'{v:.2f}',ha='center',size=8)
        ax.set_ylabel('Autocrine Score',fontsize=12);ax.set_title(f'{self.organ} Autocrine (Self) Interaction',fontsize=16,fontweight='bold')
        self._save(fig,output_dir,'autocrine_scores.png')

        # ── 6. Circular chord diagram ──
        fig,ax=plt.subplots(figsize=(max(18,n*1.5),max(18,n*1.5)))
        ax.set_xlim(-1.35,1.35);ax.set_ylim(-1.35,1.35);ax.set_aspect('equal');ax.axis('off')
        angles=np.linspace(0,2*np.pi,n+1)[:n];r=1.0
        pos2={ct:(r*np.cos(a),r*np.sin(a))for ct,a in zip(cell_types,angles)}
        for ct,sz,ang in zip(cell_types,ns,angles):
            color=plt.cm.RdBu_r(0.3+0.4*(1+matrix.loc[ct,ct]/mabs)if mabs else 0.5)
            circle=Circle(pos2[ct],0.04,facecolor=color,edgecolor='black',linewidth=1,zorder=10)
            ax.add_patch(circle)
            ax.annotate(ct,pos2[ct],textcoords='offset points',xytext=(10*np.cos(ang),10*np.sin(ang)),
                        fontsize=5,fontweight='bold',ha='center',va='center',zorder=11)
        for si,sr in enumerate(cell_types):
            for ri,rc in enumerate(cell_types):
                sc=matrix.iloc[si,ri]
                if abs(sc)<0.003:continue
                color='#e74c3c'if sc>0 else'#3498db'
                lw=max(0.3,abs(sc)/mabs*3)
                alpha=min(0.4,0.1+abs(sc)/mabs*0.3)
                t1,t2=angles[si],angles[ri]
                rad=0.25 if abs(t1-t2)<np.pi else-0.25
                if si!=ri:
                    arrow=FancyArrowPatch(pos2[sr],pos2[rc],connectionstyle=f'arc3,rad={rad}',
                                          arrowstyle='->',mutation_scale=8,lw=lw,color=color,alpha=alpha,zorder=5)
                    ax.add_patch(arrow)
        ax.set_title(f'{self.organ} Circular Interaction Network',fontsize=16,fontweight='bold',pad=15)
        self._save(fig,output_dir,'interaction_circular.png')

        # ── 7. GS comparison ──
        if gs_df is not None and 'Consensus_Score'in gs_df.columns:
            fig,ax=plt.subplots(figsize=(10,8))
            gs_=gs_df.copy()
            gs_['pair']=gs_['Sender']+'_'+gs_['Receiver']
            df_=pd.read_csv(os.path.join(output_dir,'detailed_predictions.csv'))
            df_['pair']=df_['Sender']+'_'+df_['Receiver']
            mrg=gs_[['pair','Consensus_Score']].merge(df_[['pair','Score']].rename(columns={'Score':'Pred'}),on='pair',how='inner')
            if len(mrg)>0:
                s=MinMaxScaler();tv=s.fit_transform(mrg[['Consensus_Score']]).ravel();pv=s.fit_transform(mrg[['Pred']]).ravel()
                pr=np.corrcoef(tv,pv)[0,1]
                ax.scatter(tv,pv,alpha=0.4,s=25,c='steelblue',edgecolors='white',linewidth=0.3)
                ax.plot([tv.min(),tv.max()],[tv.min(),tv.max()],'r--',lw=2,label='Perfect')
                ax.set_xlabel('Gold Standard Consensus',fontsize=12);ax.set_ylabel('GeneCompass Predicted',fontsize=12)
                ax.set_title(f'{self.organ}: GeneCompass vs Gold Standard\nPearson={pr:.4f}',fontsize=14,fontweight='bold')
                ax.legend()
            self._save(fig,output_dir,'true_vs_predicted.png')


def main():
    p=argparse.ArgumentParser(description='GeneCompass CCI Modular Pipeline')
    p.add_argument('--model',required=True,help='Model path')
    p.add_argument('--dataset',required=True,help='Dataset path')
    p.add_argument('--token_dict',required=True,help='Token dict .pickle')
    p.add_argument('--output',required=True,help='Output dir')
    p.add_argument('--organ',default='Organ',help='Organ name')
    p.add_argument('--gold_standard',default=None,help='GS CSV')
    p.add_argument('--proj_root',default=None,help='Project root')
    args=p.parse_args()
    pred=GeneCompassPredictor(args.model,args.token_dict,args.proj_root)
    mat,cts=pred.analyze(args.dataset,args.output,args.organ)
    gs_df=pd.read_csv(args.gold_standard)if args.gold_standard else None
    viz=CellChatVisualizer(args.organ)
    viz.generate_all(mat,args.output,cts,gs_df)
    logger.info(f"\n{'='*60}\n  {args.organ} complete!\n{'='*60}")

if __name__=='__main__':main()
