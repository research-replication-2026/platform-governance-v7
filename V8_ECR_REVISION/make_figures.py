"""Create journal figures and exact plotted-data CSVs, without generative imagery."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator
B=Path(__file__).resolve().parent;F=B/'FIGURES';A=B/'ANALYSIS'
SCENARIOS=['balanced_reference','high_security_pressure','high_review_capacity','constrained_review_capacity','high_false_restriction_cost','participation_sensitive_ecosystem']
LABELS=['Reference','Security pressure','Review capacity +','Review capacity −','False restriction cost','Participation sensitive']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})

def save(fig,number):
    fig.savefig(F/f'Fig{number}.png',dpi=180,facecolor='white')
    for suffix in ['pdf','svg','eps']: fig.savefig(F/f'Fig{number}.{suffix}',facecolor='white')
    fig.savefig(F/f'Fig{number}.tif',dpi=600,facecolor='white',pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)

def intervals(ax,g,labels,panel):
    y=np.arange(len(g))
    ax.hlines(y,g.simultaneous_low,g.simultaneous_high,color='#aebfc8',lw=5,label='Simultaneous 95% CI')
    ax.hlines(y,g.ci95_low,g.ci95_high,color='#155770',lw=1.8,label='Pointwise 95% CI')
    ax.scatter(g['mean'],y,s=22,color='#102f42',zorder=3)
    ax.axvline(0,color='0.4',ls='--',lw=.8);ax.set_yticks(y,labels);ax.invert_yaxis()
    lo=min(0,g.simultaneous_low.min());hi=max(0,g.simultaneous_high.max());padding=max(.003,(hi-lo)*.08)
    ax.set_xlim(lo-padding,hi+padding);ax.xaxis.set_major_locator(MaxNLocator(4));ax.grid(axis='x',color='.9')
    ax.text(-.02,1.08,panel,transform=ax.transAxes,fontweight='bold',fontsize=11)

def main():
    F.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(6.85,4.8));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    boxes=[(.04,.76,.92,.18,'Public documentary sources\n50 sources → 32 retained governance events'),(.04,.48,.92,.18,'Stylized decision environment\nDocumented mechanisms; synthetic probabilities and utilities'),(.04,.16,.285,.20,'A: policy robustness\n30 training blocks\n6 scenarios × 2 settings'),(.357,.16,.285,.20,'B: reward reports\n0%, 25%, 50% loss\n3 sequential scenarios'),(.67,.16,.285,.20,'C: observations\nOriginal / reduced / enriched\n3 sequential scenarios')]
    for x,y,w,h,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.01',fc='#edf3f6',ec='#36596c',lw=1))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=9)
    ax.annotate('',xy=(.5,.67),xytext=(.5,.75),arrowprops={'arrowstyle':'->','color':'#36596c'})
    for x in [.18,.5,.82]: ax.annotate('',xy=(x,.37),xytext=(.5,.47),arrowprops={'arrowstyle':'->','color':'#36596c'})
    ax.text(.5,.065,'Common legal-action masks • Full-reward evaluation • Training-block inference',ha='center',fontsize=9)
    fig.subplots_adjust(left=.015,right=.985,top=.99,bottom=.01);save(fig,1)
    pd.DataFrame([{'x':x,'y':y,'width':w,'height':h,'label':t} for x,y,w,h,t in boxes]).to_csv(F/'Fig1_source_data.csv',index=False)
    c=pd.read_csv(A/'paired_policy_contrasts.csv')
    g=c[(c.condition=='A_corrected')&(c.metric=='mean_reward')&(c.comparison=='q_learning_minus_contextual_bandit')]
    g.to_csv(F/'Fig2_source_data.csv',index=False)
    fig,axes=plt.subplots(2,1,figsize=(6.85,6.6))
    for ax,env,panel in zip(axes,['nonsequential','sequential'],['a','b']):
        z=g[g.environment==env].set_index('scenario').loc[SCENARIOS].reset_index();intervals(ax,z,LABELS,panel)
    axes[-1].set_xlabel('Q-learning − contextual bandit: reward per decision')
    handles,names=axes[0].get_legend_handles_labels();fig.legend(handles,names,loc='lower center',frameon=False,ncol=1)
    fig.subplots_adjust(left=.34,right=.98,bottom=.17,top=.94,hspace=.42);save(fig,2)
    i=pd.read_csv(A/'condition_interactions.csv');g=i[i.metric=='mean_reward'];g.to_csv(F/'Fig3_source_data.csv',index=False)
    fig,axes=plt.subplots(2,1,figsize=(6.85,6.6))
    for ax,conditions,panel in zip(axes,[['B_loss25','B_loss50'],['C_reduced','C_enriched']],['a','b']):
        rows=[];labels=[]
        for condition in conditions:
            for scenario,short in zip([SCENARIOS[0],SCENARIOS[1],SCENARIOS[3]],['reference','security','capacity']):
                rows.append(g[(g.condition==condition)&(g.scenario==scenario)].iloc[0]);labels.append({'B_loss25':'25% loss','B_loss50':'50% loss','C_reduced':'Reduced','C_enriched':'Enriched'}[condition]+': '+short)
        intervals(ax,pd.DataFrame(rows),labels,panel)
    axes[-1].set_xlabel('Change in Q-learning − bandit reward gap')
    handles,names=axes[0].get_legend_handles_labels();fig.legend(handles,names,loc='lower center',frameon=False)
    fig.subplots_adjust(left=.32,right=.98,bottom=.17,top=.94,hspace=.42);save(fig,3)
    g=pd.read_csv(A/'budget_policy_gaps.csv');g=g[g.comparison=='q_learning_minus_contextual_bandit'];g.to_csv(F/'Fig4_source_data.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(6.85,3.7),sharey=True)
    for ax,scenario,panel in zip(axes,SCENARIOS[:2],['a','b']):
        z=g[g.scenario==scenario].sort_values('budget')
        ax.errorbar(z.budget,z['mean'],yerr=[z['mean']-z.ci95_low,z.ci95_high-z['mean']],fmt='o-',capsize=3,lw=1.3,color='#155770')
        ax.axhline(0,color='.4',ls='--',lw=.8);ax.set_xticks([180,360,720]);ax.set_xlabel('Training episodes');ax.text(.03,.95,panel,transform=ax.transAxes,fontweight='bold',fontsize=11);ax.grid(axis='y',color='.9')
    axes[0].set_ylabel('Q-learning − bandit reward gap');fig.subplots_adjust(left=.15,right=.98,bottom=.2,top=.97,wspace=.12);save(fig,4)
    (F/'README.md').write_text('# Figures\n\nFig1: design; Fig2: corrected main reward contrasts (a immediate, b delayed); Fig3: interactions (a feedback, b observations); Fig4: exploratory nested budgets (a reference, b security pressure). Captions are in the manuscript. Exact plotted estimates are in the corresponding source_data.csv. EPS/PDF/SVG are vectors with embedded or editable fonts; TIFFs are losslessly compressed at 600 dpi. Font sizes are 9–11 pt at 174-mm width. No generative image model was used. All plots are scripted from executed simulation data.\n',encoding='utf-8')

if __name__=='__main__': main()
