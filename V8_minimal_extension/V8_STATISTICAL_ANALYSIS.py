"""Training-cluster inference, paired condition interactions and publication plots.

Run from any directory: python -B V8_minimal_extension/V8_STATISTICAL_ANALYSIS.py
No experiment is run by this script. Refuses incomplete confirmatory datasets.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import json,math,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_v8 import LEARNED,RULES
SEEDS=list(range(10001,10031)) # amendment 01: disjoint paired training blocks

BASE=Path(__file__).resolve().parent
OUT=BASE/'analysis'
FIG=BASE/'V8_FIGURES'
METRICS=['mean_reward','legitimate_contributions_accepted','security_incidents',
         'rejected_legitimate_innovation','human_review_workload','governance_expenditure',
         'safety_guard_interventions','harmful_submissions_released','developer_trust',
         'developer_retention','unseen_eval_state_rate']

def interval(x,alpha=.05):
    x=np.asarray(x,dtype=float)
    n=len(x);mean=float(x.mean());sd=float(x.std(ddof=1)) if n>1 else math.nan
    se=sd/math.sqrt(n)
    radius=float(stats.t.ppf(1-alpha/2,n-1))*se if n>1 else math.nan
    if sd>0:
        statistic=mean/se
        p=float(2*stats.t.sf(abs(statistic),n-1))
    else: p=1. if mean==0 else 0.
    return dict(n_training=n,mean=mean,sd_training=sd,ci95_low=mean-radius,
                ci95_high=mean+radius,p_raw=p,dz=mean/sd if sd>0 else math.nan)

def holm(p):
    p=np.asarray(p,dtype=float)
    order=np.argsort(p);ordered=p[order]
    adjusted=np.minimum(1,np.maximum.accumulate(ordered*(len(p)-np.arange(len(p)))))
    result=np.empty_like(adjusted);result[order]=adjusted
    return result

def crossed_interval(matrix,seed=819731,draws=5000):
    rng=np.random.default_rng(seed)
    n,k=matrix.shape
    ri=rng.integers(n,size=(draws,n));ci=rng.integers(k,size=(draws,k))
    values=matrix[ri[:,:,None],ci[:,None,:]].mean(axis=(1,2))
    return np.quantile(values,[.025,.975])

def matrix(frame,policy,metric):
    data=frame[frame.policy==policy]
    evaluations=sorted(frame.eval_seed.unique())
    if policy in RULES:
        assert set(data.train_seed)=={0}
        a=data.set_index('eval_seed').reindex(evaluations)[metric].to_numpy(dtype=float)
        assert len(data)==10 and not np.isnan(a).any()
        return np.broadcast_to(a,(30,10)).copy()
    table=data.pivot(index='train_seed',columns='eval_seed',values=metric).reindex(index=SEEDS,columns=evaluations)
    assert table.shape==(30,10) and not table.isna().any().any(),(policy,metric,'Incomplete training/evaluation panel')
    return table.to_numpy(dtype=float)

def analyze():
    OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
    status=json.loads((BASE/'V8_RUN_STATUS.json').read_text())
    assert status['complete_jobs']==status['planned_jobs'],'Campaign incomplete; no confirmatory analysis generated'
    frame=pd.read_csv(BASE/'V8_RESULTS.csv')
    assert not frame.duplicated(['condition','environment','scenario','policy','train_seed','eval_seed']).any()
    summaries=[];contrasts=[];training=[];stored={}
    comparisons=[('q_learning','contextual_bandit'),('linear_q_approx','contextual_bandit')]
    comparisons += [(p,b) for p in ['q_learning','contextual_bandit'] for b in RULES]
    for keys,group in frame.groupby(['condition','environment','scenario'],sort=True):
        meta=dict(zip(['condition','environment','scenario'],keys))
        stored[keys]=group
        for policy in LEARNED+RULES:
            for metric in METRICS:
                a=matrix(group,policy,metric)
                if policy in RULES:
                    summary=dict(n_training=0,mean=float(a[0].mean()),sd_training=math.nan,
                                 ci95_low=math.nan,ci95_high=math.nan,p_raw=math.nan,dz=math.nan,
                                 sd_evaluation=float(a[0].std(ddof=1)),inference='descriptive_fixed_rule')
                else:
                    means=a.mean(axis=1)
                    summary=interval(means)
                    summary.update(sd_evaluation=float(a.std(ddof=1)),inference='training_clusters_conditional_on_eval_panel')
                    training.extend(dict(**meta,policy=policy,metric=metric,train_seed=s,value=float(x)) for s,x in zip(SEEDS,means))
                summaries.append(dict(**meta,policy=policy,metric=metric,**summary))
        for p,b in comparisons:
            for metric in METRICS:
                differences=matrix(group,p,metric)-matrix(group,b,metric)
                row=dict(**meta,comparison=f'{p}_minus_{b}',metric=metric,**interval(differences.mean(axis=1)),
                         family='',p_holm=math.nan,simultaneous_low=math.nan,simultaneous_high=math.nan,
                         crossed_bootstrap_low=math.nan,crossed_bootstrap_high=math.nan)
                if p=='q_learning' and b=='contextual_bandit' and metric=='mean_reward':
                    if keys[0]=='A_v7_exact': row['family']='RQ1_legacy_12'
                    elif keys[0]=='A_corrected': row['family']='RQ1_corrected_12'
                    low,high=crossed_interval(differences)
                    row.update(crossed_bootstrap_low=low,crossed_bootstrap_high=high)
                contrasts.append(row)
    interactions=[]
    for keys,g in stored.items():
        condition,environment,scenario=keys
        if condition not in ['B_loss25','B_loss50','C_reduced','C_enriched']: continue
        control=stored[('A_corrected',environment,scenario)]
        for metric in METRICS:
            d=(matrix(g,'q_learning',metric)-matrix(g,'contextual_bandit',metric)-
               matrix(control,'q_learning',metric)+matrix(control,'contextual_bandit',metric))
            low,high=crossed_interval(d) if metric=='mean_reward' else (math.nan,math.nan)
            interactions.append(dict(condition=condition,environment=environment,scenario=scenario,
                comparison='change_in_Q_minus_bandit_vs_A_corrected',metric=metric,**interval(d.mean(axis=1)),
                family=('RQ2_feedback_6' if condition.startswith('B') else 'RQ3_observability_6') if metric=='mean_reward' else '',
                p_holm=math.nan,simultaneous_low=math.nan,simultaneous_high=math.nan,
                crossed_bootstrap_low=low,crossed_bootstrap_high=high))
    c=pd.DataFrame(contrasts);inter=pd.DataFrame(interactions)
    for table in [c,inter]:
        for family,g in table[table.family.ne('')].groupby('family'):
            count=len(g)
            assert count==(12 if family.startswith('RQ1') else 6)
            table.loc[g.index,'p_holm']=holm(g.p_raw)
            radius=stats.t.ppf(1-.05/(2*count),29)*g.sd_training/np.sqrt(30)
            table.loc[g.index,'simultaneous_low']=g['mean']-radius
            table.loc[g.index,'simultaneous_high']=g['mean']+radius
    summary=pd.DataFrame(summaries)
    summary.to_csv(OUT/'policy_levels.csv',index=False)
    c.to_csv(OUT/'paired_policy_contrasts.csv',index=False)
    inter.to_csv(OUT/'condition_interactions.csv',index=False)
    pd.DataFrame(training).to_csv(OUT/'training_unit_metrics.csv',index=False)
    # Difference between implementation strata is diagnostic and not attributable
    # to either individual code correction.
    implementation=[]
    for keys,g in stored.items():
        if keys[0]!='A_corrected': continue
        legacy=stored[('A_v7_exact',keys[1],keys[2])]
        for metric in METRICS:
            d=matrix(g,'q_learning',metric)-matrix(g,'contextual_bandit',metric)-matrix(legacy,'q_learning',metric)+matrix(legacy,'contextual_bandit',metric)
            implementation.append(dict(environment=keys[1],scenario=keys[2],metric=metric,**interval(d.mean(axis=1))))
    pd.DataFrame(implementation).to_csv(OUT/'implementation_sensitivity.csv',index=False)
    checks=dict(result_rows=len(frame),unique_training_labels=len(frame[frame.policy.isin(LEARNED)].train_seed.unique()),
                learned_cells=len(frame[frame.policy.isin(LEARNED)].groupby(['condition','environment','scenario','policy'])),
                independent_runs_per_learned_cell=30,evaluations_per_run=10,
                corrected_residual_pending_items_max=int(frame[frame.implementation=='v8_corrected'].residual_pending_items.max()),
                reward_additivity_error=float(np.abs(frame.total_reward-frame[[
                    'current_action_reward_total','delayed_past_reward_total','collective_state_reward_total',
                    'terminal_liquidation_reward_total','guard_penalty_total']].sum(axis=1)).max()),
                fixed_rule_duplicate_training_labels=False,
                raw_results_sha256=hashlib.sha256((BASE/'V8_RESULTS.csv').read_bytes()).hexdigest())
    assert checks['corrected_residual_pending_items_max']==0
    assert checks['reward_additivity_error']<1e-9
    (OUT/'analysis_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    figures(c,inter,summary)
    write_summary(c,inter,summary,checks)
    print(json.dumps(checks,indent=2))

def table_md(table,columns):
    # Avoid an optional tabulate dependency.
    lines=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
    for _,r in table.iterrows():
        lines.append('| '+' | '.join(str(r[x]) for x in columns)+' |')
    return '\n'.join(lines)

def format_effects(table):
    table=table.copy()
    table['difference [95% t CI]']=[f'{r["mean"]:.4f} [{r.ci95_low:.4f}, {r.ci95_high:.4f}]' for _,r in table.iterrows()]
    table['simultaneous 95% family CI']=[f'[{r.simultaneous_low:.4f}, {r.simultaneous_high:.4f}]' for _,r in table.iterrows()]
    table['Holm p']=table.p_holm.map(lambda p:f'{p:.4g}')
    table['paired dz']=table.dz.map(lambda x:f'{x:.3f}')
    table['crossed bootstrap 95%']=[f'[{r.crossed_bootstrap_low:.4f}, {r.crossed_bootstrap_high:.4f}]' for _,r in table.iterrows()]
    return table

def write_summary(c,inter,levels,checks):
    parts=['# V8 results — completed and numerically verified',
           f'Analyzed {checks["result_rows"]:,} primary evaluation rows. Every learned cell has 30 independent training runs (labels 10001..10030, amendment 01) and 10 fixed evaluation seeds per run. The three historical groups are retained separately because their cross-algorithm episode streams overlap across groups. Four deterministic rules have ten evaluations and no training replicates. Intervals quantify training variation conditional on this evaluation panel; crossed bootstrap intervals additionally assess evaluation-panel sensitivity.',
           'The exact V7 arm and corrected arm are separate. B1/C1 reuse the corrected control. All signs below are Q minus bandit, or changes in that difference. A positive interaction means a relative improvement of Q; it need not imply that Q exceeds the bandit.']
    parts.append('The primary reward disadvantage persists in all twelve main cells, both before and after the implementation corrections. Neither the feedback-loss family nor the observability family establishes a change in that policy gap after the prespecified multiplicity adjustment. Thus robustness is strengthened, while the proposed feedback/observability explanations remain unconfirmed under these interventions. This is not evidence of equivalence or absence of all such mechanisms.')
    counts={}
    for condition in ['A_v7_exact','A_corrected']:
        g=c[(c.condition==condition)&(c.comparison=='q_learning_minus_contextual_bandit')&(c.metric=='mean_reward')]
        counts[condition]=dict(negative_point_means=int((g['mean']<0).sum()),positive_point_means=int((g['mean']>0).sum()),
                               negative_holm=int(((g['mean']<0)&(g.p_holm<.05)).sum()),positive_holm=int(((g['mean']>0)&(g.p_holm<.05)).sum()),
                               negative_crossed_ci=int((g.crossed_bootstrap_high<0).sum()),positive_crossed_ci=int((g.crossed_bootstrap_low>0).sum()))
        parts += [f'## RQ1 — {condition}',f'{counts[condition]["negative_point_means"]}/12 negative mean contrasts; {counts[condition]["negative_holm"]}/12 negative Holm-adjusted results; {counts[condition]["positive_holm"]}/12 positive adjusted results. All {counts[condition]["negative_crossed_ci"]} negative crossed-bootstrap intervals are pointwise sensitivity results, not a separate simultaneous family test.',
                  table_md(format_effects(g),['environment','scenario','difference [95% t CI]','simultaneous 95% family CI','Holm p','paired dz','crossed bootstrap 95%'])]
    for prefix,title in [('B','RQ2 — reward-feedback loss'),('C','RQ3 — state observability')]:
        g=inter[inter.condition.str.startswith(prefix)&(inter.metric=='mean_reward')]
        counts[prefix]=dict(relative_improvements_holm=int(((g['mean']>0)&(g.p_holm<.05)).sum()),
                            relative_deteriorations_holm=int(((g['mean']<0)&(g.p_holm<.05)).sum()),
                            total_comparisons=len(g))
        parts += [f'## {title}',f'{counts[prefix]["relative_improvements_holm"]} relative improvements and {counts[prefix]["relative_deteriorations_holm"]} relative deteriorations established after Holm adjustment across six contrasts. Intervals crossing zero do not establish equivalence.',table_md(format_effects(g),['condition','scenario','difference [95% t CI]','simultaneous 95% family CI','Holm p','paired dz','crossed bootstrap 95%'])]
    secondary=c[(c.condition=='A_corrected')&(c.comparison=='linear_q_approx_minus_contextual_bandit')&(c.metric=='mean_reward')].copy()
    secondary['difference [pointwise 95% CI]']=[f'{r["mean"]:.4f} [{r.ci95_low:.4f}, {r.ci95_high:.4f}]' for _,r in secondary.iterrows()]
    parts += ['## Secondary comparison — linear Q approximation',
              'These are secondary, pointwise intervals over training means, without a prespecified multiplicity-adjusted superiority family. Linear Q uses all raw features, whereas the two tabular learners share a reduced state key. Its contrasts therefore combine representation and learning differences.',
              table_md(secondary,['environment','scenario','difference [pointwise 95% CI]'])]
    reference_rows=[]
    for scenario,g in levels[(levels.condition=='A_corrected')&(levels.environment=='sequential')&(levels.metric=='mean_reward')].groupby('scenario'):
        row={'scenario':scenario}
        for policy in LEARNED+['security_rule','heuristic_score_threshold']:
            r=g[g.policy==policy].iloc[0]
            row[policy]=f'{r["mean"]:.4f} ± {r.sd_training:.4f}' if policy in LEARNED else f'{r["mean"]:.4f}'
        reference_rows.append(row)
    parts += ['## Corrected sequential reward levels and deterministic references',
              'Learned-policy entries give mean ± between-training SD. Rule entries give fixed evaluation-panel means only; there are no independent rule training runs. Permissive and restrictive reference levels and all nonsequential cells remain in the complete levels table.',
              table_md(pd.DataFrame(reference_rows),['scenario']+LEARNED+['security_rule','heuristic_score_threshold'])]
    parts+=['## All operational outcomes and secondary policy contrasts',
            'Full means, between-training SD and pointwise confidence intervals are in `analysis/policy_levels.csv`; paired differences and standardized effects are in `analysis/paired_policy_contrasts.csv`; the complete mechanism contrasts are in `analysis/condition_interactions.csv`. No overall governance dominance is inferred from reward alone. Rule levels are descriptive; rule-relative learner intervals condition on the same fixed ten rule evaluations.',
            '### Corrected sequential Q-minus-bandit operational differences (pointwise intervals)',]
    operational=c[(c.condition=='A_corrected')&(c.environment=='sequential')&
                  (c.comparison=='q_learning_minus_contextual_bandit')&c.metric.isin(METRICS[1:7])].copy()
    operational['difference [95% t CI]']=[f'{r["mean"]:.3f} [{r.ci95_low:.3f}, {r.ci95_high:.3f}]' for _,r in operational.iterrows()]
    parts.append(table_md(operational,['scenario','metric','difference [95% t CI]']))
    parts+=['## Limits on inference and interpretation',
            'Holm correction applies within each specified primary reward family; pointwise intervals and secondary operational metrics do not establish family-wise superiority. The crossed bootstrap is a sensitivity over only ten evaluation seeds. Reused evaluation seeds and the action-dependent RNG prevent a claim of perfectly paired exogenous shocks. Neither a zero-crossing interval nor an insignificant interaction establishes equivalence.',
            'Feedback loss manipulates missing scalar teaching reports with zero encoding, not the true governance objective or all state sensors. Observation changes also change statistical complexity; sparse tabular coverage is an alternative explanation. The inherited V7 state already contains privileged outcome aggregates. C enrichment adds only administrative schedules and ages. The two terminal corrections are bundled, so their individual contributions cannot be identified.',
            'This simulation cannot establish an intrinsic ranking of reinforcement learning and contextual bandits or validate effects on real platforms. No feedback-by-observability interactions were executed. Source preservation, completeness and additive reward checks are reported in the execution report.']
    (BASE/'V8_RESULTS_SUMMARY.md').write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
    (OUT/'primary_result_counts.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')
    insertions(c,inter,counts)

def insertions(c,inter,counts):
    a=counts['A_corrected'];legacy=counts['A_v7_exact']
    primary_mechanisms=inter[inter.metric=='mean_reward']
    confirmed=int((primary_mechanisms.p_holm<.05).sum())
    mechanism_interpretation=('None of the twelve prespecified feedback or observability interaction contrasts survived the multiplicity adjustment. These interventions therefore did not substantiate feedback-report availability or the tested administrative information channels as explanations of the policy gap. This does not establish equivalence or the absence of other feedback/observability mechanisms.' if confirmed==0 else 'The mechanism claims are restricted to the specific contrasts surviving the prespecified multiplicity adjustment.')
    operational=c[(c.condition=='A_corrected')&(c.environment=='sequential')&
                  (c.scenario=='balanced_reference')&(c.comparison=='q_learning_minus_contextual_bandit')].set_index('metric')['mean'].to_dict()
    text=f'''# Manuscript insertions — English, based on completed V8 runs

## Methodology

We extended the existing simulator without altering its archived implementation. The robustness experiment included 30 independent training runs for each learning policy in each of six scenarios and two environments, with 180 episodes of 100 decisions per run and ten evaluation seeds per trained policy. A prospective seed-audit amendment retained the three historical groups as diagnostics and selected thirty newly trained, disjoint seed blocks for primary inference. An exact-replication stratum retained the original implementation; a separate corrected stratum removed terminal value bootstrapping and settled incidents generated during terminal review resolution. We retained the contextual bandit, tabular Q-learning, a secondary linear Q approximation, and four deterministic reference policies. The contextual bandit updated only from current-action reward, whereas the sequential learners updated from transition rewards. The common evaluation objective remained total governance reward per decision. We averaged paired evaluations within training run and constructed Student t intervals over 30 training means, with Holm adjustment within the prespecified primary contrast families. These intervals condition on the ten fixed evaluation seeds; a crossed bootstrap over training and evaluation indices provided a separate sensitivity analysis. Common evaluation seeds do not guarantee identical exogenous trajectories because the inherited simulator consumes random draws in an action-dependent order.

To examine mechanisms separately, we selected the balanced, high-security-pressure, and constrained-review-capacity scenarios before running the expanded experiments. Reward-feedback availability was set to 100%, 75%, or 50% by independently masking scalar component reports during training, with unavailable reports encoded as zero. The true environment dynamics and evaluation rewards were unchanged. The observability experiment retained the original observation, masked queue/capacity channels, or added administrative timing information for existing jobs. The primary tabular comparison used the same representation within each condition; enrichment added one binary workload-timing state component, limiting the tabular expansion to a factor of two. Both interventions were evaluated as changes in the paired Q-learning-minus-bandit reward contrast relative to the corrected control.

## Results

In the exact-replication stratum, the Q-learning-minus-bandit mean reward difference was negative in {legacy['negative_point_means']} of twelve environment–scenario cells; {legacy['negative_holm']} negative contrasts remained significant after Holm adjustment within the twelve-comparison family, and {legacy['positive_holm']} positive contrasts did so. In the corrected stratum, {a['negative_point_means']} of twelve mean contrasts were negative, with {a['negative_holm']} negative and {a['positive_holm']} positive Holm-adjusted results. These statements concern variation between training runs conditional on a fixed evaluation panel. The full scenario-level estimates, standardized effects, pointwise and simultaneous confidence intervals, and crossed-bootstrap sensitivity intervals should accompany this paragraph; the counts do not replace those estimates.

The feedback experiment yielded {counts['B']['relative_improvements_holm']} Holm-adjusted relative improvements and {counts['B']['relative_deteriorations_holm']} relative deteriorations for Q-learning across six loss-versus-complete-feedback contrasts. The observability experiment yielded {counts['C']['relative_improvements_holm']} relative improvements and {counts['C']['relative_deteriorations_holm']} relative deteriorations across six reduced/enriched-versus-original-information contrasts. A relative improvement is a change in the policy difference and must not be reported as an absolute Q-learning advantage. The accompanying interaction table reports every scenario, including intervals crossing zero. Separate acceptance, incident, restriction, workload, expenditure, and guard-intervention outcomes qualify any reward-based interpretation.

{mechanism_interpretation}

The secondary operational outcomes show why reward rankings should not be interpreted as complete governance dominance. In the corrected balanced sequential scenario, Q-learning descriptively accepted {operational['legitimate_contributions_accepted']:.2f} more legitimate contributions and imposed {abs(operational['rejected_legitimate_innovation']):.2f} fewer unjustified restrictions per episode than the bandit, while requiring {operational['human_review_workload']:.2f} additional human-review actions and {operational['governance_expenditure']:.2f} additional governance-expenditure units. These are secondary point estimates; their pointwise intervals and the separate incident contrast are reported in the operational-outcome table.

## Discussion

The extension distinguishes the existence of delayed governance consequences from the ability of a particular learning policy to exploit them. The mechanism interventions test the consequences of missing teaching signals and altered administrative information in this simulator. They do not identify a general explanation for the relative performance of all contextual and sequential algorithms. For boundary-resource governance, the relevant design issue is whether the available feedback and state representation support decisions whose downstream benefits exceed their learning and operational costs. The results therefore support conditional evaluation of algorithmic sophistication, with separate attention to participation, legitimate contributions, security, and human review. They do not imply that reinforcement learning is intrinsically inferior to contextual learning.

## Limitations

The simulator contains stylized, uncalibrated behavioral probabilities and an inherited observation vector with privileged aggregate outcome indicators. Participation changes the distribution of submissions rather than their arrival count. The contextual comparator is an immediate-reward learner, not a delayed-feedback bandit that reattributes later outcomes to originating decisions. Missing reports change the training signal, while information enrichment also changes learning complexity; neither intervention isolates every aspect of feedback quality or observability. Thirty independent training runs improve inference about training variability, but only ten fixed evaluation seeds were used per run. The corrected-versus-legacy sensitivity bundles two implementation changes. No feedback-by-observability factorial interaction or real-platform intervention was conducted. All conclusions are limited to the policies, scenario configurations, safety masks, horizons, and training budgets examined.

## Editorial integration note

Replace the old three-training-run uncertainty statements with the appropriate V8 stratum and tables. Keep the exact V7 and corrected estimates visibly separate. These paragraphs are proposed insertions, not edits to any archived manuscript; a full editorial revision for Electronic Commerce Research has not been performed.
'''
    (BASE/'V8_MANUSCRIPT_INSERTIONS.md').write_text(text,encoding='utf-8')

def save_plot(fig,name):
    fig.tight_layout()
    for extension in ['png','pdf','svg']:
        fig.savefig(FIG/f'{name}.{extension}',dpi=350,bbox_inches='tight')
    plt.close(fig)

def forest(ax,g,label_columns,title):
    g=g.reset_index(drop=True)
    y=np.arange(len(g))
    ax.hlines(y,g.simultaneous_low,g.simultaneous_high,color='#adc4d6',lw=4,label='Simultaneous 95% family CI')
    ax.hlines(y,g.ci95_low,g.ci95_high,color='#215a7b',lw=1.8,label='Pointwise 95% CI')
    ax.scatter(g['mean'],y,c='#112f42',s=24,zorder=3)
    ax.axvline(0,color='.4',lw=.8,ls='--')
    lo=min(0.,float(g.simultaneous_low.min()));hi=max(0.,float(g.simultaneous_high.max()))
    padding=max(.001,(hi-lo)*.06)
    ax.set_xlim(lo-padding,hi+padding)
    ax.set_yticks(y,[' / '.join(str(r[c]).replace('_',' ') for c in label_columns) for _,r in g.iterrows()],fontsize=8)
    ax.invert_yaxis();ax.set_title(title,fontsize=11)
    ax.grid(axis='x',alpha=.15)
    ax.spines[['top','right']].set_visible(False)

def figures(c,inter,levels):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    for condition in ['A_v7_exact','A_corrected']:
        g=c[(c.condition==condition)&(c.comparison=='q_learning_minus_contextual_bandit')&(c.metric=='mean_reward')]
        fig,ax=plt.subplots(figsize=(9.2,5.3))
        title='V7 implementation' if condition=='A_v7_exact' else 'Corrected implementation'
        forest(ax,g,['environment','scenario'],title+' — 30 independent training runs per cell')
        ax.set_xlabel('Q-learning − contextual bandit: governance reward per decision')
        ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.15),ncol=2,frameon=False)
        save_plot(fig,condition+'_reward_contrasts')
    for prefix,title in [('B','Feedback availability'),('C','Administrative observability')]:
        g=inter[inter.condition.str.startswith(prefix)&(inter.metric=='mean_reward')]
        fig,ax=plt.subplots(figsize=(9.3,4))
        forest(ax,g,['condition','scenario'],title+' — change relative to the corrected control')
        ax.set_xlabel('Change in Q-learning − bandit reward gap (positive = relative improvement of Q)')
        ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.20),ncol=2,frameon=False)
        save_plot(fig,prefix+'_mechanism_interactions')
    diagnostics=pd.read_csv(BASE/'V8_TRAINING_DIAGNOSTICS.csv')
    diag=diagnostics[(diagnostics.environment=='sequential')&diagnostics.policy.isin(['contextual_bandit','q_learning'])&
                     diagnostics.scenario_index.isin([0,1,3])&diagnostics.condition.ne('A_v7_exact')]
    agg=diag.groupby(['condition','policy']).visited_states.mean().unstack('policy')
    fig,ax=plt.subplots(figsize=(7,4))
    agg.plot.bar(ax=ax,color=['#27858e','#c57435'])
    ax.set_ylabel('Mean visited tabular states per training run\n(same three mechanism scenarios)')
    ax.set_xlabel('Condition');ax.tick_params(axis='x',labelrotation=25)
    ax.legend(['Contextual bandit','Q-learning'],frameon=False)
    labels={'A_corrected':'Control','B_loss25':'25% feedback loss','B_loss50':'50% feedback loss',
            'C_enriched':'Enriched observation','C_reduced':'Reduced observation'}
    ax.set_xticklabels([labels[x] for x in agg.index],rotation=25,ha='right')
    ax.spines[['top','right']].set_visible(False)
    save_plot(fig,'state_coverage_diagnostic')

if __name__=='__main__': analyze()
