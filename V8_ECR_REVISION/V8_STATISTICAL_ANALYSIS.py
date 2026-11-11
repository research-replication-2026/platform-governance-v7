"""Recompute all A/B/C and nested-budget estimates from executed evaluation rows.

No model is trained here. The frozen primary implementation reproduces the
previously validated analysis; the exploratory budget analysis is separate.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import hashlib,json,math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B/'FROZEN_SIMULATOR'))
import frozen_primary_analysis as primary
O=B/'ANALYSIS'

def summarize(values,metadata):
    row=primary.interval(np.asarray(values,dtype=float));row.update(metadata)
    return row

def budgets():
    status=json.loads((B/'BUDGET_STATUS.json').read_text())
    assert status['completed_jobs']==status['planned_jobs']==60
    d=pd.read_csv(B/'BUDGET_RESULTS.csv')
    assert len(d)==5400 and not d.duplicated(['scenario','policy','budget','train_seed','eval_seed']).any()
    assert d.residual_pending_items.max()==0
    unit=d.groupby(['scenario','policy','budget','train_seed'],as_index=False).mean_reward.mean()
    unit.to_csv(O/'budget_training_means.csv',index=False)
    levels=[];gaps=[];changes=[]
    for (scenario,budget),g in unit.groupby(['scenario','budget']):
        for policy,z in g.groupby('policy'):
            assert len(z)==30
            levels.append(summarize(z.mean_reward,dict(scenario=scenario,budget=budget,policy=policy)))
        p=g.pivot(index='train_seed',columns='policy',values='mean_reward').sort_index()
        for policy in ['q_learning','linear_q_approx']:
            gaps.append(summarize(p[policy]-p.contextual_bandit,dict(scenario=scenario,budget=budget,
               comparison=policy+'_minus_contextual_bandit',family='budget_absolute_Q_4' if policy=='q_learning' and budget>180 else '',p_holm=np.nan,simultaneous_low=np.nan,simultaneous_high=np.nan)))
    for scenario,g in unit.groupby('scenario'):
        p=g.pivot(index='train_seed',columns=['policy','budget'],values='mean_reward')
        for policy in ['q_learning','linear_q_approx']:
            baseline=p[(policy,180)]-p[('contextual_bandit',180)]
            for budget in [360,720]:
                values=p[(policy,budget)]-p[('contextual_bandit',budget)]-baseline
                changes.append(summarize(values,dict(scenario=scenario,budget=budget,comparison=policy+'_minus_contextual_bandit',family='budget_change_Q_4' if policy=='q_learning' else '',p_holm=np.nan,simultaneous_low=np.nan,simultaneous_high=np.nan)))
    gaps=pd.DataFrame(gaps);changes=pd.DataFrame(changes)
    for table in [gaps,changes]:
        selected=table.index[table.family.ne('')];assert len(selected)==4
        table.loc[selected,'p_holm']=primary.holm(table.loc[selected,'p_raw'])
        radius=stats.t.ppf(1-.05/8,29)*table.loc[selected,'sd_training']/np.sqrt(30)
        table.loc[selected,'simultaneous_low']=table.loc[selected,'mean']-radius
        table.loc[selected,'simultaneous_high']=table.loc[selected,'mean']+radius
    pd.DataFrame(levels).to_csv(O/'budget_policy_levels.csv',index=False)
    gaps.to_csv(O/'budget_policy_gaps.csv',index=False);changes.to_csv(O/'budget_gap_changes.csv',index=False)
    operational=[]
    for (scenario,budget),group in d.groupby(['scenario','budget']):
        for metric in primary.METRICS[1:7]:
            p=group.groupby(['train_seed','policy'])[metric].mean().unstack('policy')
            operational.append(summarize(p.q_learning-p.contextual_bandit,dict(scenario=scenario,budget=budget,metric=metric)))
    pd.DataFrame(operational).to_csv(O/'budget_operational_contrasts.csv',index=False)
    return gaps,changes

def md(table,columns): return primary.table_md(table,columns)
def main():
    primary.BASE=B;primary.OUT=O;primary.FIG=B/'FIGURES'
    primary.figures=lambda *args:None;primary.write_summary=lambda *args:None
    primary.analyze()
    gaps,changes=budgets()
    c=pd.read_csv(O/'paired_policy_contrasts.csv');i=pd.read_csv(O/'condition_interactions.csv')
    main=c[(c.condition=='A_corrected')&(c.comparison=='q_learning_minus_contextual_bandit')&(c.metric=='mean_reward')].copy()
    mechanism=i[i.metric=='mean_reward'].copy()
    def format_rows(t):
        t=t.copy();t['difference [pointwise 95% CI]']=[f'{r["mean"]:.5f} [{r.ci95_low:.5f}, {r.ci95_high:.5f}]' for _,r in t.iterrows()]
        t['family 95% CI']=[f'[{r.simultaneous_low:.5f}, {r.simultaneous_high:.5f}]' for _,r in t.iterrows()]
        return t
    parts=['# V8 ECR — results from completed experiments',
      'A/B/C reuse 33,840 verified primary evaluations: 30 independent training blocks per learned-policy cell, ten evaluations per block. Exact V7 and corrected V8 remain separate. The additional nested-budget supplement contributes 5,400 snapshot evaluations from 180 policy trajectories; its 1,800 evaluations at 180 episodes reproduce existing primary results and are not new independent evidence.',
      'Primary unit: the mean over ten evaluations of one independently trained policy. Paired t intervals use 29 degrees of freedom. Holm tests and Bonferroni simultaneous intervals use separate 12-cell A families and separate six-cell B/C interaction families. Secondary operational and linear-Q intervals are pointwise. Standardized paired effects dz are conditional on the ten-seed evaluation panel.',
      '## Main corrected reward comparisons',md(format_rows(main),['environment','scenario','difference [pointwise 95% CI]','family 95% CI','p_holm','dz']),
      'All twelve corrected and all twelve exact-V7 primary reward contrasts are negative after their respective Holm adjustment. This concerns tested algorithms under specified budgets and synthetic dynamics; it is not a general ranking of reinforcement learning.',
      '## Feedback and observation interventions',md(format_rows(mechanism),['condition','scenario','difference [pointwise 95% CI]','family 95% CI','p_holm','dz']),
      'No B or C interaction survives its six-comparison family adjustment. This does not establish equivalence or exclude other feedback and observability mechanisms. B masks scalar teaching reports with zero encoding, scaling expected non-guard training rewards by 1-p; C also changes representation complexity. Inherited privileged aggregate indicators limit real-world applicability.',
      '## Exploratory nested-budget supplement',
      'This supplement was prospectively specified after primary V8 results were known. It extends exactly the same trajectories, with frozen epsilon decay of 60 episodes, to snapshots at 180/360/720 episodes in two sequential scenarios. Separate four-test exploratory families address absolute tabular-Q gaps and their changes from 180. No hyperparameter search or equivalence margin was specified.',
      md(format_rows(gaps),['scenario','budget','comparison','difference [pointwise 95% CI]','family 95% CI','p_holm','dz']),
      md(format_rows(changes),['scenario','budget','comparison','difference [pointwise 95% CI]','family 95% CI','p_holm','dz']),
      '## Interpretation of the new budget results',
      'At 720 episodes the Q-minus-bandit disadvantage narrows under high security pressure but widens in the reference scenario. Both changes survive their separate exploratory four-test adjustment. All four new absolute tabular-Q gaps remain negative after adjustment. Additional experience has scenario-dependent effects and does not reverse the ranking in these two settings.',
      'Longer training is a sensitivity check, not a proof of convergence or adequate sample size for every representation. Shared random seeds do not perfectly pair exogenous shocks because draw consumption is action dependent. Inference remains conditional on the ten-seed evaluation panel. Budget scenarios are purposive and do not represent all six governance settings.',
      '## Operational outcomes and complete estimates',
      'ANALYSIS/policy_levels.csv and paired_policy_contrasts.csv report legitimate acceptance, incidents, unjustified restrictions, review workloads, costs, guard interventions and further diagnostics. Reward advantages do not establish operational dominance. Four deterministic references have ten evaluations each and no independent training replications. Source data, all effect sizes and pointwise intervals remain available, including results that contradict simple policy rankings.']
    (B/'V8_RESULTS_REPORT.md').write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
    print('Primary and nested-budget analyses complete.')

if __name__=='__main__': main()
