"""Verify scientific results in the public package without private archives."""
import sys
sys.dont_write_bytecode=True
import argparse,ast,gzip,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from public_materialize import materialize
BASE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fresh',action='store_true',help='Check a newly recomputed dataset without archive materialization')
    args=parser.parse_args()
    if not args.fresh: materialize()
    ledger=json.loads((BASE/'PUBLIC_DATA_MANIFEST.json').read_text())
    for name,fingerprint in ledger['scientific_source_sha256'].items():
        assert hashlib.sha256((BASE/name).read_bytes()).hexdigest()==fingerprint,name
    experiment=json.loads((BASE/'V8_EXPERIMENT_MANIFEST.json').read_text())
    amendment=json.loads((BASE/'V8_AMENDMENT_01_MANIFEST.json').read_text())
    assert experiment['engine_sha256']==ledger['scientific_source_sha256']['v8_engine.py']
    assert experiment['runner_sha256']==ledger['scientific_source_sha256']['run_v8.py']
    assert experiment['vendor_sha256']==ledger['scientific_source_sha256']['vendor/v7_frozen.py']
    assert amendment['runner_sha256']==ledger['scientific_source_sha256']['run_v8_replacements.py']
    assert amendment['primary_seeds']==list(range(10001,10031))
    # Public V7 changes only report wording/path presentation relative to the
    # frozen simulation used for V8. Compare every scientific AST node exactly.
    old=ast.parse((BASE.parent/'experiments_v7/scripts/run_experiments_v7.py').read_text(encoding='utf-8-sig'))
    frozen=ast.parse((BASE/'vendor/v7_frozen.py').read_text(encoding='utf-8-sig'))
    presentation={'scientific_report_md','methods_report_md','make_figures'}
    for node in [old,frozen]: node.body=[n for n in node.body if getattr(n,'name',None) not in presentation]
    assert ast.dump(old,include_attributes=False)==ast.dump(frozen,include_attributes=False)
    x=pd.read_csv(BASE/'V8_RESULTS.csv')
    learned=x[x.policy.isin(['contextual_bandit','q_learning','linear_q_approx'])]
    rules=x[~x.policy.isin(['contextual_bandit','q_learning','linear_q_approx'])]
    assert len(x)==33840 and len(learned)==32400 and len(rules)==1440
    assert set(learned.train_seed)==set(range(10001,10031)) and set(rules.train_seed)=={0}
    keys=['condition','environment','scenario','policy']
    assert learned.groupby(keys).train_seed.nunique().eq(30).all()
    assert learned.groupby(keys+['train_seed']).size().eq(10).all()
    assert not x.duplicated(keys+['train_seed','eval_seed']).any()
    assert x[x.implementation=='v8_corrected'].residual_pending_items.eq(0).all()
    components=['current_action_reward_total','delayed_past_reward_total','collective_state_reward_total',
                'terminal_liquidation_reward_total','guard_penalty_total']
    assert np.abs(x.total_reward-x[components].sum(axis=1)).max()<1e-9
    panel=learned[learned.policy.isin(['contextual_bandit','q_learning'])].pivot(
        index=['condition','environment','scenario','train_seed','eval_seed'],columns='policy',values='mean_reward')
    difference=(panel.q_learning-panel.contextual_bandit).groupby(level=[0,1,2,3]).mean()
    reported=pd.read_csv(BASE/'analysis/paired_policy_contrasts.csv')
    reported=reported[(reported.metric=='mean_reward')&(reported.comparison=='q_learning_minus_contextual_bandit')].set_index(['condition','environment','scenario'])
    for key,g in difference.groupby(level=[0,1,2]):
        expected=reported.loc[key]
        radius=stats.t.ppf(.975,29)*g.std(ddof=1)/np.sqrt(30)
        np.testing.assert_allclose([g.mean(),g.std(ddof=1),g.mean()-radius,g.mean()+radius],
                                  expected[['mean','sd_training','ci95_low','ci95_high']].to_numpy(dtype=float),atol=1e-12,rtol=0)
    mechanisms=pd.read_csv(BASE/'analysis/condition_interactions.csv')
    for _,r in mechanisms[mechanisms.metric=='mean_reward'].iterrows():
        d=difference.loc[(r.condition,r.environment,r.scenario)]-difference.loc[('A_corrected',r.environment,r.scenario)]
        assert abs(d.mean()-r['mean'])<1e-12 and abs(d.std(ddof=1)-r.sd_training)<1e-12
    all_x=pd.read_csv(BASE/'V8_RESULTS_ALL_EXECUTED.csv')
    assert len(all_x)==37080
    assert len(pd.read_csv(BASE/'V8_HISTORICAL_SEED_DIAGNOSTICS.csv'))==3240
    seed_audit=pd.read_csv(BASE/'PRIMARY_EFFECTIVE_SEED_AUDIT.csv')
    assert seed_audit.cross_block_episode_seed_collisions_across_all_algorithms.eq(0).all()
    counts=json.loads((BASE/'analysis/primary_result_counts.json').read_text())
    assert counts['A_corrected']['negative_holm']==12 and counts['A_v7_exact']['negative_holm']==12
    assert counts['B']['relative_improvements_holm']==counts['B']['relative_deteriorations_holm']==0
    assert counts['C']['relative_improvements_holm']==counts['C']['relative_deteriorations_holm']==0
    print(json.dumps(dict(status='verified',primary_rows=len(x),all_executed_rows=len(all_x),
                         learned_cells=108,training_blocks_per_cell=30,evaluations_per_block=10,
                         corrected_terminal_residuals=0,scientific_code_matches_V7=True,
                         note='Public scientific verification; private manuscript archives and machine checkpoints are not required.'),indent=2))

if __name__=='__main__': main()
