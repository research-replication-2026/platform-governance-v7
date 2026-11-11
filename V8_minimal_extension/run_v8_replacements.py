"""Additive prospective seed amendment; retain all original runs, select 30 new blocks."""
import sys
sys.dont_write_bytecode=True
import argparse,hashlib,itertools,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import pandas as pd
import run_v8 as original
BASE=Path(__file__).resolve().parent
PRIMARY_SEEDS=list(range(10001,10031))
EXTRA_SEEDS=[10028,10029,10030]
# Must also be applied when Windows spawn imports this module in each worker.
original.SEEDS.extend(s for s in EXTRA_SEEDS if s not in original.SEEDS)

def manifest():
    path=BASE/'V8_AMENDMENT_01_MANIFEST.json'
    if path.exists():
        result=json.loads(path.read_text())
        assert result['amendment_sha256']==original.digest(BASE/'V8_PROTOCOL_AMENDMENT_01.md')
        assert result['runner_sha256']==original.digest(Path(__file__))
        return result
    initial=json.loads((BASE/'V8_EXPERIMENT_MANIFEST.json').read_text())
    templates=[j for j in initial['jobs']+initial['reused_jobs'] if j['train_seed']==1101]
    assert len(templates)==36
    jobs=[]
    for template in templates:
        for seed in EXTRA_SEEDS:
            job=dict(template,train_seed=seed)
            job['id']=f"{job['condition']}__{job['environment']}__s{job['scenario_index']}__t{seed}"
            jobs.append(job)
    rows=[]
    for environment in ['nonsequential','sequential']:
        for i in range(6):
            streams=[]
            for label in PRIMARY_SEEDS:
                united=set()
                for policy,stride in [('contextual_bandit',13),('q_learning',17),('linear_q_approx',19)]:
                    united.update(original.actual_seed(label,i,environment,policy)+e*stride for e in range(180))
                streams.append(united)
            overlap=sum(len(a&b) for a,b in itertools.combinations(streams,2))
            evals={original.v7.SEED+(500000 if environment=='sequential' else 0)+10000*i+e for e in original.v7.MAIN_EVAL_SEEDS}
            eval_overlap=len(set.union(*streams)&evals)
            assert overlap==eval_overlap==0
            rows.append(dict(environment=environment,scenario=original.v7.SCENARIOS[i].name,
                             cross_block_episode_seed_collisions_across_all_algorithms=overlap,
                             train_eval_seed_collisions=eval_overlap,primary_training_blocks=30))
    pd.DataFrame(rows).to_csv(BASE/'PRIMARY_EFFECTIVE_SEED_AUDIT.csv',index=False)
    result=dict(created_utc=original.now(),amendment_sha256=original.digest(BASE/'V8_PROTOCOL_AMENDMENT_01.md'),
                runner_sha256=original.digest(Path(__file__)),original_manifest_sha256=original.digest(BASE/'V8_EXPERIMENT_MANIFEST.json'),
                primary_seeds=PRIMARY_SEEDS,diagnostic_seeds=[1101,1201,1301],jobs=jobs,
                jobs_total_all_manifests=initial['jobs_total']+len(jobs),status='frozen_before_primary_outcome_analysis')
    path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

def collect(amendment):
    initial=json.loads((BASE/'V8_EXPERIMENT_MANIFEST.json').read_text())
    original.collect(initial)
    more_rows=[];more_diags=[];more_curves=[]
    for job in amendment['jobs']:
        p=BASE/'checkpoints'/f"{job['id']}.json"
        assert p.exists(),f'Amendment incomplete: {job["id"]}'
        payload=json.loads(p.read_text())
        more_rows.extend(payload['rows'])
        meta={k:job[k] for k in ['condition','environment','train_seed','scenario_index']}
        more_diags.extend(dict(**meta,**d) for d in payload['diagnostics'])
        more_curves.extend(dict(**meta,**d) for d in payload['curves'])
    for name,extra in [('V8_RESULTS',more_rows),('V8_TRAINING_DIAGNOSTICS',more_diags),('V8_TRAINING_CURVES',more_curves)]:
        original_data=pd.read_csv(BASE/f'{name}.csv')
        all_data=pd.concat([original_data,pd.DataFrame(extra)],ignore_index=True)
        all_data.to_csv(BASE/f'{name}_ALL_EXECUTED.csv',index=False)
        primary=all_data[all_data.train_seed.isin([0]+PRIMARY_SEEDS)]
        primary.to_csv(BASE/f'{name}.csv',index=False)
        if name=='V8_RESULTS':
            all_data[all_data.train_seed.isin([1101,1201,1301])].to_csv(BASE/'V8_HISTORICAL_SEED_DIAGNOSTICS.csv',index=False)
            primary_counts=primary.groupby('condition').size().to_dict()
    status=dict(updated_utc=original.now(),complete_jobs=amendment['jobs_total_all_manifests'],
                planned_jobs=amendment['jobs_total_all_manifests'],result_rows=33840,
                all_executed_rows=37080,complete_by_condition=primary_counts,pending_jobs=[],
                primary_seeds=PRIMARY_SEEDS,amendment='01')
    (BASE/'V8_RUN_STATUS.json').write_text(json.dumps(status,indent=2,default=int),encoding='utf-8')
    print(json.dumps(status,default=int),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--collect',action='store_true')
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    amendment=manifest()
    if args.prepare:
        print(f'Frozen additive manifest with {len(amendment["jobs"])} jobs, primary labels 10001..10030.');return
    if args.collect: collect(amendment);return
    pending=[j for j in amendment['jobs'] if not (BASE/'checkpoints'/f"{j['id']}.json").exists()]
    print(f'Executing {len(pending)} additional seed-amendment jobs.',flush=True)
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(original.run_job,j):j for j in pending}
        for n,f in enumerate(as_completed(futures),1):
            ident,seconds=f.result()
            event=dict(time_utc=original.now(),job=ident,status='complete',seconds=seconds,amendment='01')
            with (BASE/'campaign.jsonl').open('a',encoding='utf-8') as handle: handle.write(json.dumps(event)+'\n')
            if n%12==0 or n==len(pending): print(f'Amendment completed={n}/{len(pending)} elapsed={time.perf_counter()-start:.1f}s',flush=True)
    collect(amendment)

if __name__=='__main__': main()
