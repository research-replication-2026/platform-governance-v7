"""Frozen manifest, resumable process-parallel V8 campaign; one checkpoint per job."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import argparse,hashlib,json,os,time,traceback,platform
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from v8_engine import v7,train,evaluate
BASE=Path(__file__).resolve().parent
LEARNED=['contextual_bandit','q_learning','linear_q_approx']
RULES=['static_permissive','static_restrictive','security_rule','heuristic_score_threshold']
SEEDS=[1101,1201,1301]+list(range(10001,10028))

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def actual_seed(label,scenario_index,environment,policy):
    offset=500000 if environment=='sequential' else 0
    add={'contextual_bandit':101,'q_learning':202,'linear_q_approx':404}[policy]
    if label in [1101,1201,1301]:
        return v7.SEED+offset+scenario_index*10000+label+add
    return 100000000+(SEEDS.index(label)-3)*1000000+offset+scenario_index*10000+add

def make_manifest():
    assert (BASE/'V7_TARGETED_REPRODUCTION.csv').exists(), 'Validate before preparation'
    benchmark=pd.read_csv(BASE/'RUNTIME_BENCHMARK.csv')
    conditions=[('A_v7_exact','A','v7',0.,True),('A_corrected','A','v7',0.,False),
                ('B_loss25','BC','v7',.25,False),('B_loss50','BC','v7',.5,False),
                ('C_reduced','BC','reduced',0.,False),('C_enriched','BC','enriched',0.,False)]
    jobs=[]; reused=[]
    for condition,stage,mode,loss,legacy in conditions:
        environments=['nonsequential','sequential'] if stage=='A' else ['sequential']
        indices=list(range(6)) if stage=='A' else [0,1,3]
        for environment in environments:
            for i in indices:
                for label in [0]+SEEDS:
                    task=dict(condition=condition,stage=stage,mode=mode,loss=loss,legacy=legacy,
                              environment=environment,scenario_index=i,train_seed=label,episodes=180)
                    task['id']=f'{condition}__{environment}__s{i}__t{label}'
                    if legacy and label in [0,1101,1201,1301]: reused.append(task)
                    else: jobs.append(task)
    # Effective-seed separation and within-policy independent training streams.
    seed_audit=[]
    for env in ['nonsequential','sequential']:
        for i in range(6):
            evaluation={v7.SEED+(500000 if env=='sequential' else 0)+i*10000+e for e in v7.MAIN_EVAL_SEEDS}
            for policy,multiplier in [('contextual_bandit',13),('q_learning',17),('linear_q_approx',19)]:
                streams=[{actual_seed(s,i,env,policy)+e*multiplier for e in range(180)} for s in SEEDS]
                overlap=sum(len(a&b) for j,a in enumerate(streams) for b in streams[j+1:])
                eval_overlap=len(set.union(*streams)&evaluation)
                assert overlap==eval_overlap==0,(env,i,policy,overlap,eval_overlap)
                seed_audit.append(dict(environment=env,scenario=v7.SCENARIOS[i].name,policy=policy,
                                       within_policy_cross_training_episode_seed_collisions=overlap,
                                       training_evaluation_seed_collisions=eval_overlap))
    # This is conditional independence of learner RNG streams within a policy.
    # Across algorithms, common legacy integer seeds may occur and are reported.
    pd.DataFrame(seed_audit).to_csv(BASE/'EFFECTIVE_SEED_AUDIT.csv',index=False)
    serial=sum(float(benchmark[(benchmark.kind==('legacy' if j['legacy'] else 'corrected'))&
                               (benchmark.policy==p)].iloc[0].seconds)*
               (1 if j['legacy'] else 9) for j in jobs if j['train_seed'] for p in LEARNED)
    manifest=dict(created_utc=now(),protocol_sha256=digest(BASE/'V8_EXPERIMENTAL_PROTOCOL.md'),
                  engine_sha256=digest(BASE/'v8_engine.py'),runner_sha256=digest(Path(__file__)),
                  vendor_sha256=digest(BASE/'vendor/v7_frozen.py'),train_seeds=SEEDS,
                  eval_seed_nominal=v7.MAIN_EVAL_SEEDS,scenarios=[asdict(c) for c in v7.SCENARIOS],
                  policies=LEARNED+RULES,conditions=conditions,jobs=jobs,reused_jobs=reused,
                  serial_training_seconds_estimate=serial,cpu_logical=os.cpu_count(),
                  python=sys.version,platform=platform.platform(),
                  jobs_total=len(jobs)+len(reused),fresh_jobs=len(jobs),reused_jobs_count=len(reused),
                  status='frozen_before_campaign')
    path=BASE/'V8_EXPERIMENT_MANIFEST.json'
    if path.exists():
        raise FileExistsError('Manifest already frozen; resume rather than overwrite')
    path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ['jobs_total','fresh_jobs','reused_jobs_count','serial_training_seconds_estimate','cpu_logical']},indent=2))

def run_job(job):
    start=time.perf_counter()
    cfg=v7.SCENARIOS[job['scenario_index']]
    env=job['environment']
    mech=v7.FULL_SEQUENTIAL if env=='sequential' else v7.NONSEQUENTIAL
    rows=[];diagnostics=[];curves=[]
    policies=RULES if job['train_seed']==0 else LEARNED
    for policy in policies:
        seed=actual_seed(job['train_seed'],job['scenario_index'],env,policy) if job['train_seed'] else None
        feedback_seed=4000000000+job['scenario_index']*10000+SEEDS.index(job['train_seed']) if seed else 0
        learner=train(cfg,env,mech,policy,seed,job['episodes'],job['mode'],job['loss'],job['legacy'],feedback_seed) if seed else None
        if learner:
            diagnostics.append(dict(policy=policy,actual_training_seed=seed,feedback_seed=feedback_seed,**learner.diagnostics))
            curves.extend(dict(policy=policy,**point) for point in learner.training_curve)
        for nominal in v7.MAIN_EVAL_SEEDS:
            eval_seed=v7.SEED+(500000 if env=='sequential' else 0)+job['scenario_index']*10000+nominal
            row=evaluate(cfg,env,mech,policy,eval_seed,job['train_seed'],learner,job['mode'],job['legacy'])
            row.update(condition=job['condition'],actual_training_seed=seed,feedback_seed=feedback_seed,
                       observation=job['mode'],feedback_loss=job['loss'],episodes=job['episodes'],
                       implementation='v7_exact' if job['legacy'] else 'v8_corrected',source='new_execution',job_id=job['id'])
            rows.append(row)
    payload=dict(job=job,started_after_manifest=True,completed_utc=now(),seconds=time.perf_counter()-start,
                 rows=rows,diagnostics=diagnostics,curves=curves)
    path=BASE/'checkpoints'/f"{job['id']}.json"
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(payload,allow_nan=False),encoding='utf-8')
    temp.replace(path)
    return job['id'],payload['seconds']

def reuse(manifest):
    data=pd.read_csv(BASE.parent/'experiments_v7/outputs/tables/v7_main_replications.csv')
    for j in manifest['reused_jobs']:
        path=BASE/'checkpoints'/f"{j['id']}.json"
        if path.exists(): continue
        label=j['train_seed'] or 1101
        policies=LEARNED if j['train_seed'] else RULES
        subset=data[(data.environment==j['environment'])&(data.scenario==v7.SCENARIOS[j['scenario_index']].name)&
                    (data.train_seed==label)&data.policy.isin(policies)].copy()
        assert len(subset)==10*len(policies)
        rows=subset.to_dict('records')
        for row in rows:
            row.update(train_seed=j['train_seed'],condition=j['condition'],
                       actual_training_seed=actual_seed(label,j['scenario_index'],j['environment'],row['policy']) if j['train_seed'] else None,
                       observation='v7',feedback_loss=0.,episodes=180,implementation='v7_exact',
                       source='v7_saved_compatible',job_id=j['id'])
        path.write_text(json.dumps(dict(job=j,seconds=0,rows=rows,diagnostics=[],curves=[],
                                       completed_utc=now(),provenance='V7 saved CSV; targeted reproduction passed')),
                        encoding='utf-8')

def collect(manifest):
    rows=[];diag=[];curves=[];done=[]
    for job in manifest['jobs']+manifest['reused_jobs']:
        path=BASE/'checkpoints'/f"{job['id']}.json"
        if not path.exists(): continue
        p=json.loads(path.read_text(encoding='utf-8'))
        rows.extend(p['rows']);done.append(job['id'])
        meta={k:job[k] for k in ['condition','environment','train_seed','scenario_index']}
        diag.extend(dict(**meta,**d) for d in p['diagnostics'])
        curves.extend(dict(**meta,**d) for d in p['curves'])
    for name,items in [('V8_RESULTS.csv',rows),('V8_TRAINING_DIAGNOSTICS.csv',diag),('V8_TRAINING_CURVES.csv',curves)]:
        pd.DataFrame(items).to_csv(BASE/name,index=False)
    state=dict(updated_utc=now(),complete_jobs=len(done),planned_jobs=manifest['jobs_total'],
               result_rows=len(rows),complete_by_condition=dict(pd.DataFrame(rows).groupby('condition').size()) if rows else {},
               pending_jobs=[j['id'] for j in manifest['jobs']+manifest['reused_jobs'] if j['id'] not in set(done)])
    (BASE/'V8_RUN_STATUS.json').write_text(json.dumps(state,indent=2,default=int),encoding='utf-8')
    return state

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--collect',action='store_true')
    parser.add_argument('--stage',choices=['A','BC','all'],default='all')
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    if args.prepare: make_manifest();return
    manifest=json.loads((BASE/'V8_EXPERIMENT_MANIFEST.json').read_text(encoding='utf-8'))
    assert manifest['engine_sha256']==digest(BASE/'v8_engine.py'),'Engine changed since manifest'
    assert manifest['runner_sha256']==digest(Path(__file__)),'Runner changed since manifest'
    assert manifest['protocol_sha256']==digest(BASE/'V8_EXPERIMENTAL_PROTOCOL.md'),'Protocol changed since manifest'
    (BASE/'checkpoints').mkdir(exist_ok=True)
    reuse(manifest)
    if args.collect: print(json.dumps(collect(manifest),default=int));return
    pending=[j for j in manifest['jobs'] if (args.stage=='all' or j['stage']==args.stage) and
             not (BASE/'checkpoints'/f"{j['id']}.json").exists()]
    log=BASE/'campaign.jsonl'
    print(f'{now()} stage={args.stage} fresh_pending={len(pending)} workers={args.workers}',flush=True)
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures={executor.submit(run_job,j):j for j in pending}
        for n,future in enumerate(as_completed(futures),1):
            job=futures[future]
            try:
                ident,seconds=future.result()
                event=dict(time_utc=now(),job=ident,status='complete',seconds=seconds)
            except Exception:
                event=dict(time_utc=now(),job=job['id'],status='error',error=traceback.format_exc())
                with log.open('a',encoding='utf-8') as f: f.write(json.dumps(event)+'\n')
                raise
            with log.open('a',encoding='utf-8') as f: f.write(json.dumps(event)+'\n')
            if n%12==0 or n==len(pending):
                print(f'{now()} completed={n}/{len(pending)} elapsed_seconds={time.perf_counter()-start:.1f}',flush=True)
            if n%60==0: collect(manifest)
    state=collect(manifest)
    print(json.dumps({k:state[k] for k in ['complete_jobs','planned_jobs','result_rows','complete_by_condition']},default=int),flush=True)

if __name__=='__main__': main()
