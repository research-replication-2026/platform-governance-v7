"""Prospectively specified nested-budget check; original V7/V8 remain read-only."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import argparse,copy,hashlib,itertools,json,math,os,time,traceback
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'FROZEN_SIMULATOR'))
import v8_engine as engine
v7=engine.v7
SEEDS=list(range(10001,10031))
POLICIES=['contextual_bandit','q_learning','linear_q_approx']
BUDGETS=[180,360,720]

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def actual_seed(label,index,policy):
    return 100000000+(label-10001)*1000000+500000+index*10000+{'contextual_bandit':101,'q_learning':202,'linear_q_approx':404}[policy]

def prepare():
    path=BASE/'BUDGET_MANIFEST.json'
    if path.exists():
        item=json.loads(path.read_text())
        assert item['runner_sha256']==digest(Path(__file__))
        assert item['protocol_sha256']==digest(BASE/'V8_PROTOCOL.md')
        return item
    jobs=[dict(scenario_index=i,train_seed=s,id=f's{i}_t{s}') for i in [0,1] for s in SEEDS]
    audits=[]
    for i in [0,1]:
        streams=[]
        for s in SEEDS:
            united=set()
            for p,stride in zip(POLICIES,[13,17,19]): united.update(actual_seed(s,i,p)+e*stride for e in range(720))
            streams.append(united)
        shared=sum(len(a&b) for a,b in itertools.combinations(streams,2))
        evals={v7.SEED+500000+i*10000+e for e in v7.MAIN_EVAL_SEEDS}
        overlap=len(set.union(*streams)&evals)
        assert shared==overlap==0
        audits.append(dict(scenario=v7.SCENARIOS[i].name,cross_block_seed_overlap=shared,train_eval_overlap=overlap))
    pd.DataFrame(audits).to_csv(BASE/'BUDGET_SEED_AUDIT.csv',index=False)
    result=dict(created_utc=now(),status='prospective_exploratory_specification',jobs=jobs,
                budgets=BUDGETS,training_blocks=SEEDS,policies=POLICIES,epsilon_decay_episodes=60,
                runner_sha256=digest(Path(__file__)),protocol_sha256=digest(BASE/'V8_PROTOCOL.md'),
                frozen_engine_sha256=digest(BASE/'FROZEN_SIMULATOR/v8_engine.py'),
                frozen_vendor_sha256=digest(BASE/'FROZEN_SIMULATOR/vendor/v7_frozen.py'),
                original_primary_results_sha256=digest(BASE/'V8_RESULTS.csv'),cpu_logical=os.cpu_count(),
                full_720_episode_policy_trajectories=180,snapshot_evaluations=5400)
    path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

def train_snapshots(config,policy,seed,label,index):
    rng=np.random.default_rng(seed)
    feedback_rng=np.random.default_rng(4000000000+index*10000+(label-10001)+3)
    q=defaultdict(lambda:np.zeros(6,dtype=float))
    counts=defaultdict(lambda:np.zeros(6,dtype=float))
    visits,sa=Counter(),Counter();weights=np.zeros((6,18))
    stride={'contextual_bandit':13,'q_learning':17,'linear_q_approx':19}[policy]
    rows=[];diagnostics=[];curves=[]
    for ep in range(720):
        episode_seed=seed+ep*stride
        env=engine.CorrectedEnv(config,'sequential',v7.FULL_SEQUENTIAL,seed=episode_seed)
        _,info=env.reset(seed=episode_seed);obs=engine.observe(env,'v7')
        epsilon=.04+.24*math.exp(-ep/60)
        total=0.
        while True:
            key=engine.state_key(obs,'v7');visits[key]+=1
            x=v7.feature_vector(obs)
            values=weights@x if policy=='linear_q_approx' else q[key]
            action=v7.choose_epsilon_greedy(values,info['action_mask'],rng,epsilon)
            _,reward,terminated,truncated,next_info=env.step(action)
            next_obs=engine.observe(env,'v7');done=terminated or truncated
            update,_=engine.feedback_reward(next_info['reward_components'],policy,0.,feedback_rng)
            sa[(key,action)]+=1
            if policy=='contextual_bandit':
                counts[key][action]+=1
                q[key][action]+=(update-q[key][action])/counts[key][action]
            elif policy=='q_learning':
                future=0. if done else np.max(np.where(next_info['action_mask'],q[engine.state_key(next_obs,'v7')],-1e12))
                q[key][action]+=.14*(update+.93*future-q[key][action])
            else:
                next_x=v7.feature_vector(next_obs)
                future=0. if done else np.max(np.where(next_info['action_mask'],weights@next_x,-1e12))
                td=np.clip(update+.93*future-values[action],-5.,5.)
                weights[action]+=.035*td*x
            total+=reward;obs,info=next_obs,next_info
            if done: break
        if ep%15==0 or ep+1 in BUDGETS:
            curves.append(dict(policy=policy,episode=ep,total_reward=total,epsilon=epsilon))
        if ep+1 in BUDGETS:
            budget=ep+1
            values=copy.deepcopy(weights if policy=='linear_q_approx' else q)
            learner=v7.LearnedPolicy(policy,values,visits.copy(),sa.copy(),[],{})
            diagnostics.append(dict(policy=policy,budget=budget,visited_states=len(visits),
                                    visited_state_actions=len(sa),state_action_coverage=len(sa)/(len(visits)*6)))
            for nominal in v7.MAIN_EVAL_SEEDS:
                eval_seed=v7.SEED+500000+index*10000+nominal
                row=engine.evaluate(config,'sequential',v7.FULL_SEQUENTIAL,policy,eval_seed,label,learner)
                row.update(budget=budget,actual_training_seed=seed,condition='exploratory_nested_budget',
                           epsilon_decay_episodes=60,training_trajectory=f's{index}_t{label}_{policy}')
                assert row['residual_pending_items']==0
                rows.append(row)
    return rows,diagnostics,curves

def job_work(job):
    start=time.perf_counter();i=job['scenario_index'];label=job['train_seed'];config=v7.SCENARIOS[i]
    rows=[];diagnostics=[];curves=[]
    for policy in POLICIES:
        a,b,c=train_snapshots(config,policy,actual_seed(label,i,policy),label,i)
        rows.extend(a);diagnostics.extend(b);curves.extend(c)
    existing=pd.read_csv(BASE/'V8_RESULTS.csv')
    expected=existing[(existing.condition=='A_corrected')&(existing.environment=='sequential')&
                      (existing.scenario==config.name)&(existing.train_seed==label)].set_index(['policy','eval_seed'])
    max_error=0.
    for row in rows:
        if row['budget']!=180: continue
        reference=expected.loc[(row['policy'],row['eval_seed'])]
        for key,value in row.items():
            if key in reference and isinstance(value,(int,float,np.number)):
                max_error=max(max_error,abs(float(value)-float(reference[key])))
    assert max_error<1e-10,('180-episode prefix mismatch',job,max_error)
    payload=dict(job=job,completed_utc=now(),seconds=time.perf_counter()-start,
                 prefix_max_error=max_error,rows=rows,diagnostics=diagnostics,curves=curves)
    destination=BASE/'BUDGET_CHECKPOINTS'/f"{job['id']}.json"
    temporary=destination.with_suffix('.tmp');temporary.write_text(json.dumps(payload),encoding='utf-8');temporary.replace(destination)
    return job['id'],payload['seconds'],max_error

def collect(manifest):
    rows=[];diagnostics=[];curves=[];complete=[]
    for job in manifest['jobs']:
        path=BASE/'BUDGET_CHECKPOINTS'/f"{job['id']}.json"
        if not path.exists(): continue
        p=json.loads(path.read_text());complete.append(job['id']);rows.extend(p['rows'])
        meta=dict(scenario=v7.SCENARIOS[job['scenario_index']].name,train_seed=job['train_seed'])
        diagnostics.extend(dict(**meta,**d) for d in p['diagnostics'])
        curves.extend(dict(**meta,**d) for d in p['curves'])
    for name,content in [('BUDGET_RESULTS.csv',rows),('BUDGET_DIAGNOSTICS.csv',diagnostics),('BUDGET_TRAINING_CURVES.csv',curves)]:
        pd.DataFrame(content).to_csv(BASE/name,index=False)
    result=dict(updated_utc=now(),completed_jobs=len(complete),planned_jobs=60,evaluation_rows=len(rows),
                pending=[j['id'] for j in manifest['jobs'] if j['id'] not in complete])
    (BASE/'BUDGET_STATUS.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--pilot',action='store_true');parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args();manifest=prepare();(BASE/'BUDGET_CHECKPOINTS').mkdir(exist_ok=True)
    if args.prepare: print(json.dumps({k:manifest[k] for k in ['created_utc','cpu_logical','full_720_episode_policy_trajectories','snapshot_evaluations']},indent=2));return
    pending=[j for j in manifest['jobs'] if not (BASE/'BUDGET_CHECKPOINTS'/f"{j['id']}.json").exists()]
    if args.pilot:
        result=job_work(pending[0]) if pending else ('already_complete',0,0)
        (BASE/'BUDGET_RUNTIME_BENCHMARK.json').write_text(json.dumps(dict(job=result[0],seconds=result[1],prefix_error=result[2],
             estimated_serial_seconds_for_60_jobs=result[1]*60),indent=2),encoding='utf-8')
        print(json.dumps(dict(pilot=result,status=collect(manifest)),indent=2));return
    assert (BASE/'BUDGET_RUNTIME_BENCHMARK.json').exists(),'Run prefix-verification pilot before full campaign'
    print(f'Nested-budget jobs pending={len(pending)} workers={args.workers}',flush=True)
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(job_work,j):j for j in pending}
        for n,future in enumerate(as_completed(futures),1):
            job=futures[future]
            try:
                name,seconds,error=future.result()
                event=dict(job=name,time_utc=now(),seconds=seconds,prefix_error=error,status='complete')
            except Exception:
                event=dict(job=job['id'],time_utc=now(),status='error',error=traceback.format_exc())
                with (BASE/'BUDGET_EXECUTION.jsonl').open('a',encoding='utf-8') as stream: stream.write(json.dumps(event)+'\n')
                raise
            with (BASE/'BUDGET_EXECUTION.jsonl').open('a',encoding='utf-8') as stream: stream.write(json.dumps(event)+'\n')
            if n%6==0 or n==len(pending):
                print(f'Budget jobs {n}/{len(pending)} elapsed={time.perf_counter()-start:.1f}s',flush=True);collect(manifest)
    print(json.dumps(collect(manifest)),flush=True)

if __name__=='__main__': main()
