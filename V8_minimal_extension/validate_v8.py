"""Behavioral verification, exact targeted reproduction and runtime benchmark."""
import sys
sys.dont_write_bytecode=True
import hashlib,json,subprocess,time,unittest
from pathlib import Path
import numpy as np
import pandas as pd
from v8_engine import v7,CorrectedEnv,observe,state_key,feedback_reward,train,evaluate,fast_discretize,ORIGINAL_DISCRETIZE
BASE=Path(__file__).resolve().parent
ROOT=BASE.parent

class Tests(unittest.TestCase):
    def test_fast_discretizer_matches_boundaries_and_random(self):
        rng=np.random.default_rng(817)
        observations=list(rng.random((1000,17)))
        for boundary in [.1,.2,.25,.33,.4,.5,.6,.66,.75,.8]:
            for x in [np.nextafter(boundary,-np.inf),boundary,np.nextafter(boundary,np.inf)]:
                observations.append(np.full(17,x))
                observations.append(np.full(17,x,dtype=np.float32))
        for obs in observations:
            self.assertEqual(fast_discretize(obs),ORIGINAL_DISCRETIZE(obs))

    def test_terminal_generated_incidents_are_paid(self):
        cfg=v7.ScenarioConfig(horizon=1,auto_sensitivity=0,human_sensitivity=0)
        legacy=v7.GovernanceEnvV7(cfg,'sequential',v7.FULL_SEQUENTIAL,seed=7)
        corrected=CorrectedEnv(cfg,'sequential',v7.FULL_SEQUENTIAL,seed=7)
        for env in [legacy,corrected]:
            env.review_queue.append((v7.Submission(.99,.1,True,.1,0),2))
            env.current_submission=v7.Submission(.1,.1,False,.8,0)
        _,a,_,_,_=legacy.step(0)
        _,b,_,_,_=corrected.step(0)
        self.assertGreater(len(legacy.incident_queue),0)
        self.assertEqual(len(corrected.incident_queue),0)
        self.assertLess(b,a)
        self.assertEqual(corrected.metrics['security_incidents'],1)

    def test_feedback_mask_does_not_change_dynamics(self):
        cfg=v7.ScenarioConfig(horizon=20)
        envs=[CorrectedEnv(cfg,'sequential',v7.FULL_SEQUENTIAL,seed=17) for _ in range(2)]
        generators=[np.random.default_rng(999) for _ in range(2)]
        for step in range(20):
            outputs=[]
            for env,rng,p in zip(envs,generators,[0.,.5]):
                result=env.step(2 if step%3 else 0)
                feedback_reward(result[-1]['reward_components'],'q_learning',p,rng)
                outputs.append(result)
            np.testing.assert_array_equal(outputs[0][0],outputs[1][0])
            self.assertEqual(outputs[0][1:4],outputs[1][1:4])
            for field in ['metrics','reward_components','executed_action','guard_applied']:
                self.assertEqual(outputs[0][-1][field],outputs[1][-1][field])
            np.testing.assert_array_equal(outputs[0][-1]['action_mask'],outputs[1][-1]['action_mask'])

    def test_observation_has_no_hidden_submission_or_future_leak(self):
        env=CorrectedEnv(v7.ScenarioConfig(),'sequential',v7.FULL_SEQUENTIAL,seed=18)
        before=observe(env,'enriched')
        env.current_submission.harmful=not env.current_submission.harmful
        env.current_submission.developer_quality=.999
        env.current_submission.diversity_group=3
        np.testing.assert_array_equal(before,observe(env,'enriched'))
        self.assertEqual(len(before),21)
        # Enrichment must change the effective tabular representation when due
        # workload appears; masked C2 channels remain zero.
        env.review_queue.append((env.current_submission,1))
        self.assertEqual(state_key(observe(env,'enriched'),'enriched')[-1],1)
        self.assertTrue(np.all(observe(env,'reduced')[[4,5,8,11,12,13,14,15,16]]==0))

    def test_terminal_q_update_has_no_bootstrap(self):
        # Horizon one, identical deterministic observation/reward across episodes.
        # All legal decisions forced to action 0. Q must approach its immediate
        # terminal reward, independent of gamma, rather than reward/(1-gamma).
        from unittest.mock import patch
        class Tiny(CorrectedEnv):
            def action_mask(self): return np.array([1,0,0,0,0,0],dtype=bool)
            def _draw_submission(self): return v7.Submission(.1,.1,False,.8,0)
        with patch('v8_engine.CorrectedEnv',Tiny):
            learner=train(v7.ScenarioConfig(horizon=1),'nonsequential',v7.NONSEQUENTIAL,
                          'q_learning',21,episodes=150)
        values=[v[0] for v in learner.values.values()]
        self.assertAlmostEqual(max(values),.8,places=7)

def main():
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    with (BASE/'validation_tests.log').open('w',encoding='utf-8') as handle:
        result=unittest.TextTestRunner(stream=handle,verbosity=2).run(suite)
    assert result.wasSuccessful(), 'V8 tests failed; read validation_tests.log'
    old=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','experiments_v7/tests','-v'],
                       cwd=ROOT,text=True,capture_output=True)
    (BASE/'v7_tests.log').write_text(old.stdout+old.stderr,encoding='utf-8')
    assert old.returncode==0
    saved=pd.read_csv(ROOT/'experiments_v7/outputs/tables/v7_main_replications.csv')
    rows=[]; timings=[]
    # Full-budget reproduction, one trained policy of each type, same scenario
    # and all ten evaluation seeds; deterministic rules across all twelve cells.
    for environment,mech in [('nonsequential',v7.NONSEQUENTIAL),('sequential',v7.FULL_SEQUENTIAL)]:
        for i,cfg in enumerate(v7.SCENARIOS):
            policies=['static_permissive','static_restrictive','security_rule','heuristic_score_threshold']
            if i==0 and environment=='sequential': policies+=['contextual_bandit','q_learning','linear_q_approx']
            offset=500000 if environment=='sequential' else 0
            for policy in policies:
                learner=None
                if policy in ['contextual_bandit','q_learning','linear_q_approx']:
                    add={'contextual_bandit':101,'q_learning':202,'linear_q_approx':404}[policy]
                    t=time.perf_counter()
                    learner=train(cfg,environment,mech,policy,v7.SEED+offset+1101+add,legacy=True)
                    timings.append(dict(kind='legacy',policy=policy,episodes=180,seconds=time.perf_counter()-t))
                for e in v7.MAIN_EVAL_SEEDS:
                    seed=v7.SEED+offset+i*10000+e
                    obtained=evaluate(cfg,environment,mech,policy,seed,1101,learner,legacy=True)
                    expected=saved[(saved.environment==environment)&(saved.scenario==cfg.name)&
                                   (saved.policy==policy)&(saved.train_seed==1101)&(saved.eval_seed==seed)].iloc[0]
                    metrics=[k for k in obtained if k in expected.index and isinstance(obtained[k],(int,float))]
                    error=max(abs(obtained[k]-float(expected[k])) for k in metrics)
                    rows.append(dict(environment=environment,scenario=cfg.name,policy=policy,eval_seed=seed,
                                     max_absolute_error=error,matched=error<1e-10,
                                     residual_pending_incidents=obtained['residual_pending_incidents']))
                    assert error<1e-10,(environment,cfg.name,policy,error)
    for policy in ['contextual_bandit','q_learning','linear_q_approx']:
        t=time.perf_counter()
        learner=train(v7.SCENARIOS[0],'sequential',v7.FULL_SEQUENTIAL,policy,100000101,episodes=20)
        timings.append(dict(kind='corrected',policy=policy,episodes=20,seconds=time.perf_counter()-t))
        a=evaluate(v7.SCENARIOS[0],'sequential',v7.FULL_SEQUENTIAL,policy,20266109,1,learner)
        b=evaluate(v7.SCENARIOS[0],'sequential',v7.FULL_SEQUENTIAL,policy,20266109,1,learner)
        assert a==b
        assert a['residual_pending_items']==0
    pd.DataFrame(rows).to_csv(BASE/'V7_TARGETED_REPRODUCTION.csv',index=False)
    pd.DataFrame(timings).to_csv(BASE/'RUNTIME_BENCHMARK.csv',index=False)
    protected=json.loads((BASE/'V7_PRESERVATION_SHA256.json').read_text())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in protected.items())
    print(json.dumps(dict(v8_tests=result.testsRun,v7_tests='passed',reproduced_evaluations=len(rows),
                          max_error=max(r['max_absolute_error'] for r in rows),timings=timings),indent=2))

if __name__=='__main__': main()
