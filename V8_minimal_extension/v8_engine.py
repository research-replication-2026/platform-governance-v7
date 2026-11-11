"""V8 extensions; all simulation mechanics inherited from a frozen V7 copy."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import math
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'vendor'))
import v7_frozen as v7

ORIGINAL_DISCRETIZE = v7.discretize_state
BINS = {
    0: [.20,.40,.60,.80], 1: [.25,.50,.75], 2: [.33,.66],
    4: [.25,.60], 5: [.25,.60], 6: [.20,.50], 7: [.10,.40],
    8: [.10,.40], 10: [.33,.66], 11: [.20,.60], 14: [.10,.40],
}

def fast_discretize(obs, mode='adjusted'):
    if mode != 'adjusted':
        return ORIGINAL_DISCRETIZE(obs, mode)
    return tuple(bisect_right(bins, float(obs[i])) for i,bins in BINS.items())

# Exact scalar replacement only; boundary equivalence is tested before production.
v7.discretize_state = fast_discretize

class CorrectedEnv(v7.GovernanceEnvV7):
    def _liquidate_terminal(self):
        reward = super()._liquidate_terminal()
        # Resolution of terminal checks/reviews can itself schedule incidents.
        while self.incident_queue:
            _, severity = self.incident_queue.popleft()
            self.metrics['v8_terminal_generated_incidents'] += 1
            reward += self._materialize_incident(severity)
        return reward

def observe(env, mode):
    obs = env._state().copy()
    if mode == 'reduced':
        # Keep positions stable for rules and linear weights; mask all load/capacity
        # channels. Keep the legal-action mask: constraints are identical.
        obs[[4,5,8,11,12,13,14,15,16]] = 0
    elif mode == 'enriched':
        # Only administrative schedules of decisions already made; no incident
        # schedule, severity, true harmfulness, quality, future arrivals or rewards.
        cfg = env.config
        auto_due = sum(delay <= 1 for _,delay in env.auto_queue)
        review_due = sum(delay <= 1 for _,delay in env.review_queue)
        ages = [sub.age for sub,_ in env.auto_queue] + [sub.age for sub,_ in env.review_queue]
        extra = [env.t / cfg.horizon, min(auto_due / cfg.max_review_queue,1),
                 min(review_due / cfg.max_review_queue,1),
                 min(max(ages, default=0) / cfg.horizon,1)]
        obs = np.concatenate([obs,np.asarray(extra,dtype=np.float32)])
    elif mode != 'v7':
        raise ValueError(mode)
    return obs

def state_key(obs, mode):
    # Enrichment adds one binary workload-timing channel (at most doubles the
    # tabular state space); the other new raw channels are available to linear Q.
    key = fast_discretize(obs)
    if mode == 'enriched':
        key += (int(obs[18] > 0 or obs[19] > 0),)
    return key

def feedback_reward(components, policy, loss, rng):
    # One independently randomized availability flag per component and transition,
    # shared masking schedule across algorithms/conditions through a separate RNG.
    # Missing scalar signals are encoded as zero, without inverse-probability
    # scaling; no state or true objective is changed. All flags drawn even at p=0.
    keys = ['current_action_reward','delayed_past_reward','collective_state_reward',
            'terminal_liquidation_reward']
    keep = rng.random(4) >= loss
    reward = float(components['current_action_reward']) * keep[0]
    if policy != 'contextual_bandit':
        reward += sum(float(components[k])*keep[i] for i,k in enumerate(keys) if i>0)
        reward += float(components['guard_penalty']) # known enforcement cost
    return float(reward), int((~keep).sum())

def train(config, environment, mechanisms, policy, actual_seed, episodes=180,
          mode='v7', loss=0.0, legacy=False, feedback_seed=1):
    if legacy:
        fn = {'contextual_bandit':v7.train_contextual_bandit,
              'q_learning':v7.train_q_learning,'linear_q_approx':v7.train_linear_q}[policy]
        return fn(config,environment,mechanisms,actual_seed,episodes=episodes)
    rng = np.random.default_rng(actual_seed)
    feedback_rng = np.random.default_rng(feedback_seed)
    q = defaultdict(lambda: np.zeros(6,dtype=float))
    counts = defaultdict(lambda: np.zeros(6,dtype=float))
    visits,sa = Counter(),Counter()
    weights = np.zeros((6,22 if mode=='enriched' else 18))
    curve=[]
    multiplier={'contextual_bandit':13,'q_learning':17,'linear_q_approx':19}[policy]
    masked_count=0
    for ep in range(episodes):
        seed=actual_seed+ep*multiplier
        env=CorrectedEnv(config,environment,mechanisms,seed=seed)
        # V7 constructs and explicitly resets twice with the same seed.
        _,info=env.reset(seed=seed)
        obs=observe(env,mode)
        total=0.; used=0.
        epsilon=.04+.24*math.exp(-ep/max(1,episodes/3))
        while True:
            key=state_key(obs,mode)
            visits[key]+=1
            x=v7.feature_vector(obs)
            values=weights @ x if policy=='linear_q_approx' else q[key]
            action=v7.choose_epsilon_greedy(values,info['action_mask'],rng,epsilon)
            _,reward,terminated,truncated,next_info=env.step(action)
            next_obs=observe(env,mode)
            done=terminated or truncated
            update,dropped=feedback_reward(next_info['reward_components'],policy,loss,feedback_rng)
            masked_count+=dropped
            sa[(key,action)]+=1
            if policy=='contextual_bandit':
                counts[key][action]+=1
                q[key][action]+=(update-q[key][action])/counts[key][action]
            elif policy=='q_learning':
                future=0. if done else np.max(np.where(next_info['action_mask'],q[state_key(next_obs,mode)],-1e12))
                q[key][action]+=.14*(update+.93*future-q[key][action])
            else:
                next_x=v7.feature_vector(next_obs)
                future=0. if done else np.max(np.where(next_info['action_mask'],weights@next_x,-1e12))
                td=np.clip(update+.93*future-values[action],-5.,5.)
                weights[action]+=.035*td*x
            total+=reward; used+=update
            obs,info=next_obs,next_info
            if done: break
        if ep % max(1,episodes//12)==0 or ep==episodes-1:
            curve.append(dict(episode=ep,total_reward=total,update_reward=used,epsilon=epsilon))
    diag=v7.diagnostics_from_visits(None,visits,sa,curve,episodes)
    diag.update(feedback_missing_components=masked_count,feedback_component_draws=episodes*config.horizon*4,
                raw_dimensions=21 if mode=='enriched' else 17,
                effective_tabular_dimensions=12 if mode=='enriched' else 11,
                terminal_bootstrap_zero=True)
    return v7.LearnedPolicy(policy,weights if policy=='linear_q_approx' else q,visits,sa,curve,diag)

def evaluate(config,environment,mechanisms,policy,seed,train_seed,learned=None,mode='v7',legacy=False):
    env=(v7.GovernanceEnvV7 if legacy else CorrectedEnv)(config,environment,mechanisms,seed=seed)
    _,info=env.reset(seed=seed)
    obs=observe(env,mode)
    total=0.; components=Counter(); unseen=0
    for _ in range(config.horizon):
        if learned is None:
            action=v7.policy_action(policy,obs,info['action_mask'])
        else:
            key=state_key(obs,mode)
            unseen+=key not in learned.state_visits
            values=learned.values@v7.feature_vector(obs) if policy=='linear_q_approx' else learned.values[key]
            action=v7.masked_argmax(values,info['action_mask'])
        _,reward,_,_,info=env.step(action)
        total+=reward
        components.update(info['reward_components'])
        obs=observe(env,mode)
    row=env.final_metrics()
    row.update(environment=environment,mechanism=mechanisms.name,scenario=config.name,
               policy=policy,train_seed=train_seed,eval_seed=seed,total_reward=total,
               mean_reward=total/config.horizon,unseen_eval_state_rate=unseen/config.horizon if learned else 0.,
               residual_pending_incidents=len(env.incident_queue),
               residual_pending_items=sum(map(len,[env.incident_queue,env.auto_queue,env.review_queue,env.appeal_queue])),
               terminal_generated_incidents=env.metrics['v8_terminal_generated_incidents'])
    row.update({k+'_total':v for k,v in components.items()})
    row.update({f'action_{name}':env.metrics[f'action_{name}'] for name in v7.ACTION_NAMES.values()})
    return row
