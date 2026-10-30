# V7 Scientific Validation of RL-Specific Value

## Scientific status

The documentary governance-event register contains 32 coded governance events, including 20 Google Play events and 5 direct ML-governance events. V7 still uses these events only to motivate mechanisms, not to estimate transition probabilities.

V7 corrects the V6 reward-attribution problem by separating current-action reward from delayed consequences of past decisions and terminal liquidation of unresolved obligations.

## Direct answers

1. Does the V6 Q-learning disadvantage reflect learning failure or absence of sequential gain?

In the six main scenarios, Q-learning or another learned RL variant is the top policy in 0/6 sequential scenarios. Q-learning beats corrected contextual bandit in 0/6 sequential paired mean differences, with strictly positive 95% CIs in 0/6.
Best Q-learning diagnostic setting: high_security_pressure, budget=360, gamma=0.85, alpha=0.08, eval mean reward=0.5404, unseen-state rate=0.0000.
The result should be interpreted jointly with coverage diagnostics. If coverage is sparse and performance remains below the corrected bandit, V7 supports both a learning-capacity limitation and weak evidence of robust sequential gain in the current simulator.

2. Does RL benefit more from delayed consequences than contextual bandit?

In the non-sequential environment, learned RL variants are top in 0/6 scenarios. Q-learning beats corrected contextual bandit in 0/6 non-sequential paired differences, with strictly positive 95% CIs in 0/6.
The corrected contextual bandit no longer trains on delayed-past reward. Any remaining Q-learning advantage is therefore attributable to total-return learning rather than reward leakage.

3. Which mechanisms explain the differences?

Largest average Q-learning reward loss after ablation: without_trust_effects (0.0539).
Use the ablation tables for mechanism-specific interpretation; full rankings alone are insufficient.

4. Is there a complexity or temporal-dependence threshold where sequential learning becomes useful?

The largest Q-learning minus bandit interaction appears at seq_scale_0_5: -0.0811.
The sequentiality-scale experiment estimates this interaction directly rather than inferring it from policy ranks.

5. Are the results robust?

Robustness is limited to the simulated parameter grid. The oracle upper bound is included only as a non-deployable theoretical ceiling with hidden information.

## Sequential winners

| scenario | policy | mean_reward_mean | security_incidents_mean | governance_expenditure_mean |
| --- | --- | --- | --- | --- |
| balanced_reference | oracle_upper_bound | 0.557128 | 0.2 | 8.28 |
| constrained_review_capacity | oracle_upper_bound | 0.553184 | 0.4 | 9.626 |
| high_false_restriction_cost | oracle_upper_bound | 0.541951 | 0.3 | 8.064 |
| high_review_capacity | oracle_upper_bound | 0.576962 | 0.2 | 7.461 |
| high_security_pressure | contextual_bandit | 0.606487 | 0.666667 | 5.444 |
| participation_sensitive_ecosystem | contextual_bandit | 0.556533 | 0.866667 | 3.707667 |

## Non-sequential winners

| scenario | policy | mean_reward_mean | security_incidents_mean | governance_expenditure_mean |
| --- | --- | --- | --- | --- |
| balanced_reference | oracle_upper_bound | 0.565619 | 0.3 | 7.395 |
| constrained_review_capacity | contextual_bandit | 0.571952 | 0.4 | 5.448 |
| high_false_restriction_cost | contextual_bandit | 0.542382 | 0.5 | 6.391333 |
| high_review_capacity | contextual_bandit | 0.577379 | 0.3 | 5.35 |
| high_security_pressure | contextual_bandit | 0.615958 | 0.6 | 6.780667 |
| participation_sensitive_ecosystem | contextual_bandit | 0.562028 | 0.333333 | 5.534667 |

## Key paired differences

| environment | scenario | comparison | mean_diff | ci95_low | ci95_high | n_pairs |
| --- | --- | --- | --- | --- | --- | --- |
| nonsequential | balanced_reference | q_learning_minus_contextual_bandit | -0.039476 | -0.059765 | -0.019186 | 30 |
| nonsequential | balanced_reference | q_learning_minus_security_rule | 0.026238 | -0.003542 | 0.056017 | 30 |
| nonsequential | balanced_reference | q_learning_minus_heuristic_score_threshold | 0.033257 | 0.008358 | 0.058157 | 30 |
| nonsequential | constrained_review_capacity | q_learning_minus_contextual_bandit | -0.046114 | -0.071972 | -0.020257 | 30 |
| nonsequential | constrained_review_capacity | q_learning_minus_security_rule | 0.054017 | 0.025732 | 0.082303 | 30 |
| nonsequential | constrained_review_capacity | q_learning_minus_heuristic_score_threshold | 0.00725 | -0.021919 | 0.036419 | 30 |
| nonsequential | high_false_restriction_cost | q_learning_minus_contextual_bandit | -0.050381 | -0.077244 | -0.023519 | 30 |
| nonsequential | high_false_restriction_cost | q_learning_minus_security_rule | 0.027401 | 0.001187 | 0.053614 | 30 |
| nonsequential | high_false_restriction_cost | q_learning_minus_heuristic_score_threshold | -0.0165 | -0.038048 | 0.005048 | 30 |
| nonsequential | high_review_capacity | q_learning_minus_contextual_bandit | -0.049421 | -0.069401 | -0.029441 | 30 |
| nonsequential | high_review_capacity | q_learning_minus_security_rule | 0.052674 | 0.023261 | 0.082088 | 30 |
| nonsequential | high_review_capacity | q_learning_minus_heuristic_score_threshold | 0.081275 | 0.051875 | 0.110675 | 30 |
| nonsequential | high_security_pressure | q_learning_minus_contextual_bandit | -0.056961 | -0.077942 | -0.035979 | 30 |
| nonsequential | high_security_pressure | q_learning_minus_security_rule | 0.065131 | 0.038136 | 0.092126 | 30 |
| nonsequential | high_security_pressure | q_learning_minus_heuristic_score_threshold | 0.128255 | 0.080477 | 0.176033 | 30 |
| nonsequential | participation_sensitive_ecosystem | q_learning_minus_contextual_bandit | -0.075972 | -0.09642 | -0.055524 | 30 |
| nonsequential | participation_sensitive_ecosystem | q_learning_minus_security_rule | 0.008311 | -0.017119 | 0.033741 | 30 |
| nonsequential | participation_sensitive_ecosystem | q_learning_minus_heuristic_score_threshold | 0.005946 | -0.01369 | 0.025582 | 30 |
| sequential | balanced_reference | q_learning_minus_contextual_bandit | -0.04949 | -0.083599 | -0.015381 | 30 |
| sequential | balanced_reference | q_learning_minus_security_rule | 0.000418 | -0.031021 | 0.031858 | 30 |
| sequential | balanced_reference | q_learning_minus_heuristic_score_threshold | -0.030242 | -0.06365 | 0.003166 | 30 |
| sequential | constrained_review_capacity | q_learning_minus_contextual_bandit | -0.092927 | -0.123267 | -0.062587 | 30 |
| sequential | constrained_review_capacity | q_learning_minus_security_rule | -0.030772 | -0.074124 | 0.01258 | 30 |
| sequential | constrained_review_capacity | q_learning_minus_heuristic_score_threshold | -0.043213 | -0.081576 | -0.00485 | 30 |
| sequential | high_false_restriction_cost | q_learning_minus_contextual_bandit | -0.038593 | -0.090759 | 0.013572 | 30 |
| sequential | high_false_restriction_cost | q_learning_minus_security_rule | -0.003357 | -0.051512 | 0.044798 | 30 |
| sequential | high_false_restriction_cost | q_learning_minus_heuristic_score_threshold | -0.037292 | -0.093592 | 0.019008 | 30 |
| sequential | high_review_capacity | q_learning_minus_contextual_bandit | -0.061563 | -0.105971 | -0.017156 | 30 |
| sequential | high_review_capacity | q_learning_minus_security_rule | -0.059488 | -0.094707 | -0.02427 | 30 |
| sequential | high_review_capacity | q_learning_minus_heuristic_score_threshold | -0.041812 | -0.075888 | -0.007736 | 30 |
| sequential | high_security_pressure | q_learning_minus_contextual_bandit | -0.14235 | -0.176454 | -0.108246 | 30 |
| sequential | high_security_pressure | q_learning_minus_security_rule | 0.008772 | -0.037566 | 0.055109 | 30 |
| sequential | high_security_pressure | q_learning_minus_heuristic_score_threshold | 0.043634 | 0.012407 | 0.074862 | 30 |
| sequential | participation_sensitive_ecosystem | q_learning_minus_contextual_bandit | -0.082781 | -0.110637 | -0.054925 | 30 |
| sequential | participation_sensitive_ecosystem | q_learning_minus_security_rule | 0.044586 | -0.002275 | 0.091448 | 30 |
| sequential | participation_sensitive_ecosystem | q_learning_minus_heuristic_score_threshold | 0.010844 | -0.024104 | 0.045792 | 30 |
| nonsequential | balanced_reference | q_learning_fine_minus_contextual_bandit | -0.071259 | -0.092227 | -0.050291 | 30 |
| nonsequential | constrained_review_capacity | q_learning_fine_minus_contextual_bandit | -0.07119 | -0.091559 | -0.050821 | 30 |
| nonsequential | high_false_restriction_cost | q_learning_fine_minus_contextual_bandit | -0.082465 | -0.113284 | -0.051645 | 30 |
| nonsequential | high_review_capacity | q_learning_fine_minus_contextual_bandit | -0.089629 | -0.11056 | -0.068698 | 30 |
| nonsequential | high_security_pressure | q_learning_fine_minus_contextual_bandit | -0.089955 | -0.122655 | -0.057254 | 30 |
| nonsequential | participation_sensitive_ecosystem | q_learning_fine_minus_contextual_bandit | -0.089997 | -0.117896 | -0.062097 | 30 |
| sequential | balanced_reference | q_learning_fine_minus_contextual_bandit | -0.179924 | -0.21969 | -0.140157 | 30 |
| sequential | constrained_review_capacity | q_learning_fine_minus_contextual_bandit | -0.154327 | -0.207902 | -0.100752 | 30 |
| sequential | high_false_restriction_cost | q_learning_fine_minus_contextual_bandit | -0.089483 | -0.147121 | -0.031844 | 30 |
| sequential | high_review_capacity | q_learning_fine_minus_contextual_bandit | -0.109511 | -0.157768 | -0.061255 | 30 |
| sequential | high_security_pressure | q_learning_fine_minus_contextual_bandit | -0.331233 | -0.38937 | -0.273096 | 30 |
| sequential | participation_sensitive_ecosystem | q_learning_fine_minus_contextual_bandit | -0.232653 | -0.268822 | -0.196484 | 30 |
| nonsequential | balanced_reference | linear_q_approx_minus_contextual_bandit | -0.017833 | -0.035891 | 0.000226 | 30 |
| nonsequential | constrained_review_capacity | linear_q_approx_minus_contextual_bandit | -0.024936 | -0.044486 | -0.005387 | 30 |
| nonsequential | high_false_restriction_cost | linear_q_approx_minus_contextual_bandit | -0.055021 | -0.088624 | -0.021418 | 30 |
| nonsequential | high_review_capacity | linear_q_approx_minus_contextual_bandit | -0.040496 | -0.056685 | -0.024307 | 30 |
| nonsequential | high_security_pressure | linear_q_approx_minus_contextual_bandit | -0.108269 | -0.140302 | -0.076236 | 30 |
| nonsequential | participation_sensitive_ecosystem | linear_q_approx_minus_contextual_bandit | -0.004577 | -0.03063 | 0.021477 | 30 |
| sequential | balanced_reference | linear_q_approx_minus_contextual_bandit | -0.142716 | -0.182294 | -0.103138 | 30 |
| sequential | constrained_review_capacity | linear_q_approx_minus_contextual_bandit | 0.013983 | -0.012393 | 0.040358 | 30 |
| sequential | high_false_restriction_cost | linear_q_approx_minus_contextual_bandit | 0.018444 | -0.037432 | 0.07432 | 30 |
| sequential | high_review_capacity | linear_q_approx_minus_contextual_bandit | -0.132759 | -0.214557 | -0.05096 | 30 |
| sequential | high_security_pressure | linear_q_approx_minus_contextual_bandit | -0.029005 | -0.058354 | 0.000344 | 30 |
| sequential | participation_sensitive_ecosystem | linear_q_approx_minus_contextual_bandit | -0.244172 | -0.35336 | -0.134985 | 30 |

## Q-learning convergence diagnostics

| scenario | budget | hparam_id | alpha | gamma | visited_states | visited_state_actions | state_action_coverage_within_seen_states | eval_mean_reward | eval_unseen_state_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high_security_pressure | 360 | 0 | 0.08 | 0.85 | 896 | 2367 | 0.44029 | 0.540377 | 0 |
| high_security_pressure | 360 | 1 | 0.14 | 0.93 | 1024 | 2578 | 0.419596 | 0.540361 | 0.003333 |
| high_security_pressure | 120 | 2 | 0.22 | 0.97 | 759 | 1792 | 0.3935 | 0.539907 | 0 |
| high_security_pressure | 360 | 1 | 0.14 | 0.93 | 1088 | 2643 | 0.404871 | 0.512828 | 0 |
| balanced_reference | 360 | 0 | 0.08 | 0.85 | 875 | 2280 | 0.434286 | 0.510651 | 0 |
| high_security_pressure | 360 | 2 | 0.22 | 0.97 | 1178 | 2924 | 0.413696 | 0.504705 | 0 |
| high_security_pressure | 360 | 0 | 0.08 | 0.85 | 920 | 2413 | 0.437138 | 0.502141 | 0 |
| high_security_pressure | 40 | 0 | 0.08 | 0.85 | 544 | 1140 | 0.349265 | 0.48825 | 0.018333 |
| high_security_pressure | 120 | 1 | 0.14 | 0.93 | 817 | 1953 | 0.398409 | 0.486808 | 0 |
| balanced_reference | 120 | 2 | 0.22 | 0.97 | 776 | 1779 | 0.382088 | 0.478431 | 0.005 |
| balanced_reference | 40 | 1 | 0.14 | 0.93 | 525 | 1016 | 0.32254 | 0.477044 | 0.013333 |
| balanced_reference | 40 | 1 | 0.14 | 0.93 | 551 | 1070 | 0.323654 | 0.475417 | 0.003333 |

## Sequentiality-scale interaction

| mechanism | policy | mean_reward_mean | security_incidents_mean | governance_expenditure_mean |
| --- | --- | --- | --- | --- |
| seq_scale_0 | contextual_bandit | 0.558686 | 0.4375 | 6.752708 |
| seq_scale_0 | linear_q_approx | 0.513203 | 1.291667 | 6.500625 |
| seq_scale_0 | q_learning | 0.472864 | 1.0625 | 9.623333 |
| seq_scale_0 | q_learning_fine | 0.491257 | 1.416667 | 8.722917 |
| seq_scale_0_5 | contextual_bandit | 0.513605 | 1.416667 | 5.679792 |
| seq_scale_0_5 | linear_q_approx | 0.487987 | 2.1875 | 5.377083 |
| seq_scale_0_5 | q_learning | 0.432486 | 2.583333 | 7.568125 |
| seq_scale_0_5 | q_learning_fine | 0.328099 | 4.791667 | 6.715208 |
| seq_scale_1 | contextual_bandit | 0.511901 | 1.583333 | 5.624375 |
| seq_scale_1 | linear_q_approx | 0.470979 | 2.9375 | 4.221875 |
| seq_scale_1 | q_learning | 0.423944 | 2.5 | 7.984583 |
| seq_scale_1 | q_learning_fine | 0.233248 | 6.458333 | 7.384375 |
| seq_scale_1_5 | contextual_bandit | 0.475097 | 2.125 | 5.637292 |
| seq_scale_1_5 | linear_q_approx | 0.398125 | 4.125 | 4.45625 |
| seq_scale_1_5 | q_learning | 0.390281 | 3 | 8.720833 |
| seq_scale_1_5 | q_learning_fine | 0.136612 | 7.666667 | 7.455 |

## Reward attribution excerpt

| environment | scenario | policy | current_action_reward_total_mean | delayed_past_reward_total_mean | terminal_liquidation_reward_total_mean | abs_delayed_share_of_total_reward |
| --- | --- | --- | --- | --- | --- | --- |
| nonsequential | balanced_reference | contextual_bandit | 54.770205 | 0 | 0 | 0 |
| nonsequential | balanced_reference | linear_q_approx | 52.986937 | 0 | 0 | 0 |
| nonsequential | balanced_reference | q_learning | 50.82263 | 0 | 0 | 0 |
| nonsequential | balanced_reference | q_learning_fine | 47.644292 | 0 | 0 | 0 |
| nonsequential | constrained_review_capacity | contextual_bandit | 57.195237 | 0 | 0 | 0 |
| nonsequential | constrained_review_capacity | linear_q_approx | 54.701623 | 0 | 0 | 0 |
| nonsequential | constrained_review_capacity | q_learning | 52.583803 | 0 | 0 | 0 |
| nonsequential | constrained_review_capacity | q_learning_fine | 50.076212 | 0 | 0 | 0 |
| nonsequential | high_false_restriction_cost | contextual_bandit | 54.23824 | 0 | 0 | 0 |
| nonsequential | high_false_restriction_cost | linear_q_approx | 48.736131 | 0 | 0 | 0 |
| nonsequential | high_false_restriction_cost | q_learning | 49.20011 | 0 | 0 | 0 |
| nonsequential | high_false_restriction_cost | q_learning_fine | 45.991769 | 0 | 0 | 0 |
| nonsequential | high_review_capacity | contextual_bandit | 57.737925 | 0 | 0 | 0 |
| nonsequential | high_review_capacity | linear_q_approx | 53.688334 | 0 | 0 | 0 |
| nonsequential | high_review_capacity | q_learning | 52.795795 | 0 | 0 | 0 |
| nonsequential | high_review_capacity | q_learning_fine | 48.775025 | 0 | 0 | 0 |
| nonsequential | high_security_pressure | contextual_bandit | 61.59581 | 0 | 0 | 0 |
| nonsequential | high_security_pressure | linear_q_approx | 50.768913 | 0 | 0 | 0 |
| nonsequential | high_security_pressure | q_learning | 55.899725 | 0 | 0 | 0 |
| nonsequential | high_security_pressure | q_learning_fine | 52.60035 | 0 | 0 | 0 |
| nonsequential | participation_sensitive_ecosystem | contextual_bandit | 56.202761 | 0 | 0 | 0 |
| nonsequential | participation_sensitive_ecosystem | linear_q_approx | 55.745075 | 0 | 0 | 0 |
| nonsequential | participation_sensitive_ecosystem | q_learning | 48.605594 | 0 | 0 | 0 |
| nonsequential | participation_sensitive_ecosystem | q_learning_fine | 47.203094 | 0 | 0 | 0 |

## Security-comparable ranking excerpt

| environment | scenario | policy | security_incidents_mean | mean_reward_mean | security_comparable_rank |
| --- | --- | --- | --- | --- | --- |
| nonsequential | balanced_reference | q_learning | 1.533333 | 0.508226 | 1 |
| nonsequential | balanced_reference | security_rule | 1.2 | 0.481989 | 2 |
| nonsequential | balanced_reference | q_learning_fine | 1.566667 | 0.476443 | 3 |
| nonsequential | balanced_reference | heuristic_score_threshold | 1.9 | 0.474969 | 4 |
| nonsequential | constrained_review_capacity | contextual_bandit | 0.4 | 0.571952 | 1 |
| nonsequential | constrained_review_capacity | oracle_upper_bound | 0.5 | 0.55752 | 2 |
| nonsequential | constrained_review_capacity | linear_q_approx | 1 | 0.547016 | 3 |
| nonsequential | constrained_review_capacity | q_learning | 0.9 | 0.525838 | 4 |
| nonsequential | constrained_review_capacity | heuristic_score_threshold | 0.8 | 0.518588 | 5 |
| nonsequential | constrained_review_capacity | q_learning_fine | 1.066667 | 0.500762 | 6 |
| nonsequential | constrained_review_capacity | security_rule | 1 | 0.471821 | 7 |
| nonsequential | high_false_restriction_cost | heuristic_score_threshold | 1.2 | 0.508501 | 1 |
| nonsequential | high_false_restriction_cost | q_learning | 1.3 | 0.492001 | 2 |
| nonsequential | high_false_restriction_cost | security_rule | 1.1 | 0.464601 | 3 |
| nonsequential | high_false_restriction_cost | q_learning_fine | 1.1 | 0.459918 | 4 |
| nonsequential | high_review_capacity | q_learning | 0.966667 | 0.527958 | 1 |
| nonsequential | high_review_capacity | q_learning_fine | 1.3 | 0.48775 | 2 |
| nonsequential | high_review_capacity | security_rule | 1.4 | 0.475284 | 3 |
| nonsequential | high_security_pressure | contextual_bandit | 0.6 | 0.615958 | 1 |
| nonsequential | high_security_pressure | oracle_upper_bound | 0.8 | 0.571856 | 2 |
| nonsequential | high_security_pressure | q_learning | 0.966667 | 0.558997 | 3 |
| nonsequential | high_security_pressure | q_learning_fine | 1.233333 | 0.526003 | 4 |
| nonsequential | participation_sensitive_ecosystem | linear_q_approx | 1.166667 | 0.557451 | 1 |
| nonsequential | participation_sensitive_ecosystem | q_learning | 1.433333 | 0.486056 | 2 |
| nonsequential | participation_sensitive_ecosystem | heuristic_score_threshold | 1.6 | 0.48011 | 3 |
| nonsequential | participation_sensitive_ecosystem | security_rule | 1.2 | 0.477745 | 4 |
| nonsequential | participation_sensitive_ecosystem | q_learning_fine | 1.566667 | 0.472031 | 5 |
| sequential | balanced_reference | contextual_bandit | 1.866667 | 0.516561 | 1 |
| sequential | balanced_reference | q_learning | 2.2 | 0.46707 | 2 |
| sequential | constrained_review_capacity | q_learning | 2.233333 | 0.438666 | 1 |
| sequential | high_false_restriction_cost | linear_q_approx | 2.333333 | 0.482223 | 1 |
| sequential | high_false_restriction_cost | contextual_bandit | 2 | 0.463779 | 2 |
| sequential | high_false_restriction_cost | heuristic_score_threshold | 1.7 | 0.462478 | 3 |
| sequential | high_false_restriction_cost | q_learning | 2 | 0.425186 | 4 |
| sequential | high_review_capacity | contextual_bandit | 2.133333 | 0.515272 | 1 |
| sequential | high_review_capacity | q_learning | 1.966667 | 0.453709 | 2 |

## Interpretation limits

- No result is presented as real-platform evidence.
- The oracle policy has access to hidden harmfulness and is only a theoretical upper bound.
- Linear function approximation is included as a lightweight RL approximation baseline; DQN is not used because the workspace does not require a deep-learning stack and the state/action space is small enough to audit tabular coverage first.
- A negative result is preserved if Q-learning remains below the corrected contextual bandit after reward attribution, terminal liquidation and convergence checks.
