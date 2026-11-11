# V8 ECR — results from completed experiments

A/B/C reuse 33,840 verified primary evaluations: 30 independent training blocks per learned-policy cell, ten evaluations per block. Exact V7 and corrected V8 remain separate. The additional nested-budget supplement contributes 5,400 snapshot evaluations from 180 policy trajectories; its 1,800 evaluations at 180 episodes reproduce existing primary results and are not new independent evidence.

Primary unit: the mean over ten evaluations of one independently trained policy. Paired t intervals use 29 degrees of freedom. Holm tests and Bonferroni simultaneous intervals use separate 12-cell A families and separate six-cell B/C interaction families. Secondary operational and linear-Q intervals are pointwise. Standardized paired effects dz are conditional on the ten-seed evaluation panel.

## Main corrected reward comparisons

| environment | scenario | difference [pointwise 95% CI] | family 95% CI | p_holm | dz |
|---|---|---|---|---|---|
| nonsequential | balanced_reference | -0.11161 [-0.13261, -0.09060] | [-0.14356, -0.07966] | 5.826436710542826e-11 | -1.983798940327699 |
| nonsequential | constrained_review_capacity | -0.10540 [-0.12179, -0.08902] | [-0.13032, -0.08049] | 1.026523294294049e-12 | -2.4026134957456087 |
| nonsequential | high_false_restriction_cost | -0.08543 [-0.10531, -0.06554] | [-0.11567, -0.05519] | 4.5596398631943325e-09 | -1.6042001135752382 |
| nonsequential | high_review_capacity | -0.08030 [-0.09613, -0.06446] | [-0.10437, -0.05622] | 1.4297163495994543e-10 | -1.893768427582725 |
| nonsequential | high_security_pressure | -0.10108 [-0.11766, -0.08451] | [-0.12629, -0.07588] | 3.157690211779363e-12 | -2.2775998441609246 |
| nonsequential | participation_sensitive_ecosystem | -0.09963 [-0.11755, -0.08171] | [-0.12689, -0.07237] | 2.661366684939281e-11 | -2.0756794792074964 |
| sequential | balanced_reference | -0.03669 [-0.04997, -0.02341] | [-0.05688, -0.01650] | 4.15300510277879e-06 | -1.0319264566500062 |
| sequential | constrained_review_capacity | -0.06797 [-0.08046, -0.05549] | [-0.08696, -0.04899] | 3.7952974602135225e-11 | -2.0334573638229134 |
| sequential | high_false_restriction_cost | -0.05600 [-0.07164, -0.04037] | [-0.07978, -0.03222] | 1.3628159716242113e-07 | -1.3374435308548769 |
| sequential | high_review_capacity | -0.04666 [-0.06184, -0.03148] | [-0.06974, -0.02358] | 1.4507133469068494e-06 | -1.14799846302398 |
| sequential | high_security_pressure | -0.15239 [-0.17434, -0.13045] | [-0.18577, -0.11902] | 1.6360375061968954e-13 | -2.5929730135019136 |
| sequential | participation_sensitive_ecosystem | -0.08367 [-0.09673, -0.07061] | [-0.10353, -0.06381] | 1.0351789643027546e-12 | -2.392651056115493 |

All twelve corrected and all twelve exact-V7 primary reward contrasts are negative after their respective Holm adjustment. This concerns tested algorithms under specified budgets and synthetic dynamics; it is not a general ranking of reinforcement learning.

## Feedback and observation interventions

| condition | scenario | difference [pointwise 95% CI] | family 95% CI | p_holm | dz |
|---|---|---|---|---|---|
| B_loss25 | balanced_reference | -0.00106 [-0.01979, 0.01767] | [-0.02699, 0.02487] | 1.0 | -0.0210999611930728 |
| B_loss25 | constrained_review_capacity | 0.00689 [-0.01060, 0.02438] | [-0.01733, 0.03110] | 1.0 | 0.1470498489652743 |
| B_loss25 | high_security_pressure | 0.01014 [-0.02296, 0.04323] | [-0.03568, 0.05595] | 1.0 | 0.1143614827144375 |
| B_loss50 | balanced_reference | -0.01721 [-0.03481, 0.00040] | [-0.04158, 0.00716] | 0.3302239813342872 | -0.3649982773137724 |
| B_loss50 | constrained_review_capacity | -0.00308 [-0.02494, 0.01877] | [-0.03334, 0.02718] | 1.0 | -0.0526732810425027 |
| B_loss50 | high_security_pressure | 0.02294 [-0.01551, 0.06140] | [-0.03029, 0.07618] | 1.0 | 0.222807066152497 |
| C_enriched | balanced_reference | 0.00481 [-0.00930, 0.01892] | [-0.01472, 0.02435] | 1.0 | 0.127356712541035 |
| C_enriched | constrained_review_capacity | 0.00267 [-0.01378, 0.01912] | [-0.02011, 0.02545] | 1.0 | 0.0605873656133513 |
| C_enriched | high_security_pressure | 0.02482 [-0.00548, 0.05512] | [-0.01714, 0.06678] | 0.523363542530547 | 0.3058242779637229 |
| C_reduced | balanced_reference | -0.01939 [-0.03603, -0.00275] | [-0.04243, 0.00365] | 0.1434981096003544 | -0.4351642856191512 |
| C_reduced | constrained_review_capacity | -0.00023 [-0.01632, 0.01586] | [-0.02250, 0.02204] | 1.0 | -0.0053757762179346 |
| C_reduced | high_security_pressure | 0.01773 [-0.01029, 0.04575] | [-0.02107, 0.05653] | 0.8234568166885534 | 0.2362574848727746 |

No B or C interaction survives its six-comparison family adjustment. This does not establish equivalence or exclude other feedback and observability mechanisms. B masks scalar teaching reports with zero encoding, scaling expected non-guard training rewards by 1-p; C also changes representation complexity. Inherited privileged aggregate indicators limit real-world applicability.

## Exploratory nested-budget supplement

This supplement was prospectively specified after primary V8 results were known. It extends exactly the same trajectories, with frozen epsilon decay of 60 episodes, to snapshots at 180/360/720 episodes in two sequential scenarios. Separate four-test exploratory families address absolute tabular-Q gaps and their changes from 180. No hyperparameter search or equivalence margin was specified.

| scenario | budget | comparison | difference [pointwise 95% CI] | family 95% CI | p_holm | dz |
|---|---|---|---|---|---|---|
| balanced_reference | 180 | q_learning_minus_contextual_bandit | -0.03669 [-0.04997, -0.02341] | [nan, nan] | nan | -1.0319264566500068 |
| balanced_reference | 180 | linear_q_approx_minus_contextual_bandit | -0.06100 [-0.08637, -0.03563] | [nan, nan] | nan | -0.8977830204342011 |
| balanced_reference | 360 | q_learning_minus_contextual_bandit | -0.03628 [-0.04921, -0.02336] | [-0.05311, -0.01945] | 3.2531273070339854e-06 | -1.0480684917453302 |
| balanced_reference | 360 | linear_q_approx_minus_contextual_bandit | -0.08618 [-0.12789, -0.04446] | [nan, nan] | nan | -0.771404446732783 |
| balanced_reference | 720 | q_learning_minus_contextual_bandit | -0.05544 [-0.06563, -0.04525] | [-0.06871, -0.04217] | 1.1146955717074957e-11 | -2.03108974156064 |
| balanced_reference | 720 | linear_q_approx_minus_contextual_bandit | -0.15268 [-0.23859, -0.06677] | [nan, nan] | nan | -0.6636442396397546 |
| high_security_pressure | 180 | q_learning_minus_contextual_bandit | -0.15239 [-0.17434, -0.13045] | [nan, nan] | nan | -2.592973013501913 |
| high_security_pressure | 180 | linear_q_approx_minus_contextual_bandit | -0.19202 [-0.27293, -0.11111] | [nan, nan] | nan | -0.8861978686118575 |
| high_security_pressure | 360 | q_learning_minus_contextual_bandit | -0.14241 [-0.16090, -0.12391] | [-0.16649, -0.11832] | 3.679750022895715e-15 | -2.8747699614280227 |
| high_security_pressure | 360 | linear_q_approx_minus_contextual_bandit | -0.25979 [-0.37469, -0.14488] | [nan, nan] | nan | -0.8442394544894515 |
| high_security_pressure | 720 | q_learning_minus_contextual_bandit | -0.11728 [-0.13249, -0.10206] | [-0.13708, -0.09747] | 3.679750022895715e-15 | -2.878852200315138 |
| high_security_pressure | 720 | linear_q_approx_minus_contextual_bandit | -0.25774 [-0.40032, -0.11516] | [nan, nan] | nan | -0.6749849792283414 |

| scenario | budget | comparison | difference [pointwise 95% CI] | family 95% CI | p_holm | dz |
|---|---|---|---|---|---|---|
| balanced_reference | 360 | q_learning_minus_contextual_bandit | 0.00041 [-0.00592, 0.00673] | [-0.00783, 0.00865] | 0.8957900008292637 | 0.024124450158492707 |
| balanced_reference | 720 | q_learning_minus_contextual_bandit | -0.01875 [-0.02956, -0.00794] | [-0.03283, -0.00467] | 0.005396117186183256 | -0.647484927356514 |
| balanced_reference | 360 | linear_q_approx_minus_contextual_bandit | -0.02518 [-0.07495, 0.02460] | [nan, nan] | nan | -0.18888986485751333 |
| balanced_reference | 720 | linear_q_approx_minus_contextual_bandit | -0.09168 [-0.17569, -0.00767] | [nan, nan] | nan | -0.4074813855708653 |
| high_security_pressure | 360 | q_learning_minus_contextual_bandit | 0.00999 [-0.00538, 0.02535] | [-0.01003, 0.03000] | 0.38853313497070247 | 0.24260950501417178 |
| high_security_pressure | 720 | q_learning_minus_contextual_bandit | 0.03512 [0.01104, 0.05920] | [0.00376, 0.06647] | 0.017231925562619092 | 0.5445324892386626 |
| high_security_pressure | 360 | linear_q_approx_minus_contextual_bandit | -0.06777 [-0.22708, 0.09155] | [nan, nan] | nan | -0.15883481318594772 |
| high_security_pressure | 720 | linear_q_approx_minus_contextual_bandit | -0.06572 [-0.23853, 0.10709] | [nan, nan] | nan | -0.14200833480661118 |

Longer training is a sensitivity check, not a proof of convergence or adequate sample size for every representation. Shared random seeds do not perfectly pair exogenous shocks because draw consumption is action dependent. Inference remains conditional on the ten-seed evaluation panel. Budget scenarios are purposive and do not represent all six governance settings.

## Operational outcomes and complete estimates

ANALYSIS/policy_levels.csv and paired_policy_contrasts.csv report legitimate acceptance, incidents, unjustified restrictions, review workloads, costs, guard interventions and further diagnostics. Reward advantages do not establish operational dominance. Four deterministic references have ten evaluations each and no independent training replications. Source data, all effect sizes and pointwise intervals remain available, including results that contradict simple policy rankings.

## Interpretation of the new budget results

At 720 episodes the paired gap change is −0.01875 in the reference scenario (95% CI −0.02956 to −0.00794; exploratory Holm p .005396), and +0.03512 under security pressure (95% CI .01104 to .05920; exploratory Holm p .01723). Thus the reference disadvantage widens while the security disadvantage narrows. All four new absolute Q-minus-bandit contrasts remain negative after their separate exploratory Holm correction. Additional experience matters differently by scenario and does not establish convergence.
