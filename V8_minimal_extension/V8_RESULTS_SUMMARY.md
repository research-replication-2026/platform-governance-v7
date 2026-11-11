# V8 results — completed and numerically verified

Analyzed 33,840 primary evaluation rows. Every learned cell has 30 independent training runs (labels 10001..10030, amendment 01) and 10 fixed evaluation seeds per run. The three historical groups are retained separately because their cross-algorithm episode streams overlap across groups. Four deterministic rules have ten evaluations and no training replicates. Intervals quantify training variation conditional on this evaluation panel; crossed bootstrap intervals additionally assess evaluation-panel sensitivity.

The exact V7 arm and corrected arm are separate. B1/C1 reuse the corrected control. All signs below are Q minus bandit, or changes in that difference. A positive interaction means a relative improvement of Q; it need not imply that Q exceeds the bandit.

The primary reward disadvantage persists in all twelve main cells, both before and after the implementation corrections. Neither the feedback-loss family nor the observability family establishes a change in that policy gap after the prespecified multiplicity adjustment. Thus robustness is strengthened, while the proposed feedback/observability explanations remain unconfirmed under these interventions. This is not evidence of equivalence or absence of all such mechanisms.

## RQ1 — A_v7_exact

12/12 negative mean contrasts; 12/12 negative Holm-adjusted results; 0/12 positive adjusted results. All 12 negative crossed-bootstrap intervals are pointwise sensitivity results, not a separate simultaneous family test.

| environment | scenario | difference [95% t CI] | simultaneous 95% family CI | Holm p | paired dz | crossed bootstrap 95% |
|---|---|---|---|---|---|---|
| nonsequential | balanced_reference | -0.0379 [-0.0453, -0.0305] | [-0.0491, -0.0267] | 1.465e-10 | -1.920 | [-0.0533, -0.0214] |
| nonsequential | constrained_review_capacity | -0.0541 [-0.0632, -0.0450] | [-0.0679, -0.0403] | 6.054e-12 | -2.227 | [-0.0705, -0.0383] |
| nonsequential | high_false_restriction_cost | -0.0345 [-0.0459, -0.0231] | [-0.0518, -0.0172] | 2.689e-06 | -1.134 | [-0.0520, -0.0167] |
| nonsequential | high_review_capacity | -0.0545 [-0.0630, -0.0460] | [-0.0675, -0.0416] | 1.166e-12 | -2.390 | [-0.0711, -0.0398] |
| nonsequential | high_security_pressure | -0.0589 [-0.0716, -0.0462] | [-0.0782, -0.0396] | 1.17e-09 | -1.731 | [-0.0776, -0.0410] |
| nonsequential | participation_sensitive_ecosystem | -0.0529 [-0.0643, -0.0416] | [-0.0702, -0.0356] | 1.17e-09 | -1.740 | [-0.0718, -0.0345] |
| sequential | balanced_reference | -0.0381 [-0.0508, -0.0254] | [-0.0574, -0.0188] | 2.689e-06 | -1.121 | [-0.0679, -0.0095] |
| sequential | constrained_review_capacity | -0.0693 [-0.0817, -0.0568] | [-0.0881, -0.0504] | 2.441e-11 | -2.083 | [-0.1001, -0.0405] |
| sequential | high_false_restriction_cost | -0.0573 [-0.0733, -0.0414] | [-0.0816, -0.0331] | 1.719e-07 | -1.341 | [-0.0831, -0.0306] |
| sequential | high_review_capacity | -0.0510 [-0.0683, -0.0337] | [-0.0774, -0.0246] | 2.689e-06 | -1.099 | [-0.0953, -0.0055] |
| sequential | high_security_pressure | -0.1610 [-0.1841, -0.1379] | [-0.1961, -0.1258] | 1.51e-13 | -2.601 | [-0.2011, -0.1231] |
| sequential | participation_sensitive_ecosystem | -0.0816 [-0.0960, -0.0672] | [-0.1035, -0.0597] | 1.875e-11 | -2.117 | [-0.1191, -0.0446] |

## RQ1 — A_corrected

12/12 negative mean contrasts; 12/12 negative Holm-adjusted results; 0/12 positive adjusted results. All 12 negative crossed-bootstrap intervals are pointwise sensitivity results, not a separate simultaneous family test.

| environment | scenario | difference [95% t CI] | simultaneous 95% family CI | Holm p | paired dz | crossed bootstrap 95% |
|---|---|---|---|---|---|---|
| nonsequential | balanced_reference | -0.1116 [-0.1326, -0.0906] | [-0.1436, -0.0797] | 5.826e-11 | -1.984 | [-0.1357, -0.0859] |
| nonsequential | constrained_review_capacity | -0.1054 [-0.1218, -0.0890] | [-0.1303, -0.0805] | 1.027e-12 | -2.403 | [-0.1259, -0.0852] |
| nonsequential | high_false_restriction_cost | -0.0854 [-0.1053, -0.0655] | [-0.1157, -0.0552] | 4.56e-09 | -1.604 | [-0.1096, -0.0604] |
| nonsequential | high_review_capacity | -0.0803 [-0.0961, -0.0645] | [-0.1044, -0.0562] | 1.43e-10 | -1.894 | [-0.1013, -0.0596] |
| nonsequential | high_security_pressure | -0.1011 [-0.1177, -0.0845] | [-0.1263, -0.0759] | 3.158e-12 | -2.278 | [-0.1217, -0.0805] |
| nonsequential | participation_sensitive_ecosystem | -0.0996 [-0.1176, -0.0817] | [-0.1269, -0.0724] | 2.661e-11 | -2.076 | [-0.1217, -0.0773] |
| sequential | balanced_reference | -0.0367 [-0.0500, -0.0234] | [-0.0569, -0.0165] | 4.153e-06 | -1.032 | [-0.0680, -0.0060] |
| sequential | constrained_review_capacity | -0.0680 [-0.0805, -0.0555] | [-0.0870, -0.0490] | 3.795e-11 | -2.033 | [-0.0987, -0.0396] |
| sequential | high_false_restriction_cost | -0.0560 [-0.0716, -0.0404] | [-0.0798, -0.0322] | 1.363e-07 | -1.337 | [-0.0829, -0.0284] |
| sequential | high_review_capacity | -0.0467 [-0.0618, -0.0315] | [-0.0697, -0.0236] | 1.451e-06 | -1.148 | [-0.0904, -0.0017] |
| sequential | high_security_pressure | -0.1524 [-0.1743, -0.1304] | [-0.1858, -0.1190] | 1.636e-13 | -2.593 | [-0.1907, -0.1173] |
| sequential | participation_sensitive_ecosystem | -0.0837 [-0.0967, -0.0706] | [-0.1035, -0.0638] | 1.035e-12 | -2.393 | [-0.1237, -0.0461] |

## RQ2 — reward-feedback loss

0 relative improvements and 0 relative deteriorations established after Holm adjustment across six contrasts. Intervals crossing zero do not establish equivalence.

| condition | scenario | difference [95% t CI] | simultaneous 95% family CI | Holm p | paired dz | crossed bootstrap 95% |
|---|---|---|---|---|---|---|
| B_loss25 | balanced_reference | -0.0011 [-0.0198, 0.0177] | [-0.0270, 0.0249] | 1 | -0.021 | [-0.0276, 0.0272] |
| B_loss25 | constrained_review_capacity | 0.0069 [-0.0106, 0.0244] | [-0.0173, 0.0311] | 1 | 0.147 | [-0.0209, 0.0405] |
| B_loss25 | high_security_pressure | 0.0101 [-0.0230, 0.0432] | [-0.0357, 0.0560] | 1 | 0.114 | [-0.0312, 0.0521] |
| B_loss50 | balanced_reference | -0.0172 [-0.0348, 0.0004] | [-0.0416, 0.0072] | 0.3302 | -0.365 | [-0.0461, 0.0099] |
| B_loss50 | constrained_review_capacity | -0.0031 [-0.0249, 0.0188] | [-0.0333, 0.0272] | 1 | -0.053 | [-0.0337, 0.0265] |
| B_loss50 | high_security_pressure | 0.0229 [-0.0155, 0.0614] | [-0.0303, 0.0762] | 1 | 0.223 | [-0.0235, 0.0668] |

## RQ3 — state observability

0 relative improvements and 0 relative deteriorations established after Holm adjustment across six contrasts. Intervals crossing zero do not establish equivalence.

| condition | scenario | difference [95% t CI] | simultaneous 95% family CI | Holm p | paired dz | crossed bootstrap 95% |
|---|---|---|---|---|---|---|
| C_enriched | balanced_reference | 0.0048 [-0.0093, 0.0189] | [-0.0147, 0.0243] | 1 | 0.127 | [-0.0186, 0.0278] |
| C_enriched | constrained_review_capacity | 0.0027 [-0.0138, 0.0191] | [-0.0201, 0.0254] | 1 | 0.061 | [-0.0237, 0.0269] |
| C_enriched | high_security_pressure | 0.0248 [-0.0055, 0.0551] | [-0.0171, 0.0668] | 0.5234 | 0.306 | [-0.0165, 0.0669] |
| C_reduced | balanced_reference | -0.0194 [-0.0360, -0.0028] | [-0.0424, 0.0036] | 0.1435 | -0.435 | [-0.0421, 0.0043] |
| C_reduced | constrained_review_capacity | -0.0002 [-0.0163, 0.0159] | [-0.0225, 0.0220] | 1 | -0.005 | [-0.0265, 0.0234] |
| C_reduced | high_security_pressure | 0.0177 [-0.0103, 0.0458] | [-0.0211, 0.0565] | 0.8235 | 0.236 | [-0.0164, 0.0558] |

## Secondary comparison — linear Q approximation

These are secondary, pointwise intervals over training means, without a prespecified multiplicity-adjusted superiority family. Linear Q uses all raw features, whereas the two tabular learners share a reduced state key. Its contrasts therefore combine representation and learning differences.

| environment | scenario | difference [pointwise 95% CI] |
|---|---|---|
| nonsequential | balanced_reference | -0.1108 [-0.1402, -0.0813] |
| nonsequential | constrained_review_capacity | -0.1519 [-0.1986, -0.1052] |
| nonsequential | high_false_restriction_cost | -0.1125 [-0.1585, -0.0665] |
| nonsequential | high_review_capacity | -0.1331 [-0.1700, -0.0962] |
| nonsequential | high_security_pressure | -0.1700 [-0.2014, -0.1385] |
| nonsequential | participation_sensitive_ecosystem | -0.1443 [-0.1833, -0.1052] |
| sequential | balanced_reference | -0.0610 [-0.0864, -0.0356] |
| sequential | constrained_review_capacity | -0.0900 [-0.1318, -0.0482] |
| sequential | high_false_restriction_cost | -0.0330 [-0.0594, -0.0065] |
| sequential | high_review_capacity | -0.0418 [-0.0779, -0.0058] |
| sequential | high_security_pressure | -0.1920 [-0.2729, -0.1111] |
| sequential | participation_sensitive_ecosystem | -0.0859 [-0.1282, -0.0437] |

## Corrected sequential reward levels and deterministic references

Learned-policy entries give mean ± between-training SD. Rule entries give fixed evaluation-panel means only; there are no independent rule training runs. Permissive and restrictive reference levels and all nonsequential cells remain in the complete levels table.

| scenario | contextual_bandit | q_learning | linear_q_approx | security_rule | heuristic_score_threshold |
|---|---|---|---|---|---|
| balanced_reference | 0.5135 ± 0.0135 | 0.4768 ± 0.0291 | 0.4525 ± 0.0708 | 0.4667 | 0.4973 |
| constrained_review_capacity | 0.5274 ± 0.0137 | 0.4594 ± 0.0298 | 0.4374 ± 0.1111 | 0.4671 | 0.4819 |
| high_false_restriction_cost | 0.4996 ± 0.0213 | 0.4436 ± 0.0339 | 0.4666 ± 0.0701 | 0.4285 | 0.4625 |
| high_review_capacity | 0.5045 ± 0.0249 | 0.4578 ± 0.0277 | 0.4626 ± 0.0933 | 0.5132 | 0.4955 |
| high_security_pressure | 0.6045 ± 0.0249 | 0.4521 ± 0.0497 | 0.4125 ± 0.2128 | 0.4554 | 0.4205 |
| participation_sensitive_ecosystem | 0.5453 ± 0.0116 | 0.4616 ± 0.0329 | 0.4594 ± 0.1114 | 0.4292 | 0.4629 |

## All operational outcomes and secondary policy contrasts

Full means, between-training SD and pointwise confidence intervals are in `analysis/policy_levels.csv`; paired differences and standardized effects are in `analysis/paired_policy_contrasts.csv`; the complete mechanism contrasts are in `analysis/condition_interactions.csv`. No overall governance dominance is inferred from reward alone. Rule levels are descriptive; rule-relative learner intervals condition on the same fixed ten rule evaluations.

### Corrected sequential Q-minus-bandit operational differences (pointwise intervals)

| scenario | metric | difference [95% t CI] |
|---|---|---|
| balanced_reference | legitimate_contributions_accepted | 0.707 [0.130, 1.284] |
| balanced_reference | security_incidents | 0.087 [-0.164, 0.337] |
| balanced_reference | rejected_legitimate_innovation | -1.130 [-1.374, -0.886] |
| balanced_reference | human_review_workload | 5.080 [4.185, 5.975] |
| balanced_reference | governance_expenditure | 2.485 [2.191, 2.780] |
| balanced_reference | safety_guard_interventions | 0.000 [0.000, 0.000] |
| constrained_review_capacity | legitimate_contributions_accepted | 1.197 [0.570, 1.823] |
| constrained_review_capacity | security_incidents | 0.820 [0.581, 1.059] |
| constrained_review_capacity | rejected_legitimate_innovation | -1.077 [-1.283, -0.871] |
| constrained_review_capacity | human_review_workload | 3.263 [2.672, 3.854] |
| constrained_review_capacity | governance_expenditure | 2.527 [2.260, 2.794] |
| constrained_review_capacity | safety_guard_interventions | 0.000 [0.000, 0.000] |
| high_false_restriction_cost | legitimate_contributions_accepted | 0.640 [-0.143, 1.423] |
| high_false_restriction_cost | security_incidents | 0.227 [-0.059, 0.512] |
| high_false_restriction_cost | rejected_legitimate_innovation | -0.363 [-0.581, -0.145] |
| high_false_restriction_cost | human_review_workload | 5.270 [4.490, 6.050] |
| high_false_restriction_cost | governance_expenditure | 2.449 [2.119, 2.779] |
| high_false_restriction_cost | safety_guard_interventions | 0.000 [0.000, 0.000] |
| high_review_capacity | legitimate_contributions_accepted | 1.147 [0.605, 1.688] |
| high_review_capacity | security_incidents | -0.200 [-0.484, 0.084] |
| high_review_capacity | rejected_legitimate_innovation | -0.587 [-0.731, -0.442] |
| high_review_capacity | human_review_workload | 5.520 [4.468, 6.572] |
| high_review_capacity | governance_expenditure | 2.681 [2.393, 2.969] |
| high_review_capacity | safety_guard_interventions | 0.000 [0.000, 0.000] |
| high_security_pressure | legitimate_contributions_accepted | -0.393 [-1.207, 0.421] |
| high_security_pressure | security_incidents | 1.240 [0.965, 1.515] |
| high_security_pressure | rejected_legitimate_innovation | 0.020 [-0.211, 0.251] |
| high_security_pressure | human_review_workload | 6.870 [5.840, 7.900] |
| high_security_pressure | governance_expenditure | 3.089 [2.732, 3.446] |
| high_security_pressure | safety_guard_interventions | 0.000 [0.000, 0.000] |
| participation_sensitive_ecosystem | legitimate_contributions_accepted | -1.780 [-2.394, -1.166] |
| participation_sensitive_ecosystem | security_incidents | 0.707 [0.482, 0.931] |
| participation_sensitive_ecosystem | rejected_legitimate_innovation | -0.417 [-0.564, -0.269] |
| participation_sensitive_ecosystem | human_review_workload | 4.057 [3.377, 4.736] |
| participation_sensitive_ecosystem | governance_expenditure | 3.782 [3.350, 4.213] |
| participation_sensitive_ecosystem | safety_guard_interventions | 0.000 [0.000, 0.000] |

## Limits on inference and interpretation

Holm correction applies within each specified primary reward family; pointwise intervals and secondary operational metrics do not establish family-wise superiority. The crossed bootstrap is a sensitivity over only ten evaluation seeds. Reused evaluation seeds and the action-dependent RNG prevent a claim of perfectly paired exogenous shocks. Neither a zero-crossing interval nor an insignificant interaction establishes equivalence.

Feedback loss manipulates missing scalar teaching reports with zero encoding, not the true governance objective or all state sensors. Observation changes also change statistical complexity; sparse tabular coverage is an alternative explanation. The inherited V7 state already contains privileged outcome aggregates. C enrichment adds only administrative schedules and ages. The two terminal corrections are bundled, so their individual contributions cannot be identified.

This simulation cannot establish an intrinsic ranking of reinforcement learning and contextual bandits or validate effects on real platforms. No feedback-by-observability interactions were executed. Source preservation, completeness and additive reward checks are reported in the execution report.
