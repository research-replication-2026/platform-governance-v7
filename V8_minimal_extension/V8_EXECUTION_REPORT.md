# V8 execution report — completed and verified

Finalized: 2026-11-11T14:52:50.509334+00:00. Client date: 11 November 2026, Europe/Paris.

## What was actually executed

Five V7 tests and five V8 behavioral tests passed before campaign execution. Targeted reproduction matched 510 saved evaluation rows (maximum absolute error 2.842170943040401e-14), including three newly retrained 180-episode policies. Corrected smoke checks used 20 episodes per learner and deterministic repeated evaluation. The discretizer acceleration matched original NumPy results at boundaries and on 1,000 random vectors. Effective episode seeds were initially checked for within-policy training-stream collisions and training/evaluation overlap; both were zero in every scenario/environment. An additional audit found cross-algorithm/cross-group overlap among historical labels. Amendment 01 added three new labels and selected thirty disjoint new blocks for primary analysis, with zero overlap between different blocks across all three algorithms.

The first V8 test-harness attempt raised a NumPy-array equality error; explicit array comparison fixed that harness. No simulator result was discarded in response. Independent analysis sanity checks verified use of 30 training means rather than 300 observations, Student t arithmetic, Holm adjustment, rule broadcasting and crossed resampling. The original V7 three-cluster analysis was recomputed as eight negative and four zero-crossing Q-minus-bandit intervals, agreeing with the canonical manuscript's grouped sensitivity count.

The complete campaign contains 1,224 checkpoint jobs. New executions fitted 3,456 policy runs at 180 episodes of 100 decisions each; 108 compatible archived policy runs were reused as exact-V7 historical diagnostics. Primary inference uses 3,240 newly trained policies in disjoint seed blocks 10001..10030. Three historical labels per cell are retained separately, including newly executed corrected/B/C diagnostics. Each learned cell has 30 independent training labels and ten evaluations per label. Rules were evaluated once per evaluation seed with training label 0. Dataset: 32,400 learned-policy rows plus 1,440 deterministic rows, 33,840 rows total. The all-executed archive contains 37,080 evaluation rows, including 3,240 historical-label diagnostic rows. Of the all-executed rows, 1,080 learned rows and 480 rule rows reuse archived V7 outcomes. The remaining 35,520 evaluation rows were newly executed. The primary dataset reuses only the 480 exact-arm deterministic rows. Validation/smoke fits and their 510 comparisons are separate from the campaign dataset.

All six original scenarios and both environments were run for the exact and corrected A strata. B and C were run separately at 30 training labels in the balanced, high-security-pressure and constrained-review-capacity sequential scenarios. B1/C1 reuse the corrected A control. No factorial interaction was executed. No training budget was reduced and no hyperparameter was tuned on campaign evaluation outcomes.

Campaign completion timestamps range from 2026-11-11T13:43:32.198084+00:00 to 2026-11-11T14:47:15.418574+00:00. Approximate elapsed span including initial tasks and the transition between A and B/C: 63.9 minutes. The benchmark estimated 10,008.6 serial training seconds before evaluation/checkpoint overhead. The host has an Intel Core i5-1345U, ten physical/twelve logical CPUs and approximately 15.6 GiB RAM. A ran with six processes. B/C began with six processes and continued with eight after a documented resource adjustment at approximately 56% total CPU utilization and 2.3 GiB free memory. The additional seed campaign used eight processes. Numerical library thread counts remained one. See `V8_RESOURCE_ADJUSTMENT.json` for the exact restart and checkpoint count. The integrated Intel UHD GPU was not used.

## Errors, provenance and preservation

Campaign error events: 0. Error details, if any, are retained verbatim in `campaign.jsonl`. The only pre-campaign harness error is described above. Atomic checkpoints preserve completed jobs; no failed job or partial dataset has been promoted to a completed result. The original manifest and additive amendment manifest record source and prospective specification hashes; original and corrected implementations remain distinguishable in every row. The seed amendment was frozen before expanded outcome analysis and was motivated only by seed-structure checks, not results. The idle pipeline watcher was restarted to insert the additive campaign before analysis. A later resource adjustment restarted the local B/C worker pool from six to eight processes. Every committed checkpoint was preserved; uncommitted in-flight work was replayed from its original seed at full budget. Partial work from that restart is not counted as completed policy fits.

All 138 original V7 and archived manuscript files retain their SHA-256 hashes. No additional files appeared in those directories. The original simulator remains unchanged; its exact semantics, including the two diagnosed defects, are preserved by the legacy arm. The corrected arm cancels terminal bootstrap and pays incidents created during terminal review/check resolution.

## Final verification

- All planned checkpoint files exist; all learned cells contain exactly 30 × 10 evaluations and all rules have ten evaluations without artificial training replication.
- Corrected episodes end with zero residual pending items. Reward-component sums match total reward within floating-point tolerance.
- Independent groupby/pivot recalculation of every Q-minus-bandit training-level mean, SD and t interval matches the analysis to a maximum error of 1.11e-16. Primary condition-interaction means and SDs were recalculated independently.
- Realized reward-report missingness: {"B_loss25": 0.24995895061728396, "B_loss50": 0.5002024691358025}. The same masking schedule was verified across learners within a condition/training group.
- All recorded learning curves reach episode 179. Source/protocol checks, exact effective seeds, run versions and analysis input fingerprints are retained.
- Five scientific figures were exported to PNG, PDF and SVG; visual review is recorded in `FIGURE_REVIEW.md`.

## Scientific results and their scope

Exact A: {"negative_point_means": 12, "positive_point_means": 0, "negative_holm": 12, "positive_holm": 0, "negative_crossed_ci": 12, "positive_crossed_ci": 0}.
Corrected A: {"negative_point_means": 12, "positive_point_means": 0, "negative_holm": 12, "positive_holm": 0, "negative_crossed_ci": 12, "positive_crossed_ci": 0}.
Feedback interactions: {"relative_improvements_holm": 0, "relative_deteriorations_holm": 0, "total_comparisons": 6}.
Observability interactions: {"relative_improvements_holm": 0, "relative_deteriorations_holm": 0, "total_comparisons": 6}.

Full effect estimates, standardized effects, pointwise/simultaneous intervals, adjusted p-values, crossed-bootstrap sensitivity and operational outcomes are in `V8_RESULTS_SUMMARY.md` and `analysis/`. Statistical conclusions condition on the fixed ten-seed evaluation panel; shared integer seeds do not imply perfectly common exogenous shocks. The extension cannot establish an intrinsic ordering of RL and bandits or causal effects on real platforms.

## Deliverables and remaining work

All ten requested deliverables are supplied: initial audit, frozen protocol, detailed results CSV, reproducible analysis script, scientific figures, results summary, theoretical contribution memo, English manuscript insertions, exact reproduction instructions, and this execution report. Supporting artifacts include the complete manifest, source hashes, seed audit, runtime benchmark, tests, checkpoints, raw logs, training diagnostics/curves, verification JSON and detailed analysis tables.

The experimental extension is complete. Remaining manuscript work is to select and integrate the proposed English paragraphs and V8 tables/figures, replace the old three-run uncertainty statements, preserve the legacy/corrected distinction, and revise claims to the mechanisms actually identified. The archived manuscript was deliberately not edited. The extension strengthens robustness and testability of the contribution; editorial fit, acceptance, real-platform calibration and a demonstrated positive sequential-value boundary are not guaranteed by these simulations.
