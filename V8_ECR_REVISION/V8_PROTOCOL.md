# ECR V8 protocol and prospective exploratory budget supplement

Frozen specification: 11 November 2026, before additional budget outcomes.

## Completed primary design reused without new training

Experiments A/B/C use the completed, verified primary data from
`../V8_minimal_extension/`. Copy the dataset byte-for-byte into this directory.
The original prospective protocol and seed amendment remain the specification
for those experiments. A contains six scenarios in non-sequential and sequential
environments, with exact-V7 and corrected strata. B/C use the three specified
sequential scenarios. Each learned cell contains 30 independent training blocks
and ten paired evaluation seeds; deterministic rules have no training replicates.
The estimand is mean true governance reward per decision. Operational metrics
remain separate; training-level t intervals, stated Holm families, simultaneous
Bonferroni intervals and crossed-bootstrap sensitivities are retained.

## Additional question and evidence status

Does a longer matched training trajectory change the Q-minus-bandit contrast in
the balanced and high-security-pressure sequential scenarios? This is an
exploratory adequacy check motivated by an audit after A/B/C outcomes are known.
It is not a retrospectively preregistered confirmatory hypothesis.

Select scenario indices 0 and 1 before new budget outcomes. All three learned
policies are retained: contextual bandit, main tabular Q and linear Q. Use the
corrected environment, full feedback and original V7 observations only. No new
feedback×observability combination or hyperparameter search is introduced.

## Nested experience budgets

Train each scenario/block/policy trajectory for 720 episodes of 100 decisions;
evaluate frozen snapshots after episodes 180, 360 and 720. Each snapshot uses
the same ten V7 evaluation seeds. These are repeated measurements of one
training trajectory, not three independent training runs. There are 30
independent blocks per scenario/policy, 180 completed 720-episode policy fits,
and 5,400 snapshot evaluations. The 1,800 prefix evaluations reproduce existing
180-episode outcomes; they are not new independent primary evidence.

Use labels 10001..10030. Actual training base = 100000000 + 1000000j + 500000
+ 10000i + policy offset (101 bandit, 202 tabular Q, 404 linear Q), with j=0..29,
i=0,1. Preserve episode strides 13/17/19 and every reward, transition, mask,
learning rate, discount, initialization and tie-breaking rule from corrected V8.
Keep epsilon=.04+.24 exp(−e/60) throughout the trajectory. Unlike simply passing
episodes=720 to the old function, this holds the 180-episode prefix fixed and
does not confound additional experience with a different exploration schedule.

Evaluation base = 20261008 + 500000 + 10000i + (5101+37k), k=0..9.
Snapshot evaluation deep-copies the policy, never consumes its training RNG
and never updates it. Audit cross-block episode-seed unions for the complete
720-episode horizon and separation from the evaluation panel.

## Gates and resource accounting

Preserve fingerprints before running. Validate one complete scenario/block job
against the existing 180-episode rows (all common numeric outcomes to 1e-10).
Require no residual terminal obligations and additive rewards. Benchmark that
job; estimate the remaining load and available CPU/RAM. Run six CPU processes
with numerical library thread counts one, without asymmetric budgets. Each
scenario/block job has an atomic checkpoint and a traceable completion record.
Resume skips only committed complete jobs. A failed prefix gate stops the run.

## Analysis fixed before new budget outcomes

Average the ten paired evaluation differences within each of the thirty blocks.
Report Q-minus-bandit and linear-minus-bandit effects at each budget, with
between-block SD, paired dz, two-sided 95% Student t intervals (df 29) and
operational metrics. For the four tabular-Q contrasts at 360/720 episodes across
two scenarios, report Holm-adjusted p-values and Bonferroni family intervals.
Report the paired change in the Q-minus-bandit gap relative to its 180-episode
prefix; these four exploratory changes form a separate Holm family.

Do not infer sufficiency of the training budget from a nonsignificant change.
Do not describe lack of Q superiority as convergence. A ranking persisting at
720 episodes weakens a short-budget explanation in these two cases only; a
reversal is reported without hiding it. The fixed evaluation-panel limitation,
residual partial observability, state complexity and algorithm-specific scope
remain. Primary A/B/C decisions and multiplicity families are unchanged.

## Document and release plan

Use the latest supplied DOCX as the only editorial baseline. Clean and visible-
change outputs must have identical final-content projections. Unchanged text
is black, additions blue, and deletions red/struck through. Technical validation
must not be represented as independent human review. Supply supplementary
methods, all principal estimates, budget sensitivity, bibliographic evidence,
source data for publication figures and an official-guidance compliance checklist.
No missing author declaration or journal-submission action is inferred.
