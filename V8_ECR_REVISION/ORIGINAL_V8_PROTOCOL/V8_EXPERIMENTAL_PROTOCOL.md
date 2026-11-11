# V8 experimental protocol — frozen before the expanded campaigns

Date: 2026-11-11. Status: specified after code audit and targeted V7 reproduction,
before inspecting any new full-campaign outcomes. This is a local prospective
protocol, not an externally registered preregistration. V7 outcomes are known.

## Research questions and estimands

RQ1: do the V7 policy contrasts persist across 30 independent training runs?
RQ2: does missing reward feedback change Q-learning's performance relative to
the contextual bandit? RQ3: does administrative state information change that
relative performance? The primary outcome is true total governance reward per
decision, including all delayed and terminal effects applicable to that arm.
The primary policy contrast is Q minus bandit; linear Q minus bandit is secondary.
No assumption of a positive or negative sign is imposed. Performance is bounded
to the simulator, algorithms, information sets and budgets examined.

## Preservation and implementation strata

The original V7 code and results remain untouched. `vendor/v7_frozen.py` is a
byte-for-byte copy. Imports and tests suppress bytecode writes. Its full main
entry point is never invoked; V8 calls only its environment/policy functions.
An exactly equivalent scalar discretizer accelerates tabular state encoding;
random and boundary-value tests plus full-budget targeted reproduction verify it.

- `A_v7_exact`: exact V7 terminal/learning semantics, six original scenarios,
  both environments, 30 training seeds. Reuse three compatible saved training
  groups and deterministic references. Execute 27 additional training groups.
- `A_corrected`: same grid and budget, 30 training groups; cancel terminal TD
  bootstrap and pay incidents generated during terminal resolution. Execute
  every corrected learner. The two changes form one implementation sensitivity;
  this design does not isolate their individual effects.
- B/C: only the corrected implementation. Their full-feedback/V7-observation
  control is the corresponding `A_corrected` cell, reused without retraining.

Do not pool legacy and corrected strata. A's comparison with the old V7 tables
uses exact V7 semantics; substantive B/C interpretations use corrected dynamics.

## Scenarios, policies and budgets

A retains all six `SCENARIOS` and nonsequential/sequential mechanisms. B and C
are separate one-factor experiments in the sequential environment, on original
scenario indices 0, 1 and 3: balanced reference, high security pressure and
constrained review capacity. This selection spans reference, high harm/penalty
and resource constraints; it is specified before new outcomes. It does not
include the full V7 stress/sensitivity grid.

Every learner receives 180 episodes of 100 decisions (18,000 transitions) and
10 evaluation episodes of 100 decisions. Every cell has 30 independent training
runs for each of bandit, Q and linear Q. Preserve V7's alpha, gamma, epsilon,
initial values, masks, tie-breaking, action definitions and reward coefficients.
Tabular Q: alpha=.14, gamma=.93; linear Q: alpha=.035, gamma=.93, clipped TD
error ±5; bandit sample means. No tuning on evaluation outcomes. No early
stopping. Equal transition budgets do not imply equal computational cost or
equal statistical complexity. Q fine and the privileged oracle are not expanded.
They remain available in V7 as diagnostics, not deployable comparators.

Four unchanged rule definitions are retained: permissive, restrictive, security
rule and heuristic score threshold. Evaluate each rule once per condition,
scenario, environment and evaluation seed (`train_seed=0`). Their repeated
appearance in learned-policy contrasts never creates independent rule training.

## Randomness and pairing

Training labels: 1101,1201,1301,10001..10027. For the first three, V7's effective
seed formula is retained. For each additional label j (zero-based among the 27
new labels), base = 100000000+1000000j. Add environment offset 500000 for
sequential, scenario offset 10000i, and policy offset 101/202/404. Episode
seeds then add 13e/17e/19e for bandit/Q/linear. Audit all effective episode
seeds within each policy for cross-training collisions and evaluation overlap.
The first three labels preserve V7's algorithm-specific streams; coincidences
between different algorithms are possible and are not evidence of shared
exogenous trajectories. Seed independence is a computational design property,
not a claim about physical independent random sources.

Evaluation seeds retain V7 exactly: 20261008 + environment offset + 10000i +
(5101+37k), k=0..9. They are common across policy/condition/training labels.
This supports seed-paired contrasts but not identical action-independent shocks:
V7 uses one action-dependent RNG and endogenous arrival distributions. We do not
reengineer that simulator feature. Inference is conditional on these fixed ten
evaluation seeds; it does not estimate uncertainty over all possible evaluation
scenarios. A crossed resampling sensitivity over training and evaluation indices
will quantify sensitivity to this fixed evaluation set.

Feedback masking uses a separate generator: seed 4000000000+10000i+training
label's position in the 30-label list. Four draws per transition regardless of
policy or loss level give nested masking schedules across conditions and shared
availability schedules across algorithms. This generator never touches the
physical environment RNG or observations.

## Experiment B: reward-feedback availability

B1 = `A_corrected`, loss=0; B2=`B_loss25`, loss=.25;
B3=`B_loss50`, loss=.50. At every training transition, independently hide each
of four scalar feedback reports (current action, delayed past, collective state,
terminal liquidation) with probability p. A missing report contributes zero
to the teaching signal; no inverse-probability correction or imputation.
Actual realized missing counts are recorded. Guards remain known enforcement
costs for Q (normally zero because policies respect masks).

The bandit always updates only from the (possibly missing) current-action report.
Q/linear update from the available sum of reports plus guard penalty. The bandit
never receives delayed reports in its update. Existing physical delays remain
unchanged; feedback is available or absent when the corresponding V7 component
would be delivered, with no additional reporting lag. This is component-report
loss, not event-level incident under-reporting and not a loss of state sensors.
Components may contain both benefits and costs, so missingness can change the
effective expected training objective/scale; evaluation always uses complete
true reward. The unchanged V7 state may still convey outcome information even
when a scalar reward report is absent. These boundaries must accompany claims.
During evaluation policies are frozen; no learning or reward masking is needed.

## Experiment C: administrative observability

C1=`A_corrected`: original raw 17-vector and original tabular 11-component key.
C2=`C_reduced`: zero indices 4,5,8,11,12,13,14,15,16: review load/capacity,
pending appeals, automatic load, queue risk/delay summaries, incident queue
load/severity and appeal queue load. Keep risk, uncertainty, participation,
recent acceptance, exposure, cumulative incidents/restrictions and trust.
Masks remain available for legal action selection. This channel can itself
reveal saturation: it is not removed because safety constraints must remain.
Zero is a specified constant encoding; policies are trained and evaluated with
the same mask, without pretending that zero represents observed empty queues.

C3=`C_enriched`: append elapsed horizon fraction, number of auto checks due next
transition/max queue, number of reviews due next transition/max queue, and oldest
administrative job age/horizon (all clipped to [0,1]). These are schedules/ages
of jobs already submitted. They do not reveal future arrivals, future random
outcomes, harmfulness or incident schedules beyond what V7 already discloses.
For tabular bandit/Q append one binary `any administrative job due next step`
channel to the original key. This adds at most a factor of two to the state
space. The other new raw channels are available only to linear Q. Thus the
primary bandit/Q contrast has identical information and representation per
condition; linear Q remains a secondary representation diagnostic.

Reduced information can improve sample efficiency; enriched information can
increase sparsity. Report visited states, state-actions and unseen evaluation
states by condition. Equal data budgets and the twofold bound control complexity
partly; they do not disentangle information gain from learning difficulty.
V7's existing incident-queue and true-outcome aggregates are inherited privileged
simulation indicators. C3 adds administrative information, not an oracle; C1
should still not be called an empirically realistic production information set.

## Statistical analysis fixed before campaign outcomes

Aggregate ten seed-paired evaluations within each independent training label.
Report mean, between-training sample SD, raw paired differences, paired
standardized effect dz = mean difference / SD of training differences,
and two-sided Student t 95% intervals over 30 training means (df=29).
For deterministic levels report mean/SD across the ten evaluation seeds,
with inference status descriptive: they have zero independent training runs.
For a learned-minus-rule contrast use the rule's matching evaluation mean
and the learner's 30 training means; intervals are conditional on the fixed
rule evaluation panel.

RQ1 primary family: 12 Q-minus-bandit reward contrasts in `A_v7_exact`.
Secondary corrected robustness family: corresponding 12 `A_corrected` contrasts.
RQ2 primary family: for each of three scenarios and two loss levels,
[(Q−bandit)_loss − (Q−bandit)_full], six contrasts. RQ3 primary family:
the analogous six contrasts for reduced/enriched vs original observation.
Pair conditions by training labels and evaluation seeds. Report raw p-values,
Holm-adjusted p-values within each declared family, pointwise 95% t intervals,
and Bonferroni simultaneous 95% family intervals for primary reward contrasts.
No claim of a confirmed family-wise result based solely on a pointwise interval.
The fixed evaluation-panel condition applies to these adjusted inferences too.

For all cells and learned-policy/rule contrasts report legitimate accepted,
security incidents, rejected legitimate innovation, human review workload,
governance expenditure, safety interventions, harmful released, trust,
participation and state-coverage diagnostics. These operational outcomes and
linear-Q contrasts are secondary/descriptive, with pointwise uncertainty only.
Positive reward without favorable security/generativity is not overall dominance.
Report all scenarios, including results unfavorable to the motivating theory.

As an explicitly secondary sensitivity, use a crossed bootstrap that independently
resamples 30 training indices and 10 evaluation indices while preserving paired
policy/condition cells (5,000 draws, seed=819731). This addresses both variation
sources for the chosen ten evaluation draws; it cannot manufacture a large,
independent scenario sample or perfect common random numbers.

## Execution gates, resources and status

Before expensive execution: preserve hashes, pass V7 and V8 behavioral tests,
reproduce saved deterministic cells plus full-budget reference learners,
benchmark elapsed times, audit effective seeds, freeze this protocol and complete
JSON manifest. The available machine has 12 logical CPUs and approximately
15.6 GiB RAM; tabular/NumPy algorithms use CPU. GPU is unnecessary. Use six
worker processes and single-thread numerical libraries to preserve host capacity.
Estimate total runtime from recorded benchmark, allowing parallel overhead.

Each scenario-training job has an atomic JSON checkpoint; journal success/errors
and regenerate CSVs periodically. Resume uses complete checkpoints and verifies
source/protocol hashes. A executes first, then B/C. No feedback×observability
interaction is planned initially; add only under a separately dated exploratory
amendment with a reason grounded in the separate B/C findings.

Deliverables distinguish planned/running/completed/verified. A campaign stopping
early must not be presented as 30 completed training runs. Report counts and
missing cells directly from checkpoint files. Preserve all original files and
verify SHA-256 after execution.
