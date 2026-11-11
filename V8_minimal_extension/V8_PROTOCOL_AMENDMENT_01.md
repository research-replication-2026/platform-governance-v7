# Prospective amendment 01 — independent paired training groups

2026-11-11, recorded during A execution, before inspecting any expanded reward
or interaction estimates. Reason: an additional seed-structure audit found that
the three historical labels 1101/1201/1301 have disjoint episode streams within
each policy but share some episode seeds across different algorithms AND
different training groups. In each scenario/environment, bandit/Q has six such
cross-group pairings sharing 9–11 episode seeds each. Other algorithm pairs show
similar coincidences. This can couple the first three paired contrast units.
The original audit's within-policy and train/evaluation checks remain correct,
but do not establish independence of the paired policy-contrast groups.

## Amendment before new outcome analysis

Retain the frozen original protocol, source and 1,116-job manifest unchanged.
Keep all three historical groups in the all-executed diagnostic archive.
Add labels 10028,10029,10030 to every learned condition/scenario/environment:
108 additional jobs, each containing three full-budget policies and ten
evaluations each. The primary analysis will use labels 10001..10030 only.
This gives 30 independent paired training blocks, with million-integer-separated
seed namespaces, without discarding or overwriting any original execution.

The new labels continue the existing seed formula. Their actual base seeds are
127000000,128000000,129000000, plus the same environment, scenario and policy
offsets and episode strides. Masking seeds continue at positions 30,31,32 of
the extended label list, shared across algorithms and conditions. All budgets,
scenarios, hypotheses, multiplicity families, reward definitions, information
manipulations, effect calculations and analysis decisions remain unchanged.

`V8_RESULTS.csv` contains only the primary 30-block dataset, with label 0 rules.
`V8_RESULTS_ALL_EXECUTED.csv` retains all 33 learned labels plus the same rules.
`V8_HISTORICAL_SEED_DIAGNOSTICS.csv` retains the three historical labels across
all conditions. Corresponding all-executed diagnostics/curves are preserved.
No outcome-dependent replacement or selective seed removal is permitted.

The resulting execution comprises 1,224 total jobs, 1,176 newly executed jobs,
3,456 newly fitted full-budget policies and 108 archived policy fits. The main
dataset remains 33,840 rows; the all-executed dataset contains 37,080 rows.
Each primary learned cell contains exactly 30 newly trained policies and 300
evaluations. Compatible historical V7 outcomes remain reused in the diagnostic
archive and validated by the original targeted reproduction.

The primary seed audit must verify disjoint episode-seed unions across different
training blocks INCLUDING all compared algorithms, as well as separation from
evaluation seeds. `run_v8_replacements.py` implements the additive manifest,
checkpoint collection and primary selection. Source preservation and the frozen
original protocol remain verifiable. Original `run_v8.py --collect` collects
the original manifest only; always follow it with amended collection before
analysis, or use `run_v8_replacements.py --collect` directly.
