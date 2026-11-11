# Platform Governance — V7 archive and V8 extension

Replication materials for “More Artificial Intelligence, Better Governance?
Challenging the Promise of Sequential Learning in Digital Platforms”.

The original `experiments_v7/` directory and documentary event register are
preserved. `V8_minimal_extension/` adds verified robustness and mechanism tests.

## Main findings

- Thirty independent paired training blocks per learned-policy cell; ten fixed
  evaluation seeds per block. The three historical groups are diagnostic only.
- The tested tabular Q-learning policy has lower reward than the contextual
  bandit in all twelve main environment–scenario cells, both with exact V7
  semantics and after two terminal-handling corrections (Holm-adjusted tests).
- Neither reward-feedback loss nor the tested administrative-observation changes
  establishes moderation of that gap after prespecified multiplicity adjustment.
- Reward rankings do not imply complete governance dominance or intrinsic
  inferiority of reinforcement learning. Operational trade-offs remain visible.

See the [results summary](V8_minimal_extension/V8_RESULTS_SUMMARY.md),
[primary reward contrasts](V8_minimal_extension/PRIMARY_REWARD_CONTRASTS.csv),
[mechanism contrasts](V8_minimal_extension/PRIMARY_MECHANISM_CONTRASTS.csv), and
[execution report](V8_minimal_extension/V8_EXECUTION_REPORT.md).

## Reanalyze supplied V8 data

Python 3.12.10 was used. From the repository root:

```sh
python -m pip install -r V8_minimal_extension/requirements-v8.txt
python -B V8_minimal_extension/public_materialize.py
python -B V8_minimal_extension/public_verify.py
python -B -m unittest discover -s experiments_v7/tests -v
python -B -m unittest discover -s V8_minimal_extension -p validate_v8.py -v
python -B V8_minimal_extension/V8_STATISTICAL_ANALYSIS.py
python -B V8_minimal_extension/public_verify.py
```

The lossless gzip archives restore byte-identical primary/all-executed CSVs,
training diagnostics/curves and detailed analysis tables. Compression hashes and
uncompressed hashes are recorded in `PUBLIC_DATA_MANIFEST.json`. No private
manuscript archive, machine checkpoint, or external service is needed for
reanalysis. The public verifier independently reconstructs training-level
contrasts, checks sample counts, reward additivity and terminal settlement,
and verifies scientific source hashes and seed-block separation.

## Recompute all training runs

```sh
python -B V8_minimal_extension/public_reproduce.py --output V8_fresh_reproduction --workers 6
```

This creates a new sibling directory and runs every full-budget A/B/C job plus
the seed-amendment jobs, then recomputes analyses. It refuses an existing output
directory. It is substantially more expensive than reanalysis: the recorded
campaign required roughly one hour on its host, with load-dependent runtime.
Worker count changes scheduling, not budgets or seeded numerical definitions.
The archived V7 pipeline can still be run with its original README commands in
an isolated copy; its output-writing main script is not needed for V8 reanalysis.

## Design and interpretation

Read the [prospective protocol](V8_minimal_extension/V8_EXPERIMENTAL_PROTOCOL.md)
with [amendment 01](V8_minimal_extension/V8_PROTOCOL_AMENDMENT_01.md).
The primary dataset has 33,840 rows; the all-executed archive has 37,080.
Historical labels 1101/1201/1301 remain separate because their episode streams
partly overlap across algorithms and groups. Primary labels 10001..10030 have
disjoint episode-seed unions across blocks.

Inference uses independent training means, not 300 pseudo-independent evaluations.
Intervals condition on ten fixed evaluation seeds; crossed bootstrap intervals
provide a secondary sensitivity. Common seeds do not ensure perfectly shared
exogenous shocks in the inherited action-dependent simulator.

The documentary register contains 32 events motivated by 50 public sources.
The full source collection and automated classification-validation pipeline are
not included, so documentary reproducibility remains partial. Transition
probabilities, trust, participation and delays are synthetic, uncalibrated
assumptions. The results are not estimates of real marketplace policy effects.
The submitted manuscript itself is not part of this public replication package.
