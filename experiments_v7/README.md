# V7 Scientific Validation of RL-Specific Value

This folder audits and corrects the V6 reinforcement-learning experiment without modifying V3, V4, V5, or V6.

## Purpose

V7 tests whether the V6 Q-learning disadvantage reflects a learning/convergence problem or a lack of robust sequential value in the simulator. It corrects reward attribution, expands state observability, adds terminal liquidation, audits Q-learning convergence, and compares corrected bandit and RL variants.

## Reproduce

From the project root:

```powershell
python -m unittest discover -s experiments_v7\tests -v
python experiments_v7\scripts\run_experiments_v7.py
```

## Main outputs

- `outputs/reports/scientific_report.md`: answers the five V7 scientific questions.
- `outputs/reports/methodological_corrections.md`: reward attribution, Markov structure, terminal liquidation, algorithms.
- `outputs/tables/v7_main_summary.csv`: main non-sequential/sequential comparison.
- `outputs/tables/v7_paired_differences.csv`: paired differences versus contextual bandit, rules, and heuristic scoring.
- `outputs/tables/v7_q_convergence_diagnostics.csv`: state coverage, unseen validation states, training budgets, alpha/gamma/epsilon sensitivity.
- `outputs/tables/v7_reward_attribution_summary.csv`: current-action, delayed-past, and terminal-liquidation reward components.
- `outputs/tables/v7_sequentiality_summary.csv`: policy interaction with sequentiality scale.
- `outputs/figures/*.png`: summary visualizations.

## Interpretation

The V7 results preserve the negative finding: corrected Q-learning and RL variants do not robustly outperform the corrected contextual bandit in the main scenarios. The result should be framed as simulation evidence, not real-platform evidence.
