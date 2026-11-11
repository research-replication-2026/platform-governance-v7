# V8 ECR simulation supplement

This directory adds a prospective exploratory nested-budget check to completed V8 A/B/C. It does not replace V7 or the original V8 archive. The primary policy comparison uses 30 independent training blocks and ten fixed evaluation seeds; snapshots at 180/360/720 episodes share trajectories.

In two sequential scenarios, Q-learning remains below the contextual bandit at the larger budgets. At 720 episodes the relative gap improves under security pressure but worsens in the reference scenario. These paired changes survive a separate exploratory Holm family. No convergence or universal policy ranking is claimed. Feedback/observation interactions in the main extension remain unconfirmed after their own family corrections.

From this directory:

```powershell
python -B public_materialize.py
python -B V8_STATISTICAL_ANALYSIS.py
python -B make_figures.py
python -B audit_delivery.py --scientific-only
```

Fresh training (use a new output directory):

```powershell
python -B reproduce_campaign.py --stage budget --output ../ECR_budget_fresh --workers 6
python -B reproduce_campaign.py --stage primary --output ../ECR_primary_fresh --workers 6
```

The original full-budget primary reproduction can also be run using the existing V8_minimal_extension/public_reproduce.py package. No manuscript or author completion form is included here. See V8_RESULTS_REPORT.md and V8_REPRODUCIBILITY.md for scope and limitations. Public source hashes and lossless compressed-data hashes are recorded in PUBLIC_ECR_MANIFEST.json. Re-rendered figure metadata can vary.
