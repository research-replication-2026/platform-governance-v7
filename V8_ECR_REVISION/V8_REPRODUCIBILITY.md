# V8 ECR reproducibility

This directory reuses completed V8 A/B/C; it does not rebuild or overwrite V7. V8_RESULTS.csv is byte-identical to the verified 33,840-row primary dataset. The original protocol, seed amendment, manifests and verification records are copied in ORIGINAL_V8_PROTOCOL. FROZEN_SIMULATOR contains unmodified scientific code. V7_REFERENCE_DATA retains the archived CSV needed by the original runner's saved-result reuse step. The three historical labels remain in the original larger archive, not in primary inference.

## Recalculate estimates and figures

Use Python 3.12 and the exact versions in requirements.txt (the recorded execution used Python 3.12). From this directory:

```powershell
python -m pip install -r requirements.txt
python -B V8_STATISTICAL_ANALYSIS.py
python -B make_figures.py
python -B audit_delivery.py --scientific-only
```

The analysis rejects incomplete datasets. It uses independent training means, preserves matched conditions and evaluation indices, and keeps separate multiplicity families. ANALYSIS holds policy levels, paired contrasts, interaction tests, standardized effects, crossed-bootstrap sensitivity, nested-budget levels/changes, operational contrasts, and complete scenario parameters. FIGURES supplies scripted source data, vector EPS/PDF/SVG and 600-dpi lossless TIFF outputs.

## Fresh training in an isolated destination

```powershell
python -B reproduce_campaign.py --stage budget --output ../ECR_budget_fresh --workers 6
python -B reproduce_campaign.py --stage primary --output ../ECR_primary_fresh --workers 6
```

The output directory must not exist. The wrapper copies frozen sources and manifests; existing supplied data are never overwritten. The primary runner executes A, then B/C, then the independent-seed amendment and analysis. Its archived V7 reuse is explicit; original V8 targeted full-budget reproduction already checked 510 evaluations. The budget wrapper verifies the prospective manifest, runs a full-job prefix pilot, then the remaining jobs. Compare fresh raw scientific numbers rather than timestamps or figure metadata.

The original V8 public package is at https://github.com/research-replication-2026/platform-governance-v7/tree/e4fe365ba70ac0354fc69420bd4f291792bc036b/V8_minimal_extension . Its public_materialize.py restores losslessly compressed data and public_verify.py checks the published package. The new public ECR supplement similarly provides a materializer and compressed-file ledger when published.

## Budget design and exact prefixes

Sixty scenario/block jobs train three learners each along 720-episode trajectories. Snapshots at 180/360/720 produce 5,400 evaluation rows. The 1,800 rows at 180 are repeated checks, not additional independent observations. Fixed exploration decay of 60 episodes preserves the original 180-episode path. All prefix gates passed; BUDGET_MANIFEST, BUDGET_SEED_AUDIT, BUDGET_RUNTIME_BENCHMARK, BUDGET_EXECUTION.jsonl, checkpoints, diagnostics and curves retain provenance. The pilot took 38.47 seconds for one full scenario/block job; the remaining campaign used six processes and took about 804 seconds. The nested experiment is exploratory because it was designed after the primary V8 results were known.

## Manuscript provenance and review boundaries

SOURCE_PROVENANCE records the supplied original DOCX hash. build_ecr_manuscripts.py applies verified V8 paragraph corrections to that original and performs the ECR revision; it requires the supplied original and the prior derived paragraph-change ledger. Manuscript builders are local editorial tools, not dependencies of scientific reproduction. The visible redline uses explicit colored runs: unchanged black, added blue, deleted red/struck through. It is a visual comparison; its accepted-content projection, rather than Word's Accept All command, is compared with the clean document. audit_delivery.py independently recalculates estimates and checks text, table and image projection equality. This is separate computational recalculation, not independent human scientific review.

The full documentary source corpus and automated coding pipeline are not publicly packaged; documentary reproducibility is partial. Simulation probabilities, utilities and participation/trust responses are synthetic. Ten fixed evaluation seeds, privileged baseline aggregates, unequal linear-Q information, action-dependent RNG consumption, changing state complexity, the feedback scaling induced by zero encoding and limited learner/budget choices remain material limitations. No convergence, equivalence, population policy dominance or real-platform validation is inferred.
