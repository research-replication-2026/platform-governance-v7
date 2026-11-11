# V8 ECR audit — before additional experiments

Date: 11 November 2026. The latest explicitly supplied editorial source is
`ISF_Manuscript_IT & AI Rev (2).docx` in Downloads. Its fingerprint is recorded
in `SOURCE_PROVENANCE.json`. This revision will compare the visible-change
manuscript directly with that source. Earlier manuscript drafts are not the
editorial baseline. The previous V8 enrichment is useful implementation evidence,
not a replacement source.

## Evidence located and preserved

- Active V7: `experiments_v7/scripts/run_experiments_v7.py`; configurations,
  six scenarios, policy definitions and seed formulas are in the code itself.
- Existing V7 outputs include main replications, reward components, policy
  contrasts, ablations, sequentiality and stress/convergence diagnostics.
- V7 tests: five tests in `experiments_v7/tests/test_v7_environment.py`.
- Completed V8 extension: `V8_minimal_extension/`, including two frozen
  manifests, the protocol/amendment, 1,224 checkpoints and verified analyses.
- The primary V8 dataset has 33,840 evaluations: 108 learned-policy cells with
  30 independent blocks × 10 evaluations, plus 1,440 deterministic-reference
  evaluations. The 37,080-row archive retains historical labels separately.
- Public replication snapshot: GitHub commit
  `e4fe365ba70ac0354fc69420bd4f291792bc036b`. V7's subtree is preserved there.

No V7 file or completed V8 artifact will be changed. Existing A/B/C data will
be reused rather than represented as new runs. Copied evidence remains distinct
from new exploratory training-budget results.

## Established by implementation

The contextual bandit uses sample-average current-action reward only. It does
not reassign a delayed review outcome to its originating decision. Tabular Q
uses total transition reward, alpha .14 and gamma .93. Linear Q uses alpha .035,
gamma .93 and TD clipping ±5. Original main budgets are 180 × 100 decisions;
epsilon = .04 + .24 exp(−episode/60). Legal-action masks precede all policy
choices, making guard interventions generally zero by construction.

The raw V7 observation has 17 components. Bandit and tabular Q share an
11-component discretized key; linear Q uses the full vector. Hidden submission
harmfulness is not given to these policies. Several inherited outcome and
pending-incident aggregates are nonetheless privileged synthetic indicators;
their empirical availability is not validated. C3 adds current administrative
schedules/ages, not future random outcomes or harmfulness. Its tabular addition
is one binary workload-timing channel; its linear representation is richer.

B independently masks scalar reward-component reports with probability .25 or
.50 using a separate RNG and zero encoding. The evaluation objective and state
sensors remain complete. In expectation, non-guard teaching rewards are scaled
by 1−p; B therefore does not isolate all meanings of organizational feedback
quality. C changes observations while holding feedback complete. These are
separate experiments, not a factorial design.

Two terminal defects were already corrected only in V8: terminal continuation
bootstrap and incidents generated during terminal review/check settlement.
Exact V7 semantics are retained as a distinct stratum. The original three labels
have cross-algorithm/cross-group episode-seed overlap; primary labels 10001–10030
have disjoint episode-seed unions across blocks. Training/evaluation seeds are
separate. Shared evaluation seeds are not perfectly shared action-independent
shocks because the inherited RNG consumes draws conditional on decisions.

## Established results versus manuscript assertions

The completed A analyses give twelve negative Q-minus-bandit reward contrasts
in each implementation stratum after within-family Holm correction. The six B
and six C primary interaction tests establish no moderation after adjustment.
These are conditional simulator estimates, not empirical app-marketplace effects.

The supplied abstract's assertion that immediate feedback is more actionable,
and practical claims presenting traceability as an established performance
condition, exceed the interventions' results. They require qualification.
Historical linear-Q favorable rankings with three runs must not replace the
expanded corrected results. The 17 raw variables must not be described as the
effective tabular information set. The 300 evaluations must not become 300
independent training runs.

## Remaining scientific uncertainty: training budget

Equal transition budgets are fairness controls, not convergence evidence.
Historical V7 40/120/360-episode diagnostics use only two seeds, change effective
seeds with budget and vary hyperparameters. They do not exclude a finite-budget
explanation of the main ranking. It would be inaccurate to claim otherwise.

An additional bounded exploratory check is therefore specified prospectively
in `V8_PROTOCOL.md`: extend exactly the same training trajectories to 360 and
720 episodes in two selected sequential scenarios, with 30 independent blocks,
all three learners, common evaluation panels and no tuning. A fixed exploration
decay preserves the first 180 episodes exactly. Prefix reproduction against the
completed V8 dataset is a gate before the expanded run. Even a persistent gap
at 720 episodes cannot prove convergence or exclude benefits at other budgets.

## Editorial and reproducibility gaps

Official ECR instructions require a 100–150-word abstract, numbered citations,
an executive summary and corresponding-author ORCID. The current source has a
longer abstract and author–year citations. Journal identity and instructions
were checked on the official Springer journal 10660 pages, not the similarly
named Journal of Electronic Commerce Research.

An older title page names Thierry Isckia and François Acquatella and declares
no funding/conflicts. These are previous-draft statements, not independently
verified current declarations. Author contributions, ORCID and current ethical
determination are not established. A clarification has been requested; missing
facts will remain explicitly pending rather than invented. Reference metadata
will be verified separately using primary bibliographic/publisher records.

Final checks will use an audit script separate from the manuscript generator,
including a visible-change projection test. This is an independent computational
check, not a claim of independent human scientific review.
