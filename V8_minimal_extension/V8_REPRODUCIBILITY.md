# Public-package reproducibility

Use the commands in the repository-root README. Run `public_materialize.py`
before reanalysis and `public_verify.py` before and after it. To recompute all
fits use `public_reproduce.py` in a new output directory. It retains all budgets
and source/seed manifests. Do not use the private-workspace `verify_outputs.py`
or `validate_v8.py` main entry point: they audit private manuscript archives and
machine checkpoints that are intentionally not distributed. The behavioral
tests in `validate_v8.py` remain public and run through unittest discovery.

The archived execution report documents checks performed in the complete
original workspace. The public verifier reproduces scientific data/code checks
without claiming to reconstruct that private file-preservation audit.

The published V7 script differs from the local frozen copy only in report
wording and output-path presentation. All scientific module nodes, including
environment dynamics, reward definitions, policies, training, evaluation and
main experiment configuration, match after excluding three presentation-only
functions; `public_verify.py` checks this equality. The archived V7 main CSV
matches the local source exactly after LF/CRLF line-ending normalization.

The frozen V8 protocol and computational sources are copied byte-for-byte.
Portable public wrappers and this documentation are new packaging code; they
do not change the experiments. `.gitattributes` preserves scientific source
bytes across platforms. CSV gzip compression is lossless, with mtime zero.
