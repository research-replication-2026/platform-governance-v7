# Platform Governance V7 — Replication Package

## Overview
This repository supports a study comparing static governance
rules, contextual bandits, and sequential reinforcement learning
in digital platform ecosystems.

## Materials
- experiments_v7/scripts: simulation code
- experiments_v7/tests: unit tests
- experiments_v7/outputs: numerical results and figures
- data/processed: register of 32 documented governance events
- requirements.txt: Python dependencies

## Reproduction
Install dependencies:
python -m pip install -r requirements.txt

Run tests:
python -m unittest discover -s experiments_v7/tests -v

Reproduce experiments:
python experiments_v7/scripts/run_experiments_v7.py

## Research scope
The simulation uses synthetic assumptions about transitions
and delayed consequences. These assumptions are not calibrated
from observed platform decisions.

The event register is supplied, but the full documentary
source collection and coding-validation pipeline are not
included. Documentary reproducibility is therefore partial.

## Peer review
This package is prepared for anonymous scholarly review.
