# AGENTS.md

## What this repo is

Python/PyTorch research code for time-series anomaly detection built around a
lewm: dense latent prediction learns how a machine's
telemetry behaves; anomalies are latent-prediction surprise localized at
sub-window granularity. At the same time trained actions on how the timeseries change overtime can propose only in normal behaviour.

## Environment

- Python 3.14 venv in `venv/` (no requirements.txt). Use `./venv/bin/python`.
  Torch 2.x, CUDA available on this machine.
- No test framework by convention: verification = short training-config runs
  plus committed assertion scripts under `tests/` (plain python, no pytest).
- `.gitignore` excludes `*.csv`, `datasets/`, `results/`, `venv/`.

## Data and output paths

- `datasets/` and `results/` are **symlinks outside this repo**. Never replace
  them with real directories or commit them.
- `configs/env.yml` sets `root_dir: results/`. JEPA runs write to
  `results/<dataset>/<version>/<fname>/jepa*/`: `checkpoint.pth.tar`,
  `model.pth.tar`, `calibration.json`, `scores.npz`, `metrics.json`, TensorBoard.
- SMD read from `datasets/SMD/{train,test,test_label}/machine-*.txt`
  (`data/jepa_dataset.py`; headerless CSV — reads use `header=None`).
- `pretext_results/` preserves historical pretext summary CSVs.

## Agents file paths

The agents can make a memory bank or md file or photos or any other file type they prefer inside a dedicated folder that they know that they can use. All these kind of files like, helping scripts for writing and extract info, glossary.md, generated tickets and specs report etc will be there. The initial directory will be AGENTIC_DOCS and there will have all the main files it need to understand the repo, the specs and new tickes. Then inside there the agents will make new directories when they think that a new piece of info need to be saved that is not already be part of specific themes that are already in another directory.
An example is, in a conversation we have the agend help with the report of different experiments but already in the AGENTIC_DOCS we have the ARCHITECTURE_EXPERIMENTS. There may have some initial architecture exploration about the experiments but the report is something totaly different and MUST be writen elseware. A small md file should be generated in each new folder in order to be easier to distinguish what is good to be written or not in there.

Keep in mind that this is not for the actual scripts. If I said to you to make something related to already made scripts in the codebase you still make it in the codebase and tested it properly.
