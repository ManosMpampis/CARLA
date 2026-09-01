# AGENTS.md

## What this repo is

Python/PyTorch research code for time-series anomaly detection built around a
convolutional-pyramid JEPA: dense latent prediction learns how a machine's
telemetry behaves; anomalies are latent-prediction surprise localized at
sub-window granularity. The legacy contrastive pipeline was deleted at cutover
(see `specs/jepa-tsad-rebuild.md` and `DESIGN_TSAD_JEPA.md`); its history lives
on other branches, and `WORKFLOW_PRETEXT.md` is kept as a historical record of
that deleted pipeline. `DESIGN_TSAD_DUALSTREAM.md` is a standalone (unimplemented)
design doc for a dual-stream time×frequency architecture; `CONTEXT.md` is the
project glossary.

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

## Running

```bash
./venv/bin/python carla_jepa.py --config_env configs/env.yml \
    --config_exp configs/jepa/<arm>.yml --fname machine-1-1.txt \
    --version myrun        # omitting --version creates a timestamped run dir
```

- Stages dispatch on the YAML `stage:` key: `pretrain` (alias `pretext`),
  `adapt`, `score`. Arms are pure YAML under `configs/jepa/`.
- Runs resume automatically when `checkpoint.pth.tar` exists in the run dir;
  use a fresh version to start over.
- Adaptation requires `pretrained_from:` pointing at a stage-A checkpoint.
- Seed default is 4 (`seed:` in config).

## Verification

```bash
./venv/bin/python tests/verify_pretrain_tracer.py   # T02 synthetic tracer + resume
./venv/bin/python tests/check_scorer_handoff.py     # overlap aggregation + metric contract
./venv/bin/python tests/verify_arms.py              # T04 SIGReg / T05 EMA / T06 codebook
./venv/bin/python tests/verify_stages.py            # T07 masking/corpus / T08 adapt modes / T09 variants
./venv/bin/python tests/verify_calibration.py       # T10 train-only thresholds (uses smoke_adapt weights)
```

The SMD smoke chain (needs CUDA, minutes): pretrain → adapt → score with
`configs/jepa/smd_pretrain_smoke.yml`, `smd_adapt_smoke.yml`,
`smd_score_smoke.yml` on `machine-1-1.txt`.

## Current design facts

- Model wiring goes through registries only: `models.BACKBONE_REGISTRY`,
  predictor registry (`tcn|gru`) and anti-collapse registry
  (`none|sigreg|ema|codebook`) consumed via `utils/common_config.get_jepa_model`.
- Loss: dense L1 between predicted and stop-gradient target latents over all
  tokens/levels/horizons (+ λ·SIGReg when enabled; + codebook term on that arm).
- Scoring: per-position L1 error per pyramid level mapped to input timesteps,
  mean-fused; overlapping windows aggregated cover-count-aware in
  `utils/scoring.Scorer`; thresholds come ONLY from clean-train quantiles,
  optionally with calibrated fusion weights from injected-probe separation
  (`utils/scoring.Calibrator`). Test labels never enter tuning paths.
- Metrics stack (`metrics/**`) is frozen: scorer emits
  `(scores, start_idxs, end_idxs)`; `combine_all_evaluation_scores` returns the
  documented dictionary; honest vs point-adjust sections grouped in
  `metrics.json` and compared against `REFERENCE_NUMBERS.md`.
- Known-bug note: `models/convolutions.Conv1dSamePadding` assumes stride 1;
  strided blocks therefore live in `models/jepa_pyramid.StridedConvBlock`.
  Legacy backbone (`models/resnet_ts.py`) is unaffected and kept importable.
- AMP (`amp: true`) trains bf16-autocast on CUDA only; scoring always fp32;
  validation statistics always computed in eval mode.
