# AGENTS.md

## What this repo is

Python/PyTorch research code for time-series anomaly detection with fully
convolutional AE/VAE baselines and composable LeWM latent predictors. The
approved current design is `AGENTIC_DOCS/specs/lewm-conditional-experiments.md`;
`AGENTIC_DOCS/CONTEXT.md` is the project glossary. Older JEPA/dual-stream documents
are historical designs, not the source of truth for the current training entry.

## Environment

- Use `./venv/bin/python` (no requirements.txt). Check the actual environment
  rather than assuming the Python version from historical documents. Torch 2.x
  and CUDA are installed on this machine.
- No test framework by convention: verification = short training-config runs
  plus committed assertion scripts under `tests/` (plain python, no pytest).
- `.gitignore` excludes `*.csv`, `datasets/`, `results/`, `venv/`.

## Data and output paths

- Preserve `datasets/` and `results/` exactly as found; datasets is an external
  symlink. Never replace either path or commit dataset/run contents. Tests use
  temporary output roots instead of writing through external paths.
- `configs/env.yml` sets `root_dir: results/`. Experiment paths isolate dataset,
  framework, experiment, optional version and series. Runs contain checkpoints,
  calibration, score/metric artifacts and TensorBoard data.
- SMD read from `datasets/SMD/{train,test,test_label}/machine-*.txt`
  (`data/jepa_dataset.py`; headerless CSV — reads use `header=None`).
- `pretext_results/` preserves historical pretext summary CSVs.

## Running

```bash
./venv/bin/python train.py --config_exp configs/training/lewm.yml \
    --dataset smd --fname machine-1-1.txt --version myrun
```

- Select `framework: ae|vae|lewm` with grouped YAML under `configs/training/`.
  `train.py --score` scores a saved experiment; use its `--help` for selection
  and concurrency options. Existing run checkpoints resume automatically.
- PSM needs no filename. SMD `all` means independent jobs per machine;
  `all-one` means one shared model. TSB-U/M support a series or a CSV manifest
  with a `file_name` column and use the declared training prefix.
- Seed default is 4. Validation uses separate reproducible RNG state; test
  data does not fit preprocessing or calibrated thresholds.

## Verification

```bash
./venv/bin/python tests/verify_conditional_models.py
./venv/bin/python tests/verify_conditional_runtime.py
./venv/bin/python tests/verify_conditional_transfer.py
./venv/bin/python tests/verify_conditional_suites.py
```

The independent acceptance report lives under
`AGENTIC_DOCS/reports/lewm_conditional_acceptance/`. Distinguish generated-format
fixtures from real collection runs; installed collections can differ by machine.

## Current design facts

- Registry assembly goes through `models.get_model`. LeWM composes an engine,
  optional backbone, encoder, action, predictor and optional raw head.
- Masked LeWM predicts clean latents from a corrupted view of the same window;
  its auxiliary predicts the soft mask, supervised by synthetic corruption
  labels. Its predictor uses one STFT and a learned multiscale pyramid.
- Guided LeWM observes the future crop to form Q, with paired historical K/V;
  a small transpose convolution predicts future latents. This is delayed
  conditional comparison. Default history/future ratio is 2:1.
- Both LeWM presets default to KL latent prediction and L1 scoring, selected
  independently. KL/L1/MSE/L2 are selectable; L2 means channel RMS per token.
  A shared encoder receives gradients through both views; dual SIGReg prevents
  collapse. There is no stop-gradient/EMA teacher or default raw reconstruction.
- A checkpoint supports variable temporal lengths and a fixed sensor schema.
  Feature strides constrain legal lengths; coarse scalar scores interpolate
  to raw target timesteps. Overlap aggregation uses cover counts.
- Feature transfer preserves architecture and task semantics as well as weight
  shapes. Frozen modules stay eval, including their normalization buffers.
- Calibration uses clean training scores only. Test-oracle F1 checkpoint
  selections are explicit diagnostics separate from best-validation (or
  best-training when validation is unavailable) selection. `metrics/**` is frozen.
- AMP (`amp: true`) trains bf16-autocast on CUDA only; scoring always fp32;
  validation statistics always computed in eval mode.

## Agents file paths

The agents can make a memory bank or md file or photos or any other file type they prefer inside a dedicated folder that they know that they can use. All these kind of files like, helping scripts for writing and extract info, glossary.md, generated tickets and specs report etc will be there. The initial directory will be AGENTIC_DOCS and there will have all the main files it need to understand the repo, the specs and new tickes. Then inside there the agents will make new directories when they think that a new piece of info need to be saved that is not already be part of specific themes that are already in another directory.
An example is, in a conversation we have the agend help with the report of different experiments but already in the AGENTIC_DOCS we have the ARCHITECTURE_EXPERIMENTS. There may have some initial architecture exploration about the experiments but the report is something totaly different and MUST be writen elseware. A small md file should be generated in each new folder in order to be easier to distinguish what is good to be written or not in there.

Keep in mind that this is not for the actual scripts. If I said to you to make something related to already made scripts in the codebase you still make it in the codebase and tested it properly.
