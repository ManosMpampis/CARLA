# AGENTS.md

## What this repo is

Python/PyTorch research code for time-series anomaly detection (pretext representation learning + classification). **Not** the CARLA driving simulator despite the directory name.

- Two-stage pipeline: `carla_pretext.py` (representation learning) -> `carla_classification.py` (anomaly classification head).
- `experiments.py` (SMD) and `experiments_psm.py` (PSM) are batch drivers that import and call both stages' `main()` directly with `EasyDict` args.
- No test suite, lint, or CI for this code. Verification means running a short training config.
- `WORKFLOW_PRETEXT.md` is the authoritative deep-dive on the pretext flow, including confirmed bugs and an open TODO list. Read it before touching pretext/dataset/checkpoint code.

## Environment

- Python 3.11 (`.python-version`). Dependencies live only in the committed-path-but-gitignored `venv/`; there is no requirements.txt/pyproject.toml. Use `./venv/bin/python`. Torch 2.x with CUDA available on this machine.
- `.gitignore` excludes `*.csv`, `datasets/`, `results/`, `venv/` — result CSVs (e.g. `pretext_summary*.csv`) are local artifacts, do not expect them in git.

## Data and output paths

- `datasets/` and `results/` are **symlinks outside this repo** (`/home/manos/plaisio/...`). Never replace them with real directories or commit them.
- `configs/env.yml` sets `root_dir: results/`. All outputs go to `results/<dataset>/<version>/<fname>/pretext*/`.
- SMD data is read from `datasets/SMD/{train,test,test_label}/machine-*.txt` (see `data/SMD.py`, `utils/mypath.py`). PSM runs use `fname=""`.

## Running

```bash
./venv/bin/python carla_pretext.py --config_env configs/env.yml \
    --config_exp configs/pretext/carla_pretext_smd.yml \
    --fname machine-1-1.txt --version myrun
```

- Config = env yml + experiment yml merged in `utils/config.py:create_config`; driver scripts add overrides via `update_dictionary`.
- If `--version` is omitted it becomes a timestamp (`utils/config.py:25`), which silently creates a new run directory.
- Runs **resume automatically** if `checkpoint.pth.tar` exists under the same `<dataset>/<version>/<fname>/pretext*/` path. Reusing a version continues training; use a fresh version to start over.
- The pretext `Logger(delete_files=True)` prunes old TensorBoard epoch dirs when finalizing a resumed run.
- Seed is hardcoded (`set_seed(4)` in both entry scripts).
- Gotcha: `experiments.py`/`experiments_psm.py` import `evaluation`, which was deleted locally but still exists in git HEAD. Restore it (`git checkout -- evaluation.py`) or both drivers fail at import.

## Known config/code traps (verified in WORKFLOW_PRETEXT.md)

- Criterion factory (`utils/common_config.py:get_criterion`) supports only `pretext`, `classification`, `classification_e2e`, `tcl`. Some YAMLs under `configs/*/new_loss/` use `criterion: pretext_new` or pass unsupported `criterion_kwargs` and will raise.
- `PretextLoss` crashes when `crop: True` (no `random_crop` method); configs must set `crop: False`.
- Loss tensors default to `cuda`; CPU-only configs must explicitly set `device` in the experiment yml.
- Backbone requires one entry in `kernel_sizes` per `mid_channels` block; mismatched configs fail at model construction (e.g. `carla_pretext_smd_threeB_twoC.yml`).
