# From-scratch AE and VAE reconstruction comparison

Two matched, independent training entries compare common reconstruction
detectors with the LeWM approach:

```bash
./venv/bin/python carla_ae.py --fname machine-1-1.txt --version ae_trial
./venv/bin/python carla_vae.py --fname machine-1-1.txt --version vae_trial
```

The steered reconstruction entry (`carla_recon.py`) uses the same two
evaluation procedures and score-mode choices. Its encoder initialization
and L1 training objective remain distinct from the from-scratch arms.

Both arms start with random weights and train the same time-domain ResNet
encoder and mirrored decoder on normal SMD windows. The deterministic AE
minimizes mean MSE. The VAE adds one 1×1 projection for each of posterior
mean and log variance at the encoder output, uses reparameterized samples
during training, and minimizes `MSE + beta * mean KL(q(z|x) || N(0,I))` with
fixed configurable `beta: 0.3` by default. At inference, the VAE decodes its
posterior mean. All reconstruction arms use configurable `score_mode: l1`
by default; alternatives are `l2` and `mse`. The score choice does not change
the training objective. Per-timestep reductions are mean absolute error,
root mean squared error, and mean squared error across channels respectively.

Evaluation at `eval_every` epochs (default 1), the final epoch, and any
intervening new validation best uses two procedures:

1. **Window:** one scalar per window, the mean of configured per-timestep
   scores. A window is anomalous if this scalar exceeds its calibrated
   threshold; the evaluation label is anomalous if any contained test
   timestep is labeled anomalous.
2. **Time series:** the configured channel reduction at each window position,
   then a cover-count mean across overlapping windows. It yields one score
   per input timestep.

Each procedure gets its own threshold: the configured quantile (default
0.99) of scores on the held-out clean validation tail. A score strictly
greater than its threshold is predicted anomalous. An oracle threshold
that maximizes point F1 on test labels is logged and selected separately.
F1 and event F1 use thresholded predictions without point adjustment.
VUS PR and VUS ROC use continuous anomaly scores and therefore have no
threshold variant. Window VUS uses an event span in window units derived
from `eval_window_size / stride`; time-series VUS uses timestep units.

Each arm writes into `results/<dataset>/<version>/<machine>/jepa_<arm>/`:

- `best_validation_loss.pth.tar`: normal-only validation selection.
- `best_test_<procedure>_<calibrated|oracle>_f1.pth.tar`: four explicitly
  test-selected F1 files across the two procedures and thresholds.
- `best_test_<procedure>_vus_<pr|roc>.pth.tar`: four test-selected,
  threshold-independent VUS files.
- `last.pth.tar`: resume state; `best_metrics.json` and `epochs.jsonl`:
  selection summary and per-epoch values; `tensorboard/`: training,
  validation, and test metric curves.

Best checkpoints carry model, optimizer and scheduler states, epoch, train
and validation losses, both procedure evaluations, selection metadata,
window length, stride, model construction arguments, normalizer statistics,
and the scoring/overlap rules. Test-selected checkpoints are oracle analyses;
the validation-selected checkpoint is the fair comparison result.
