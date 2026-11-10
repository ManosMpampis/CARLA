"""Render the independent acceptance summaries from authoritative JSON results."""

import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


HERE=Path(__file__).resolve().parent


def rows(name):
    return json.loads((HERE/name).read_text())


def detail(values,name):
    return next(row["detail"] for row in values if row["case"]==name)


models=rows("model_checks.json")
runtime=rows("runtime_checks.json")
transfer=rows("transfer_checks.json")
suites=rows("suite_checks.json")
commands=rows("commands.json")
assert all(row["status"]=="passed" for group in (models,runtime,transfer,suites) for row in group)
runs=detail(runtime,"training and score reload")["runs"]
fig,axes=plt.subplots(2,2,figsize=(13,8),layout="constrained")
for ax,(name,run) in zip(axes.flat,runs.items()):
    history=run["history"]
    epochs=[row["epoch"] for row in history]
    train=[row["losses"]["train"] for row in history]
    validation=[row["losses"]["validation"] for row in history]
    ax.plot(epochs,[row["loss"] for row in train],"o-",label="train total",linewidth=2)
    ax.plot(epochs,[row["loss"] for row in validation],"s--",label="validation total",linewidth=2)
    names=("mse","kl") if name in ("ae","vae") else ("pred_loss","aux_loss","sigreg","sigreg_tgt")
    for term in names:
        ax.plot(epochs,[row[term] for row in train],".-",label=f"train {term}",alpha=.75)
    ax.set(title=name,xlabel="epoch",ylabel="loss (components unweighted)",xticks=epochs)
    ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle("Conditional LeWM acceptance: two-epoch plumbing runs (not benchmark comparisons)")
fig.savefig(HERE/"component_curves.png",dpi=160)
plt.close(fig)

lines=["# Conditional LeWM independent acceptance", "",
       "Status: **passed, 2026-11-10** — 29 assertion groups across four plain-Python scripts. "
       "The implementation and tests were owned by different agents. No source implementation was changed by the acceptance owner.","",
       f"Environment: Python {platform.python_version()}, PyTorch {torch.__version__}; "
       f"CUDA device {torch.cuda.get_device_name() if torch.cuda.is_available() else 'unavailable'}. "
       "Main experiments ran on CPU with one Torch thread. Module-level BF16 and a normal Trainer/GradScaler run also passed on CUDA.","",
       "## Reproduction and evidence", "",
       "Run these commands from the repository root. Each script generates isolated fixtures/checkpoints under `/tmp`; "
       "the JSON files and captured logs beside this report are the authoritative recorded results.","", "```bash"]
for command in commands:
    lines.append(" ".join(command["command"]))
lines.extend(["```","", "| Script | Groups passed | Elapsed | Evidence |",
              "| --- | ---: | ---: | --- |"])
for (name,count,report),command in zip((("models",12,"model_checks.json"),("runtime",9,"runtime_checks.json"),
                                      ("transfer",3,"transfer_checks.json"),("suites",5,"suite_checks.json")),commands):
    lines.append(f"| {name} | {count} | {command['seconds']:.2f}s | [{report}]({report}), [{name}.log]({name}.log) |")
lines.extend(["", "The first execution of the new model assertions was already green because concurrent implementation fixes had landed. "
              "The same first eight groups were then replayed against an isolated archive of baseline "
              "`22dc54c1a8ac573c198e2517c46ba8915d33c9c7`; all eight failed there. "
              "[initial_models.json](initial_models.json) records that regression evidence. This is archived-baseline verification, not a claim of strict red-first development.","",
              "## Architecture and numerical contracts", "",
              "Shapes below are measured by assertions with raw sensor width 3 and latent width 6. One checkpoint accepts different legal temporal lengths; "
              "separate runs also passed with sensor widths 1 and 5. Real samples exercised 25 and 38 sensors.","",
              "| Experiment | Raw input | Current/history latent | Target/prediction latent | Output score |",
              "| --- | --- | --- | --- | --- |",
              "| AE / VAE | `[B,3,48]` | `[B,6,48]` | decoder `[B,3,48]` | `[B,48]` |",
              "| Masked frequency LeWM | `[B,3,48]` | corrupted `[B,6,48]` | clean same-window `[B,6,48]` | `[B,48]` |",
              "| Guided LeWM, stride 1 | `[B,3,48]` | history `[B,6,32]` | observed future / prediction `[B,6,16]` | `[B,16]`, raw interval `[32,48)` |",
              "| Guided LeWM, source stride 2 | `[B,3,48]` | history `[B,6,16]` | future / prediction `[B,6,8]` | scalar latent scores interpolated to 16 raw positions |",
              "| LeWM backbone + identity AE | `[B,3,48]` | source `[B,6,24]`, identity unchanged | decoder `[B,3,48]` | `[B,48]` |","",
              "Future-derived Q was observed to change only when future observations changed; paired historical K/V retained the historical token count. "
              "History/future lengths 32/16, 42/21, and 60/30 passed with one model. The guided predictor contains a length-preserving ConvTranspose1d; "
              "neither LeWM default has a raw reconstruction head. The frequency test checked actual packed complex coordinates and actual unpacked iSTFT output "
              "for multiple batch sizes and odd/even frame counts, plus unchanged STFT/iSTFT roundtrips.","",
              "KL/L1/MSE/L2 loss and score equations were checked against literal independent values. L2 is channel RMS, then mean over batch/time for loss; "
              "zero-error L2 gradients are finite and zero. Default latent KL and default L1 score remain independent: a shared channel offset gives zero KL "
              "but positive L1. Exact weighted summation, disabled operands, shape rejection, target-branch gradients, auxiliary BCE and unrestricted negative latents passed. "
              "Raw reconstruction was separately enabled and its decoder gradients verified.","",
              "## Short training, validation and test results", "",
              "All four runs used two epochs, batch size 4, a 48-step window, stride 8, seed 17, and synthetic periodic/AR telemetry. "
              "Optimization used 123 clean rows; held-out validation used the remaining 53 rows. Test used a separate 112-row anomalous realization. "
              "All summaries and components are finite; actual checkpoints, resolved configs, TensorBoard files, architecture text, score arrays and per-aggregation exports "
              "are in the run directories linked by the JSON evidence.","",
              "| Run | Epoch-2 train loss | Epoch-2 validation loss | Best validation loss | Timestep calibrated F1 | Timestep oracle F1 | Covered test rows |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
for name,run in runs.items():
    history=run["history"][-1]["losses"];score=run["score_report"]
    metrics=score["evaluation"]["timeseries"]
    coverage=score["coverage"]
    lines.append(f"| {name} | {history['train']['loss']:.6f} | {history['validation']['loss']:.6f} | {run['best']:.6f} | "
                 f"{metrics['calibrated']['f1']:.6f} | {metrics['oracle']['f1']:.6f} | {coverage['scored_timesteps']}/{coverage['total_timesteps']} |")
lines.extend(["", "The table uses validation-selected weights for scoring; epoch-2 losses need not describe the best epoch. "
              "Window and non-overlap-window protocols are recorded separately in `runtime_checks.json`, with both calibrated and oracle results. "
              "The guided run correctly excludes its initial 32 context rows; raw reconstruction runs cover all 112 rows under overlapping evaluation. "
              "Non-overlap evaluation intentionally leaves incomplete final regions uncovered.","", "![Loss components](component_curves.png)","",
              "The two LeWM objectives were independently optimized on a fixed batch for 45 updates with fixed regularizer sketches:","",
              "| Model | Initial total | Final total | Guide-shuffle mean prediction change | History-shuffle mean prediction change |",
              "| --- | ---: | ---: | ---: | ---: |"])
for name,item in detail(models,"fixed batch optimization conditioning").items():
    lines.append(f"| {name} | {item['initial_loss']:.6f} | {item['final_loss']:.6f} | "
                 f"{item['guide_shuffle_mean_difference']:.6f} | {item['history_shuffle_mean_difference']:.6f} |")
lines.extend(["", "Guide/history ablations demonstrate sensitivity of the trained computation, not a causal interpretation of the conditioning signal.","",
              "## Transfer and reproducibility", "",
              "Five trained-source routes passed: AE→guided backbone, masked LeWM→guided backbone, LeWM→AE backbone plus identity, "
              "LeWM→trainable AE encoder (default), and LeWM→explicitly frozen AE encoder. All first transferred features matched the source exactly. "
              "Frozen parameters and BatchNorm buffers stayed unchanged while destination gradients flowed; trainable encoder parameters changed. "
              "Every route also ran the normal two-epoch training/validation/test entry. A guided destination then resumed through epoch 3 and scored "
              "while its original source checkpoint was unavailable.","",
              "Two actual baseline-source models were one-step trained in an archived baseline subprocess. Their saved constructor configs and legacy weights "
              "transferred without changing GELU/identity/stride behavior: maximum source-feature error was exactly zero for AE and LeWM. "
              "Those sources carried full-training normalization statistics, while the new destination retained optimization-row statistics; "
              "`normalization_changed: true` and both preprocessing descriptions were recorded. Equality is defined on the same normalized tensor input. "
              "Missing legacy schema is explicitly qualified by matched saved collection/series/width plus the destination reader order. "
              "Incompatible supplied schema/order, stride, activation and same-shaped forecast task semantics were rejected.","",
              "An uninterrupted three-epoch run and a one-epoch interruption followed by resume had bitwise-identical final weights and final component losses "
              "for both VAE and masked LeWM. The latter included an explicitly seeded SIGReg local generator. Repeated fixed-corruption validation and preflight "
              "left training RNG, model parameters and normalization buffers unchanged. A 4+4+1 batch fixture distinguished correct sample-weighted summaries "
              "from incorrect equal batch averaging.","",
              "## Dataset jobs and checkpoint protocol", "",
              "The full PSM/SMD/TSB-U/TSB-M selection matrix passed, including PSM clearing inherited filenames, SMD file/all/all-one, and explicit TSB files "
              "or ordered file_name CSV lists with optional train_end and unrelated metadata. Missing/blank/duplicate selections failed preflight. "
              "TSB evaluation covered the 112-row suffix after the declared 160-row prefix; Label was excluded from features.","",
              "Pooled SMD trained one boundary-safe model, with per-machine normalization, thresholds, evaluation artifacts and macro metrics. "
              "Its oracle-F1 diagnostic checkpoint recorded a per-machine threshold scope instead of inventing a macro threshold. "
              "Separate source mappings selected each machine's best-validation checkpoint, falling back to best-training for a no-validation source.","",
              "Two concurrent worker processes had isolated output directories and fresh model/optimizer/RNG state. Removing one selected training file after "
              "preflight produced one recorded runtime failure while the other job completed; restoring it retried the failed job and skipped the completed one "
              "without touching its checkpoint. The same seeded job produced bitwise-identical weights sequentially and in the process pool. "
              "CLI training followed by --score executed scoring, an explicit distinct checkpoint changed scores, and missing output metrics were regenerated "
              "instead of being silently accepted as complete.","",
              "Primary best-validation (or best-training when validation is unavailable), rolling last, and ordinary-oracle-F1 checkpoints for all three "
              "aggregation strategies were separately verified. Clean thresholds were independently recomputed from the full clean training score distribution "
              "and its configured quantile. Exported cover counts were independently reconstructed from target intervals. Reloaded predictions/thresholds/scores "
              "were identical. A source with only one legal training span used every training row, no fabricated validation, and best-training selection; "
              "exactly two legal windows retained a disjoint train/validation pair even with requested validation fraction 0.1.","",
              "## Installed-data checks and limits", ""])
for collection,item in detail(runtime,"real data samples").items():
    if item["available"]:
        lines.append(f"- {collection.upper()}: actual reader, first/last training windows, model/loss and test score sample passed with "
                     f"{item['feature_count']} features; source `{item['root']}`, file `{item['fname']}`.")
    else:lines.append(f"- {collection}: real collection not installed; generated format/list fixtures passed.")
lines.extend(["", "These are plumbing and numerical checks, not benchmark-quality measurements. The tiny synthetic test has dense/overlapping anomaly segments "
              "and some window-level labels lack both classes; high oracle F1 must not be interpreted as detector quality. "
              "The format fixtures deliberately shift the clean validation distribution to expose scaler leakage, so their calibrated F1 can be zero. "
              "No full real-data training sweep, long-run stability evaluation, GPU out-of-memory failure injection, cross-device bitwise reproducibility, "
              "or physical/causal action validation was performed. All generated experiment outputs stayed under `/tmp`; the existing datasets symlink "
              "and pre-existing results directory were preserved.",""])
(HERE/"REPORT.md").write_text("\n".join(lines))
print(HERE/"REPORT.md")
