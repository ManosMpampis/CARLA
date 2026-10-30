"""Read-only SMD window experiment audit. Run with ./venv/bin/python.

Outputs stay beside this script; results/ and datasets/ are never modified.
Each machine gets equal weight. Test-derived diagnostics are retrospective.
"""
from __future__ import annotations

import json
import math
import os
import hashlib
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/carla-smd-report-mpl")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SMD = ROOT / "results/smd"


def finite(x):
    return float(x) if x is not None and math.isfinite(float(x)) else np.nan


def corr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return np.nan
    return float(spearmanr(a[ok], b[ok]).statistic)


def flatten(d):
    w = d["evaluation"]["window"]
    c, o = w["calibrated"], w["oracle"]
    s = d.get("evaluation_settings", {})
    return dict(cal_f1=c["f1_no_pa"], oracle_f1=o["f1_no_pa"],
                event_f1=c.get("event_f1_no_pa"), vus_pr=w["vus_pr"], vus_roc=w["vus_roc"],
                threshold=c["threshold"], oracle_threshold=o["threshold"],
                oracle_q=o.get("calibration_quantile"),
                oracle_in_range=o.get("calibration_threshold_in_range"),
                oracle_clean_alarm=o.get("clean_anomaly_fraction"),
                input_window=s.get("input_window"), stride=s.get("stride"),
                output_window=d.get("coverage", {}).get("output_window"),
                calibration_q=c.get("calibration_quantile"),
                eval_window_size=s.get("eval_window_size"))


def f1_at(scores, labels, threshold):
    pred = scores > threshold
    tp = int(np.sum(pred & (labels == 1)))
    fp = int(np.sum(pred & (labels == 0)))
    fn = int(np.sum(~pred & (labels == 1)))
    return 2 * tp / max(1, 2 * tp + fp + fn)


def score_diagnostics(path, row):
    with np.load(path) as z:
        scores, labels = z["window_scores"], z["window_labels"]
        starts, ends = z["start_idxs"], z["end_idxs"]
    p, r, t = precision_recall_curve(labels, scores)
    fs = np.divide(2*p[:-1]*r[:-1], p[:-1]+r[:-1],
                   out=np.zeros_like(t), where=(p[:-1]+r[:-1]) > 0)
    best = float(fs.max())
    near = fs >= best - 0.02
    bestidx = int(fs.argmax())
    lo, hi = bestidx, bestidx
    while lo > 0 and near[lo-1]:
        lo -= 1
    while hi < len(t)-1 and near[hi+1]:
        hi += 1
    threshold = float(row["threshold"])
    pred = scores > threshold
    tp = int(np.sum(pred & (labels == 1)))
    fp = int(np.sum(pred & (labels == 0)))
    fn = int(np.sum(~pred & (labels == 1)))
    tn = int(np.sum(~pred & (labels == 0)))
    result = dict(n_windows=len(scores), prevalence=float(labels.mean()),
                  always_positive_f1=2*float(labels.mean())/(1+float(labels.mean())),
                  tp=tp, fp=fp, fn=fn, tn=tn,
                  precision=tp/max(1,tp+fp), recall=tp/max(1,tp+fn),
                  test_normal_fpr=fp/max(1,fp+tn),
                  reconstructed_cal_f1=f1_at(scores,labels,threshold),
                  reconstructed_oracle_f1=best,
                  near_oracle_threshold_lo=float(t[lo]),
                  near_oracle_threshold_hi=float(t[hi]),
                  near_oracle_width_relative=float((t[hi]-t[lo])/max(abs(t[bestidx]),1e-12)),
                  vanilla_ap=average_precision_score(labels,scores),
                  vanilla_roc=roc_auc_score(labels,scores),
                  score_min=float(scores.min()), score_max=float(scores.max()),
                  score_p99=float(np.quantile(scores,.99)),
                  actual_output_window=int(np.median(ends-starts)))
    for factor in (.8,.9,.95,1.05,1.1,1.2):
        result[f"cal_f1_factor_{factor}"] = f1_at(scores,labels,threshold*factor)
    result["cal_f1_worst_10pct"] = min(result["cal_f1_factor_0.9"],
                                      result["cal_f1_factor_1.1"],result["reconstructed_cal_f1"])
    return result


def read_run(path):
    d = json.loads(path.read_text())
    config = yaml.safe_load((path.parent / "resolved_config.yml").read_text())
    row = dict(experiment=d["experiment_key"], machine=d["fname"],
               runner=d.get("runner"), path=str(path.relative_to(ROOT)),
               seed=config.get("seed"), configured_epochs=config.get("epochs"),
               selection=json.dumps(d.get("selection", {})), **flatten(d))
    row.update(score_diagnostics(path.parent / "scores.npz", row))
    row["gap"] = row["oracle_f1"] - row["cal_f1"]
    row["near_oracle_selected"] = row["gap"] <= .02
    row["above_trivial_f1"] = row["cal_f1"] - row["always_positive_f1"]
    histories, duplicates = {}, 0
    hpath = path.parent / "test_evaluations.jsonl"
    if hpath.exists():
        for line in hpath.read_text().splitlines():
            entry = json.loads(line)
            if "window" not in entry.get("evaluation", {}):
                continue
            epoch = entry["epoch"]
            duplicates += epoch in histories
            histories[epoch] = dict(experiment=row["experiment"], machine=row["machine"],
                                    epoch=epoch, **flatten(entry))
    hist = sorted(histories.values(), key=lambda e:e["epoch"])
    scalar_rows, tags, duplicate_steps = [], [], 0
    tb = path.parent / "tensorboard"
    if tb.exists():
        ea = EventAccumulator(str(tb),size_guidance={"scalars":0,"images":1,"histograms":1,"tensors":1})
        ea.Reload()
        tags = ea.Tags()["scalars"]
        wanted = [tag for tag in tags if not tag.startswith("test/") and
                  (tag.startswith(("train/","validation/")) or
                   tag in ("Train Loss/latent_var",))]
        for tag in wanted:
            vals = {}
            for event in ea.Scalars(tag):
                duplicate_steps += event.step in vals
                vals[event.step] = event.value
            for step, value in vals.items():
                scalar_rows.append(dict(experiment=row["experiment"],machine=row["machine"],
                                        tag=tag,epoch=step,value=value))
        valtag = "validation/loss" if "validation/loss" in tags else "validation/pred_loss"
        val = {s["epoch"]:s["value"] for s in scalar_rows if s["tag"] == valtag}
        row["val_tag"] = valtag
        row["n_validation"] = len(val)
        if val:
            row["min_val_epoch_tb"] = min(val,key=val.get)
            row["last_val_epoch_tb"] = max(val)
            v = np.array([val[e] for e in sorted(val)])
            tail = v[len(v)*2//3:]
            row["val_tail_cv"] = float(np.std(tail)/max(abs(np.mean(tail)),1e-12))
            row["val_tail_rel_jump"] = float(np.median(2*np.abs(np.diff(tail))/np.maximum(np.abs(tail[:-1])+np.abs(tail[1:]),1e-12))) if len(tail)>1 else np.nan
            row["val_final_over_min"] = v[-1]/max(abs(v.min()),1e-12)
            for e in hist:
                e["val_loss"] = val.get(e["epoch"],np.nan)
            row["val_cal_spearman"] = corr([e.get("val_loss",np.nan) for e in hist],[e["cal_f1"] for e in hist])
            row["val_oracle_spearman"] = corr([e.get("val_loss",np.nan) for e in hist],[e["oracle_f1"] for e in hist])
    row.update(history_points=len(hist), history_duplicate_epochs=duplicates,
               tensorboard_duplicate_steps=duplicate_steps)
    if hist:
        h=pd.DataFrame(hist)
        tail=h[h.epoch>=h.epoch.max()*2/3]
        row.update(first_eval_epoch=int(h.epoch.min()),last_eval_epoch=int(h.epoch.max()),
                   peak_cal_f1=float(h.cal_f1.max()),peak_oracle_f1=float(h.oracle_f1.max()),
                   peak_cal_epoch=int(h.loc[h.cal_f1.idxmax(),"epoch"]),
                   peak_oracle_epoch=int(h.loc[h.oracle_f1.idxmax(),"epoch"]),
                   cal_epoch_near_peak_fraction=float((h.cal_f1>=h.cal_f1.max()-.02).mean()),
                   oracle_epoch_near_peak_fraction=float((h.oracle_f1>=h.oracle_f1.max()-.02).mean()),
                   epoch_near_oracle_fraction=float(((h.oracle_f1-h.cal_f1)<=.02).mean()),
                   tail_cal_std=float(tail.cal_f1.std(ddof=0)),
                   tail_oracle_std=float(tail.oracle_f1.std(ddof=0)),
                   tail_vus_pr_std=float(tail.vus_pr.std(ddof=0)),
                   tail_vus_roc_std=float(tail.vus_roc.std(ddof=0)),
                   tail_threshold_cv=float(tail.threshold.std(ddof=0)/max(abs(tail.threshold.mean()),1e-12)),
                   tail_oracle_threshold_cv=float(tail.oracle_threshold.std(ddof=0)/max(abs(tail.oracle_threshold.mean()),1e-12)),
                   tail_oracle_q_std=float(tail.oracle_q.std(ddof=0)),
                   tail_gap_mean=float((tail.oracle_f1-tail.cal_f1).mean()),
                   selected_vs_peak_cal=float(h.cal_f1.max()-row["cal_f1"]),
                   selected_vs_peak_oracle=float(h.oracle_f1.max()-row["oracle_f1"]))
        for field in ("threshold","oracle_threshold"):
            v=tail[field].to_numpy()
            row[f"tail_{field}_rel_jump"] = float(np.median(2*np.abs(np.diff(v))/np.maximum(np.abs(v[:-1])+np.abs(v[1:]),1e-12))) if len(v)>1 else np.nan
    return row,hist,scalar_rows


def main():
    paths=sorted(p for p in SMD.rglob("metrics.json") if "/evaluations/" not in str(p))
    rows,histories,scalars=[],[],[]
    # Initialize TensorBoard's lazy compatibility imports on the main thread.
    # Concurrent first-use imports can otherwise wait on each other's locks.
    read_run(paths[0])
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,(row,h,s) in enumerate(pool.map(read_run,paths),1):
            rows.append(row); histories.extend(h);scalars.extend(s)
            if i%100==0: print(f"Extracted {i}/{len(paths)} runs",flush=True)
    base=pd.DataFrame(rows)
    base.to_csv(OUT/"selected_runs.csv",index=False)
    pd.DataFrame(histories).to_csv(OUT/"epoch_metrics.csv",index=False)
    pd.DataFrame(scalars).to_csv(OUT/"training_scalars.csv.gz",index=False,compression="gzip")
    counts=base.groupby("experiment").machine.nunique()
    full=base[base.experiment.isin(counts[counts==28].index)]
    agg=full.groupby("experiment").mean(numeric_only=True)
    agg["machines"]=full.groupby("experiment").machine.nunique()
    for key in ("cal_f1","oracle_f1","vus_pr","vus_roc","gap","above_trivial_f1"):
        agg[key+"_std"]=full.groupby("experiment")[key].std()
        agg[key+"_min"]=full.groupby("experiment")[key].min()
        agg[key+"_median"]=full.groupby("experiment")[key].median()
    agg.sort_values("cal_f1",ascending=False).to_csv(OUT/"family_summary.csv")
    sweep=[]
    sp=SMD/"evaluations/inference_window_sweep/summary.json"
    if sp.exists():
        for d in json.loads(sp.read_text()):
            if d.get("status") != "completed": continue
            row=dict(experiment=d["experiment_key"],machine=d["fname"],case=d["case"],
                     checkpoint=d.get("checkpoint"),path=d.get("metrics_path"),**flatten(d))
            m=json.loads(Path(d["metrics_path"]).read_text())
            row["output_window"]=m.get("coverage",{}).get("output_window")
            row["gap"]=row["oracle_f1"]-row["cal_f1"]
            sweep.append(row)
        sw=pd.DataFrame(sweep)
        sw.to_csv(OUT/"inference_cases.csv",index=False)
        sa=sw.groupby(["experiment","case"]).mean(numeric_only=True)
        sa["machines"]=sw.groupby(["experiment","case"]).machine.nunique()
        sa.sort_values("cal_f1",ascending=False).to_csv(OUT/"inference_case_summary.csv")
    checks=dict(base_runs=len(base),complete_families=len(agg),complete_runs=len(full),
                partial_families=counts[counts<28].to_dict(),history_rows=len(histories),
                scalar_rows=len(scalars),sweep_rows=len(sweep),seeds=sorted(base.seed.unique().tolist()),
                max_cal_f1_reconstruction_error=float(abs(base.cal_f1-base.reconstructed_cal_f1).max()),
                max_oracle_f1_reconstruction_error=float(abs(base.oracle_f1-base.reconstructed_oracle_f1).max()),
                duplicate_epoch_runs=int((base.history_duplicate_epochs>0).sum()),
                duplicate_tb_runs=int((base.tensorboard_duplicate_steps>0).sum()))
    (OUT/"extraction_checks.json").write_text(json.dumps(checks,indent=2))
    assert checks["max_cal_f1_reconstruction_error"] < 1e-12
    assert checks["max_oracle_f1_reconstruction_error"] < 1e-12
    assert (base.output_window == base.actual_output_window).all()
    best=full.loc[full.groupby(["machine","output_window"]).cal_f1.idxmax()]
    best.sort_values(["output_window","machine"]).to_csv(OUT/"best_per_machine.csv",index=False)
    sizes=[]
    for path in (ROOT/"datasets/SMD/train").glob("machine-*.txt"):
        with path.open() as stream: n=sum(1 for _ in stream)
        tail=n-int(.9*n)
        def count(window):
            quotient,remainder=divmod(tail-window,10)
            return quotient+1+(remainder>0)
        sizes.append(dict(machine=path.name,train_timesteps=n,calibration_timesteps=tail,
                          cal_windows_1024=count(1024),cal_windows_512=count(512)))
    pd.DataFrame(sizes).to_csv(OUT/"calibration_sizes.csv",index=False)
    source_files=["utils/experiment_suite.py","utils/reconstruction_baselines.py",
                  "utils/trainer.py","data/jepa_dataset.py","models/lewm.py",
                  "losses/lewm.py","models/lewm_cross_attention.py","metrics/vus/metrics.py",
                  "metrics/vus/utils/metrics_torch.py"]
    provenance=dict(generated_at=datetime.now(ZoneInfo("Europe/Athens")).isoformat(),
        git_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        source_hashes={x:hashlib.sha256((ROOT/x).read_bytes()).hexdigest() for x in source_files},
        metrics_inputs=[dict(path=x,sha256=hashlib.sha256((ROOT/x).read_bytes()).hexdigest(),
                             mtime_ns=(ROOT/x).stat().st_mtime_ns) for x in base.path],
        scope="SMD evaluation.window only; no retraining or modifications of experiment artifacts")
    (OUT/"provenance.json").write_text(json.dumps(provenance,indent=2))
    print(json.dumps(checks,indent=2))
    print(agg.sort_values("cal_f1",ascending=False)[["cal_f1","oracle_f1","vus_pr","vus_roc","tail_cal_std","tail_threshold_cv"]].to_string())


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
