"""Compare four saved SMD cross-attention families at matched score geometry.

Read-only, no inference. Run from repo root with ./venv/bin/python.
Timeline comparisons use NPZ score_batch_size=256 archives; the separately
exported decision-trace JSON generally uses batch size 1 and can differ slightly.
"""
from pathlib import Path
import csv
import json

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FAMILIES = {
    "features": "frequency_aux/cross_attention/observed_features",
    "query": "frequency_aux/cross_attention/observed_query",
    "input": "frequency_aux/cross_attention/observed_input",
    "no_film_features": "frequency_aux_no_film/cross_attention/observed_features",
}


def events(labels):
    padded = np.r_[False, labels, False]
    edges = np.diff(padded.astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def decisions(scores, labels, threshold):
    pred = scores > threshold
    ev = events(labels)
    return {
        "f1": float(f1_score(labels, pred)),
        "fpr": float(pred[~labels].mean()),
        "recall": float(pred[labels].mean()),
        "predicted_fraction": float(pred.mean()),
        "event_hit_rate": float(np.mean([pred[a:b].any() for a, b in ev])),
        "events_hit": int(sum(pred[a:b].any() for a, b in ev)),
        "event_count": len(ev),
    }


def main():
    rows = []
    reference = {}
    for family, relative in FAMILIES.items():
        paths = sorted((ROOT / "results/smd/lewm_encoder" / relative).glob("machine-*/*/scores.npz"))
        assert len(paths) == 28
        for p in paths:
            machine = p.parent.parent.name
            m = json.loads((p.parent / "metrics.json").read_text())
            settings = m["evaluation_settings"]
            assert settings["input_window"] == 1024 and settings["stride"] == 10
            assert settings["check_mask"] == [512, 1024]
            with np.load(p, allow_pickle=False) as z:
                wscore = z["window_scores"]
                wlabels = z["window_labels"].astype(bool)
                start = z["start_idxs"]
                end = z["end_idxs"]
                covered = z["cover_counts"] > 0
                labels = z["timestep_labels"].astype(bool)
                tscores = z["timeseries_scores"]
                wpred = z["window_predictions"].astype(bool)
                tpred = z["timeseries_predictions"].astype(bool)
            signature = (wlabels, start, end, covered, labels)
            if family == "features":
                reference[machine] = signature
            else:
                assert all(np.array_equal(a, b) for a, b in zip(reference[machine], signature)), (family, machine)
            assert np.isfinite(wscore).all() and np.isfinite(tscores[covered]).all()
            wm = m["evaluation"]["window"]
            tm = m["evaluation"]["timeseries"]
            assert np.array_equal(wscore > wm["calibrated"]["threshold"], wpred)
            assert np.array_equal((tscores > tm["calibrated"]["threshold"])[covered], tpred[covered])
            assert np.isclose(f1_score(wlabels, wpred), wm["calibrated"]["f1_no_pa"])
            cy = labels[covered]
            cs = tscores[covered]
            row = {
                "family": family, "machine": machine,
                "scores_path": str(p.relative_to(ROOT)),
                "metrics_path": str((p.parent / "metrics.json").relative_to(ROOT)),
                "windows": len(wlabels), "covered_timesteps": int(covered.sum()),
                "timeline_anomaly_fraction": float(cy.mean()),
                "window_anomaly_fraction": float(wlabels.mean()),
                "window_calibrated_f1": float(wm["calibrated"]["f1_no_pa"]),
                "window_oracle_f1": float(wm["oracle"]["f1_no_pa"]),
                "window_vus_pr": float(wm["vus_pr"]),
                "window_vus_roc": float(wm["vus_roc"]),
                "window_ordinary_ap": float(average_precision_score(wlabels, wscore)),
                "window_ordinary_roc": float(roc_auc_score(wlabels, wscore)),
                "timeline_ordinary_ap": float(average_precision_score(cy, cs)),
                "timeline_ordinary_roc": float(roc_auc_score(cy, cs)),
            }
            for source in ["calibrated", "oracle"]:
                d = decisions(cs, cy, float(tm[source]["threshold"]))
                assert np.isclose(d["f1"], tm[source]["f1_no_pa"]), (family, machine, source)
                row.update({f"timeline_{source}_{k}": v for k, v in d.items()})
            rows.append(row)
    fields = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
    summaries = {f: {k: float(np.mean([r[k] for r in rows if r["family"] == f])) for k in fields}
                 for f in FAMILIES}
    by = {(r["family"], r["machine"]): r for r in rows}
    comparisons = {}
    key_fields = ["window_calibrated_f1", "window_oracle_f1", "window_vus_pr", "window_vus_roc",
                  "timeline_ordinary_ap", "timeline_ordinary_roc", "timeline_calibrated_f1",
                  "timeline_calibrated_fpr", "timeline_calibrated_event_hit_rate"]
    for other in ["query", "input", "no_film_features"]:
        pairs = [(r, by[(other, r["machine"])]) for r in rows if r["family"] == "features"]
        comp = {}
        for field in key_fields:
            diff = np.array([a[field] - b[field] for a, b in pairs])
            order = np.argsort(diff)
            comp[field] = {"features_minus_other_macro_mean": float(diff.mean()),
                           "features_numerically_higher_count": int((diff > 1e-9).sum()),
                           "features_numerically_lower_count": int((diff < -1e-9).sum()),
                           "tie_count": int((abs(diff) <= 1e-9).sum()),
                           "largest_features_advantages": [{"machine": pairs[i][0]["machine"],
                               "features": pairs[i][0][field], "other": pairs[i][1][field], "difference": float(diff[i])}
                               for i in order[-3:][::-1]],
                           "largest_features_disadvantages": [{"machine": pairs[i][0]["machine"],
                               "features": pairs[i][0][field], "other": pairs[i][1][field], "difference": float(diff[i])}
                               for i in order[:3]]}
        comparisons[other] = comp
    payload = {"method": "All28 matched base-machine NPZ archives per family; equal-machine macro means; actual covered timeline only; ordinary AP/ROC are threshold-free descriptive scores; oracle threshold maximizes timestep F1 not event hit. Event hit = any positive prediction during one contiguous annotated event, without point adjustment; high event hit alone can be attained by excessive false alarms. Timeline comparison uses NPZ batch256, not decision-trace JSON batch1.",
               "families": FAMILIES, "summary": summaries, "comparisons": comparisons, "machines": rows}
    (OUT / "matched_heads.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (OUT / "matched_heads.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"summary": summaries, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
