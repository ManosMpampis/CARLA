"""Bound near-oracle reachability using saved clean-range boundary diagnostics.

Only one clean-range endpoint is known for out-of-range oracle thresholds.
Relax the unknown other endpoint: the computed maximum is an UPPER BOUND on
what any clean empirical quantile can achieve. Failure of this relaxed bound
proves failure to reach the target; success does not prove reachability.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]


def main():
    rows=[]
    for path in (ROOT/'results/smd').rglob('metrics.json'):
        if '/evaluations/' in str(path):continue
        d=json.loads(path.read_text());w=d['evaluation']['window'];o=w['oracle']
        b=o['calibration_threshold'];t=o['threshold']
        with np.load(path.parent/'scores.npz') as z:s,y=z['window_scores'],z['window_labels']
        p,r,thresholds=precision_recall_curve(y,s)
        cutoffs=np.nextafter(thresholds,-np.inf)
        fs=np.divide(2*p[:-1]*r[:-1],p[:-1]+r[:-1],out=np.zeros_like(thresholds),where=(p[:-1]+r[:-1])>0)
        if o['calibration_threshold_in_range']:
            bound=o['f1_no_pa'];direction='in_range'
        else:
            eligible=(cutoffs<=b) if t>b else (cutoffs>=b)
            pred=s>b;tp=np.sum(pred&(y==1));fp=np.sum(pred&(y==0));fn=np.sum(~pred&(y==1))
            at_boundary=2*tp/max(1,2*tp+fp+fn)
            bound=max(float(fs[eligible].max()) if eligible.any() else 0,float(at_boundary))
            direction='above_clean_max' if t>b else 'below_clean_min'
        rows.append(dict(experiment=d['experiment_key'],machine=d['fname'],
                         oracle_f1=o['f1_no_pa'],cal_f1=w['calibrated']['f1_no_pa'],
                         direction=direction,clean_boundary=b,oracle_threshold=t,
                         relaxed_quantile_f1_upper_bound=bound,
                         near_oracle_impossible=bool(bound<o['f1_no_pa']-.02-1e-12),
                         boundary_oracle_ratio=t/b if b else None))
    f=pd.DataFrame(rows);f.to_csv(OUT/'threshold_reachability.csv',index=False)
    full=f[f.groupby('experiment').experiment.transform('count')==28]
    a=full.groupby('experiment').agg(impossible=('near_oracle_impossible','sum'),
                                   relaxed_best_f1=('relaxed_quantile_f1_upper_bound','mean'))
    a.to_csv(OUT/'threshold_reachability_summary.csv')
    print(a.to_string())


if __name__=='__main__':main()
