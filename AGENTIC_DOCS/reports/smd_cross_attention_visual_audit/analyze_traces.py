"""Render every saved SMD frequency_aux observed_features replay, read-only.

Uses JSON final timeline states; recovers raw window means from cumulative
means/counts; compares both with batch-scored NPZ. Test-oracle thresholds and
test-normal false-positive budgets are retrospective diagnostics only.
"""
import gc
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/tmp/carla-cross-visual-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter,MaxNLocator
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve,roc_curve,roc_auc_score,average_precision_score

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
BASE=ROOT/'results/smd/lewm_encoder/frequency_aux/cross_attention/observed_features'
plt.rcParams.update({'figure.dpi':125,'savefig.dpi':160,'font.size':10,
    'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.16})


def intervals(labels):
    edges=np.diff(np.r_[0,np.asarray(labels,dtype=int),0])
    return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))


def stats(scores,labels,threshold):
    pred=scores>threshold;labels=labels.astype(bool)
    tp=int(np.sum(pred&labels));fp=int(np.sum(pred&~labels))
    fn=int(np.sum(~pred&labels));tn=int(np.sum(~pred&~labels))
    return dict(f1=2*tp/max(1,2*tp+fp+fn),precision=tp/max(1,tp+fp),
                recall=tp/max(1,tp+fn),fpr=fp/max(1,fp+tn),tp=tp,fp=fp,fn=fn,tn=tn)


def curve(scores,labels):
    p,r,t=precision_recall_curve(labels,scores)
    fs=np.divide(2*p[:-1]*r[:-1],p[:-1]+r[:-1],out=np.zeros_like(t),where=p[:-1]+r[:-1]>0)
    idx=int(np.argmax(fs));threshold=float(np.nextafter(t[idx],-np.inf))
    near=fs>=fs[idx]-.02;lo=hi=idx
    while lo>0 and near[lo-1]:lo-=1
    while hi<len(t)-1 and near[hi+1]:hi+=1
    return threshold,stats(scores,labels,threshold),dict(ap=average_precision_score(labels,scores),
        roc=roc_auc_score(labels,scores),oracle_near_lo=float(t[lo]),oracle_near_hi=float(t[hi]),
        oracle_near_relative_width=float((t[hi]-t[lo])/max(abs(threshold),1e-12)))


def flatten(prefix,values):return {prefix+'_'+key:value for key,value in values.items()}


def draw_score(ax,x,score,threshold,oracle,truth,title,window=False):
    ratio=score/threshold
    for start,end in truth:ax.axvspan(start,end,color='#9966cc',alpha=.19,lw=0)
    ax.plot(x,ratio,lw=.95,color='#245991')
    ax.axhline(1,color='#128375',lw=1.2,ls='--',label='Calibrated threshold')
    ax.axhline(oracle/threshold,color='#ba6134',lw=1.2,ls=':',label='Retrospective oracle')
    ax.set_yscale('function',functions=(lambda y:np.sign(y)*np.log1p(abs(y)),
                                      lambda y:np.sign(y)*np.expm1(abs(y))))
    maximum=max(float(np.nanmax(ratio)),1,float(oracle/threshold))
    upper=float(np.expm1(np.log1p(maximum)*1.05))
    ax.set_ylim(0,upper)
    ticks=np.expm1(np.linspace(0,np.log1p(upper),5))
    nearest=int(np.argmin(abs(np.log1p(ticks)-np.log(2))))
    if nearest>0 and abs(np.log1p(ticks[nearest])-np.log(2))<.05*np.log1p(upper):
        ticks[nearest]=1
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:.2g}'))
    ax.set_ylabel('Score / calibrated threshold\nHTML log1p scale')
    ax.set_title(title,loc='left',fontsize=10)
    ax.set_xlim(0,max(x)+1)
    ax.ticklabel_format(axis='x',style='plain',useOffset=False)


def run(path):
    machine=path.parent.parent.name
    report=json.loads((path.parent/'metrics.json').read_text())
    trace=json.loads(path.read_text())
    with np.load(path.parent/'scores.npz') as archive:
        z={name:archive[name].copy() for name in ['timeseries_scores','window_scores','window_labels',
             'timestep_labels','start_idxs','end_idxs','cover_counts']}
    n=trace['series_length'];scores=np.full(n,np.nan);counts=np.zeros(n,int);sums=np.zeros(n)
    window=[]
    for update in trace['updates']:
        idx=np.asarray(update['indices']);values=np.asarray(update['scores']);cnt=np.asarray(update['counts'])
        new_sums=values*cnt;window.append(float(np.mean(new_sums-sums[idx])))
        sums[idx]=new_sums;scores[idx]=values;counts[idx]=cnt
    covered=counts>0;window=np.asarray(window);y=z['timestep_labels'].astype(bool)
    truth=intervals(y);wy=z['window_labels'].astype(bool)
    assert not trace['scope']['truncated']
    assert np.array_equal(counts,z['cover_counts'])
    assert np.array_equal(np.asarray(truth),np.asarray(trace['ground_truth_intervals']))
    assert len(window)==len(z['window_scores'])
    c=report['evaluation']['window']['calibrated']['threshold'];tc=trace['threshold']
    assert tc==report['evaluation']['timeseries']['calibrated']['threshold']
    oracle,om,extra=curve(scores[covered],y[covered]);cal=stats(scores[covered],y[covered],tc)
    wo,wom,wextra=curve(z['window_scores'],wy)
    wc=stats(z['window_scores'],wy,c)
    replay_wo,replay_wom,replay_extra=curve(window,wy)
    replay_wc=stats(window,wy,c)
    official=stats(z['timeseries_scores'][covered],y[covered],tc)
    diff=scores[covered]-z['timeseries_scores'][covered]
    events=[]
    pred=scores>tc;pred_oracle=scores>oracle
    for index,(start,end) in enumerate(truth,1):
        mask=covered[start:end];sc=scores[start:end][mask];length=end-start
        if len(sc)==0:
            events.append(dict(machine=machine,event=index,start=start,end=end,duration=length,covered=0));continue
        hit=np.flatnonzero(pred[start:end]&mask);ohit=np.flatnonzero(pred_oracle[start:end]&mask)
        events.append(dict(machine=machine,event=index,start=start,end=end,duration=length,covered=int(mask.sum()),
            peak_score=float(sc.max()),median_score=float(np.median(sc)),
            peak_over_calibrated=float(sc.max()/tc),
            calibrated_hit=bool(len(hit)),oracle_hit=bool(len(ohit)),
            calibrated_coverage=float(np.mean(sc>tc)),oracle_coverage=float(np.mean(sc>oracle)),
            calibrated_first_hit_offset=int(hit[0]) if len(hit) else None))
    eligible=[e for e in events if e['covered']]
    normal=scores[covered&~y];positive=scores[covered&y]
    fpr,tpr,_=roc_curve(y[covered],scores[covered])
    row=dict(machine=machine,source=str(path.relative_to(ROOT)),timesteps=n,covered_timesteps=int(covered.sum()),
        output_window=int(z['end_idxs'][0]-z['start_idxs'][0]),windows=len(window),
        timeline_threshold=tc,timeline_oracle_threshold=oracle,window_threshold=c,window_oracle_threshold=wo,
        timeline_prevalence=float(y[covered].mean()),window_prevalence=float(wy.mean()),
        timeline_vus_pr=report['evaluation']['timeseries']['vus_pr'],timeline_vus_roc=report['evaluation']['timeseries']['vus_roc'],
        window_vus_pr=report['evaluation']['window']['vus_pr'],window_vus_roc=report['evaluation']['window']['vus_roc'],
        trace_max_abs_difference=float(abs(diff).max()),trace_rmse=float(np.sqrt(np.mean(diff**2))),
        trace_npz_decision_flips=int(np.sum((scores[covered]>tc)!=(z['timeseries_scores'][covered]>tc))),
        trace_window_max_abs_difference=float(abs(window-z['window_scores']).max()),
        trace_window_decision_flips=int(np.sum((window>c)!=(z['window_scores']>c))),
        timeline_events=len(eligible),events_calibrated_hit=sum(e['calibrated_hit'] for e in eligible),
        events_oracle_hit=sum(e['oracle_hit'] for e in eligible),
        event_recall_calibrated=float(np.mean([e['calibrated_hit'] for e in eligible])),
        event_recall_oracle=float(np.mean([e['oracle_hit'] for e in eligible])),
        event_coverage_calibrated_median=float(np.median([e['calibrated_coverage'] for e in eligible])),
        normal_median=float(np.median(normal)),anomaly_median=float(np.median(positive)),normal_p99=float(np.quantile(normal,.99)),
        recall_at_1pct_fpr=float(tpr[fpr<=.01].max()),recall_at_5pct_fpr=float(tpr[fpr<=.05].max()),
        false_positive_runs_calibrated=len(intervals(pred&covered&~y)),
        peak_timestep=int(np.nanargmax(scores)),peak_is_anomaly=bool(y[np.nanargmax(scores)]),
        top_1pct_precision=float(np.mean(y[covered][scores[covered]>=np.quantile(scores[covered],.99)])),
        window_always_positive_f1=2*float(wy.mean())/(1+float(wy.mean())),
        timeline_always_positive_f1=2*float(y[covered].mean())/(1+float(y[covered].mean())),
        **flatten('timeline_cal',cal),**flatten('timeline_oracle',om),**flatten('timeline',extra),
        **flatten('npz_timeline_cal',official),**flatten('window_cal',wc),**flatten('window_oracle',wom),
        **flatten('window',wextra),**flatten('replay_window_cal',replay_wc),**flatten('replay_window_oracle',replay_wom))
    assert abs(wc['f1']-report['evaluation']['window']['calibrated']['f1_no_pa'])<1e-12
    assert abs(wom['f1']-report['evaluation']['window']['oracle']['f1_no_pa'])<1e-12
    fig,axes=plt.subplots(3,1,figsize=(14,9),gridspec_kw={'height_ratios':[1.3,1.3,1]})
    draw_score(axes[0],np.arange(n),scores,tc,oracle,truth,
        f'Full-series replay: F1 {cal["f1"]:.3f} → oracle {om["f1"]:.3f}; event hits {row["events_calibrated_hit"]}/{len(eligible)}; normal FPR {cal["fpr"]:.1%}')
    axes[0].axvspan(0,np.flatnonzero(covered)[0],color='gray',alpha=.12)
    centers=(z['start_idxs']+z['end_idxs'])/2
    edges=np.r_[centers[0]-(centers[1]-centers[0])/2,(centers[:-1]+centers[1:])/2,centers[-1]+(centers[-1]-centers[-2])/2]
    wtruth=[(edges[start],edges[end]) for start,end in intervals(wy)]
    draw_score(axes[1],centers,z['window_scores'],c,wo,wtruth,
        f'Window means (512-step target): F1 {wc["f1"]:.3f} → oracle {wom["f1"]:.3f}; window VUS-PR {row["window_vus_pr"]:.3f}, ROC {row["window_vus_roc"]:.3f}')
    axes[1].plot(centers,window/c,color='#999999',lw=.6,alpha=.6,label='Replay-derived means')
    for ax in axes[:2]:ax.set_xlim(0,n);ax.set_xlabel('Original timestep (window means plotted at target centers)')
    axes[0].legend(loc='upper left',fontsize=8)
    for sc,lab,threshold,name,color in [(scores[covered],y[covered],tc,'Timeline','#245991'),(z['window_scores'],wy,c,'Window','#9b5443')]:
        p,r,t=precision_recall_curve(lab,sc);fs=np.divide(2*p[:-1]*r[:-1],p[:-1]+r[:-1],out=np.zeros_like(t),where=p[:-1]+r[:-1]>0)
        axes[2].plot(t/threshold,fs,color=color,label=name,lw=1.2)
    axes[2].axvline(1,color='#128375',ls='--');axes[2].set(xscale='log',xlabel='Threshold / current calibrated threshold',ylabel='Ordinary F1',ylim=(0,1.02))
    axes[2].legend(loc='best',fontsize=8)
    fig.suptitle(f'{machine.removesuffix(".txt")} · frequency_aux / observed_features\nPurple = actual anomalous timesteps (top); positive-window center ranges (middle). No point adjustment.',fontsize=13)
    fig.tight_layout();fig.savefig(OUT/'machines'/f'{machine.removesuffix(".txt")}.png',bbox_inches='tight');plt.close(fig)
    # Compact, exact final graph data supports fast human inspection and rerendering.
    np.savez_compressed(OUT/'data'/f'{machine.removesuffix(".txt")}.npz',
        replay_scores=scores,covered=covered,labels=y,window_scores=z['window_scores'],
        replay_window_scores=window,window_labels=wy,centers=centers,
        timeline_threshold=tc,timeline_oracle_threshold=oracle,window_threshold=c,window_oracle_threshold=wo)
    del trace;gc.collect()
    print(f'{machine}: timeline {cal["f1"]:.3f}/{om["f1"]:.3f}; window {wc["f1"]:.3f}/{wom["f1"]:.3f}; replay flips {row["trace_npz_decision_flips"]}',flush=True)
    return row,events


def contacts(rows):
    for page,start in enumerate(range(0,len(rows),7),1):
        group=rows[start:start+7];fig,axes=plt.subplots(len(group),2,figsize=(16,2.35*len(group)),squeeze=False)
        for axesrow,row in zip(axes,group):
            z=np.load(OUT/'data'/f'{row["machine"].removesuffix(".txt")}.npz');truth=intervals(z['labels'])
            draw_score(axesrow[0],np.arange(len(z['replay_scores'])),z['replay_scores'],row['timeline_threshold'],row['timeline_oracle_threshold'],truth,
                f'{row["machine"]}  TIMELINE  F1 {row["timeline_cal_f1"]:.2f} → {row["timeline_oracle_f1"]:.2f}')
            wc=z['window_labels'];centers=z['centers'];edges=np.r_[centers[0],(centers[:-1]+centers[1:])/2,centers[-1]]
            bands=[(edges[a],edges[b]) for a,b in intervals(wc)]
            draw_score(axesrow[1],centers,z['window_scores'],row['window_threshold'],row['window_oracle_threshold'],bands,
                f'WINDOW MEANS  F1 {row["window_cal_f1"]:.2f} → {row["window_oracle_f1"]:.2f}')
            for ax in axesrow:ax.set_ylabel('Score / threshold',fontsize=8);ax.tick_params(labelsize=8)
        fig.suptitle(f'All-machine visual audit · page {page}/4\nBlue = actual scores; green dashed = calibrated; orange dotted = oracle; purple = labels; log1p scale',fontsize=13)
        fig.tight_layout(rect=(0,0,1,.965));fig.savefig(OUT/f'contact_{page}.png',bbox_inches='tight');plt.close(fig)


def main():
    (OUT/'machines').mkdir(exist_ok=True);(OUT/'data').mkdir(exist_ok=True)
    rows=[];events=[]
    for path in sorted(BASE.glob('machine-*/*/decision_trace.json')):
        row,es=run(path);rows.append(row);events.extend(es)
    assert len(rows)==28
    pd.DataFrame(rows).to_csv(OUT/'machine_summary.csv',index=False)
    pd.DataFrame(events).to_csv(OUT/'events.csv',index=False)
    contacts(rows)
    summary=pd.DataFrame(rows).mean(numeric_only=True).to_dict()
    (OUT/'summary.json').write_text(json.dumps(dict(machines=len(rows),macro_means=summary,
        trace_support_exact=True,all_traces_complete=True),indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
