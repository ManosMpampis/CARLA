"""Build descriptive comparisons and static figures from analyze.py exports."""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/carla-smd-report-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_curve, roc_curve

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
plt.rcParams.update({"figure.dpi":150,"savefig.dpi":170,"font.size":10,
                     "axes.spines.top":False,"axes.spines.right":False,
                     "axes.titleweight":"bold","axes.grid":True,"grid.alpha":.17})


def short(key):
    return key.replace("lewm_encoder/", "").replace("/cross_attention/", " / CA ").replace("/reconstruction/", " / recon ").replace("auxilary_added","aux added")


def save(fig, name):
    fig.savefig(OUT/(name+".png"),bbox_inches="tight")
    plt.close(fig)


def main():
    f=pd.read_csv(OUT/"selected_runs.csv")
    a=pd.read_csv(OUT/"family_summary.csv").set_index("experiment")
    f=f[f.experiment.isin(a.index)].copy()
    h=pd.read_csv(OUT/"epoch_metrics.csv")
    s=pd.read_csv(OUT/"training_scalars.csv.gz")
    variance=[]
    for (key,machine),g in s[s.tag=="Train Loss/latent_var"].groupby(["experiment","machine"]):
        if key.count("/") != 1:continue
        values=g.sort_values("epoch").value.to_numpy()
        variance.append(dict(experiment=key,machine=machine,first_var=values[0],
                             last_var=values[-1],tail_var_mean=values[-100:].mean()))
    pd.DataFrame(variance).to_csv(OUT/"latent_variance.csv",index=False)
    sw=pd.read_csv(OUT/"inference_cases.csv")
    full=a[a.output_window==1024]
    cross=a[a.output_window==512]
    shortlist=["lewm_encoder/frequency","lewm_encoder/frequency_aux","lewm_encoder/frequency_aux_no_film",
               "ae/default","vae/default","lewm_encoder/time","lewm_encoder/frequency/reconstruction/default",
               "lewm_encoder/frequency_aux/cross_attention/observed_query",
               "lewm_encoder/frequency_aux_no_film/cross_attention/observed_query",
               "lewm_encoder/frequency_aux/cross_attention/observed_features"]
    q=a.loc[shortlist].iloc[::-1]
    fig,axes=plt.subplots(1,2,figsize=(14,6),sharey=True)
    yy=np.arange(len(q))
    axes[0].barh(yy+.17,q.cal_f1,height=.32,color="#167c80",label="Clean-calibrated F1")
    axes[0].barh(yy-.17,q.oracle_f1,height=.32,color="#bed5d2",label="Test-oracle F1")
    axes[1].plot(q.vus_pr,yy,"o",color="#a05426",label="VUS-PR")
    axes[1].plot(q.vus_roc,yy,"s",color="#584e9a",label="VUS-ROC")
    axes[0].set_yticks(yy,[short(x)+f" [{int(q.loc[x,'output_window'])}]" for x in q.index])
    for ax in axes:ax.set_xlim(0,1);ax.legend(loc="lower right");ax.set_xlabel("Equal-machine mean (28 SMD machines)")
    fig.suptitle("Window-only results: good ranking does not guarantee a usable threshold\n[brackets] = scored output length; 1024 and 512 are different labeling tasks",fontsize=13)
    save(fig,"leaderboard")

    keys=shortlist[:7]+["lewm_encoder/frequency_aux/reconstruction/auxilary_added"]
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    handles=[]
    for number,key in enumerate(keys,1):
        r=a.loc[key]
        handle=axes[0].scatter(r.tail_threshold_cv,r.tail_cal_std,s=65)
        handles.append(handle)
        axes[0].annotate(str(number),(r.tail_threshold_cv,r.tail_cal_std),xytext=(4,4),textcoords="offset points",fontsize=8)
        axes[1].scatter(r.tail_threshold_cv,r.gap,s=65)
        axes[1].annotate(str(number),(r.tail_threshold_cv,r.gap),xytext=(4,4),textcoords="offset points",fontsize=8)
    axes[0].set_ylabel("Late calibrated F1 standard deviation")
    axes[1].set_ylabel("Selected oracle F1 − calibrated F1")
    for ax in axes:ax.set_xlabel("Late clean-threshold CV (lower = steadier scale)")
    fig.legend(handles,[f"{i}. {short(k)}" for i,k in enumerate(keys,1)],loc="center left",bbox_to_anchor=(1.01,.5),fontsize=9)
    fig.suptitle("Stability and calibration transfer measure different things (family means)")
    save(fig,"stability")

    cases=[("lewm_encoder/frequency","machine-3-11.txt"),
           ("lewm_encoder/frequency","machine-1-7.txt"),
           ("lewm_encoder/frequency","machine-3-3.txt")]
    fig,axes=plt.subplots(3,3,figsize=(14,10),sharex="col")
    for col,(key,machine) in enumerate(cases):
        hh=h[(h.experiment==key)&(h.machine==machine)].sort_values("epoch")
        ss=s[(s.experiment==key)&(s.machine==machine)&s.tag.str.startswith("validation/")].sort_values("epoch")
        axes[0,col].plot(ss.epoch,ss.value,color="#42647f");axes[0,col].set_title(f"frequency / {machine.replace('.txt','')}")
        axes[1,col].plot(hh.epoch,hh.cal_f1,label="Calibrated",color="#167c80")
        axes[1,col].plot(hh.epoch,hh.oracle_f1,label="Oracle",color="#a05426")
        axes[1,col].set_ylim(0,1)
        axes[2,col].plot(hh.epoch,hh.threshold,label="Clean q=.995",color="#167c80")
        axes[2,col].plot(hh.epoch,hh.oracle_threshold,label="Oracle",color="#a05426")
        axes[2,col].set_yscale("log");axes[2,col].set_xlabel("Training epoch")
    axes[0,0].set_ylabel("Validation prediction loss")
    axes[1,0].set_ylabel("Window F1")
    axes[2,0].set_ylabel("Window threshold (log)")
    axes[1,2].legend();axes[2,2].legend()
    fig.suptitle("Training loss, ranking and thresholds follow different trajectories",fontsize=14)
    fig.tight_layout();save(fig,"training_examples")

    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    divergence=[]
    for variant,color in [("time_aux","#167c80"),("frequency_aux","#a05426")]:
        key=f"lewm_encoder/{variant}/cross_attention/observed_query"
        row=f[(f.experiment==key)&(f.machine=="machine-2-3.txt")].iloc[0]
        with np.load(ROOT/row.path.replace("metrics.json","scores.npz")) as z:sc,lab=z["window_scores"],z["window_labels"]
        p,r,_=precision_recall_curve(lab,sc);fp,tp,_=roc_curve(lab,sc)
        axes[0].plot(r,p,color=color,label=variant)
        axes[1].plot(fp,tp,color=color,label=variant)
        divergence.append(dict(experiment=key,machine=row.machine,prevalence=row.prevalence,
                               vus_pr=row.vus_pr,vus_roc=row.vus_roc,cal_f1=row.cal_f1,
                               oracle_f1=row.oracle_f1,vanilla_ap=row.vanilla_ap,vanilla_roc=row.vanilla_roc,
                               tpr_at_fpr_1pct=float(tp[fp<=.01].max()),
                               tpr_at_fpr_5pct=float(tp[fp<=.05].max()),
                               top_5pct_precision=float(lab[sc>=np.quantile(sc,.95)].mean())))
    axes[0].axhline(float(row.prevalence),color="gray",ls="--",label="Positive-window prevalence")
    axes[0].set(xlabel="Recall",ylabel="Precision",xlim=(0,1),ylim=(0,1))
    axes[1].plot([0,1],[0,1],"--",color="gray");axes[1].set(xlabel="False-positive rate",ylabel="True-positive rate",xlim=(0,1),ylim=(0,1))
    for ax in axes:ax.legend()
    fig.suptitle("Same machine, same 512-step output: curves cross\nOrdinary PR/ROC curves explain the ranking; these plots are not VUS curves",fontsize=12)
    save(fig,"metric_divergence")

    fig,axes=plt.subplots(1,3,figsize=(14,4.2))
    for ax,(key,machine) in zip(axes,cases):
        row=f[(f.experiment==key)&(f.machine==machine)].iloc[0]
        with np.load(ROOT/row.path.replace("metrics.json","scores.npz")) as z:sc,lab=z["window_scores"],z["window_labels"]
        p,r,t=precision_recall_curve(lab,sc)
        fs=np.divide(2*p[:-1]*r[:-1],p[:-1]+r[:-1],out=np.zeros_like(t),where=(p[:-1]+r[:-1])>0)
        ax.plot(t,fs,color="#42647f");ax.axvline(row.threshold,color="#167c80",label="Clean threshold")
        ax.axvline(row.oracle_threshold,color="#a05426",ls="--",label="Oracle")
        ax.axhline(row.always_positive_f1,color="gray",ls=":",label="Always alert F1")
        ax.set(xscale="log",ylim=(0,1.03),xlabel="Threshold (log)",title=machine.replace('.txt',''))
        lower=min(float(sc[sc>0].min()),row.threshold,row.oracle_threshold)
        upper=max(float(sc.max()),row.threshold,row.oracle_threshold)
        ticks=np.geomspace(lower,upper,4)
        ax.set_xticks(ticks,[f"{v:.2g}" for v in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylabel("Window F1");axes[2].legend(fontsize=8)
    fig.suptitle("Three threshold failure modes: under-thresholding, over-thresholding, trivial oracle")
    save(fig,"threshold_examples")

    heatkeys=shortlist[:7]+[shortlist[7]]
    heat=f[f.experiment.isin(heatkeys)].pivot(index="experiment",columns="machine",values="cal_f1").loc[heatkeys]
    fig,ax=plt.subplots(figsize=(15,4.8));im=ax.imshow(heat,aspect="auto",vmin=0,vmax=1,cmap="YlGnBu")
    ax.set_yticks(range(len(heat)),[short(x) for x in heat.index]);ax.set_xticks(range(len(heat.columns)),[x.replace("machine-","").replace(".txt","") for x in heat.columns],rotation=65)
    ax.set_title("Calibrated window F1 varies much more across machines than the leading means")
    fig.colorbar(im,ax=ax,label="Calibrated F1");save(fig,"machine_heatmap")

    rng=np.random.default_rng(4);pairs=[]
    pivot=f.pivot(index="machine",columns="experiment",values="cal_f1")
    for x,y in [("lewm_encoder/frequency","ae/default"),("lewm_encoder/frequency_aux","lewm_encoder/frequency"),("lewm_encoder/frequency_aux_no_film","lewm_encoder/frequency_aux")]:
        delta=(pivot[x]-pivot[y]).to_numpy();boots=rng.choice(delta,(20000,len(delta)),replace=True).mean(axis=1)
        pairs.append(dict(a=x,b=y,mean_difference=float(delta.mean()),machines_a_better=int((delta>0).sum()),
                          paired_machine_bootstrap_ci95=np.quantile(boots,[.025,.975]).tolist()))
    correlations=[]
    for (machine,output),gg in f.groupby(["machine","output_window"]):
        if len(gg)<4:continue
        correlations.append(dict(machine=machine,output_window=int(output),
                                 pr_roc_spearman=float(spearmanr(gg.vus_pr,gg.vus_roc).statistic),
                                 cal_pr_spearman=float(spearmanr(gg.cal_f1,gg.vus_pr).statistic)))
    pd.DataFrame(correlations).to_csv(OUT/"within_machine_correlations.csv",index=False)
    # Family summaries are descriptive, never a composite hand-weighted score.
    details=dict(paired_comparisons=pairs,divergence_example=divergence,
                 correlation_means=pd.DataFrame(correlations).groupby("output_window").mean(numeric_only=True).to_dict(),
                 full_span_always_positive_f1_mean=float(f[f.output_window==1024].always_positive_f1.mean()),
                 half_span_always_positive_f1_mean=float(f[f.output_window==512].always_positive_f1.mean()))
    (OUT/"comparisons.json").write_text(json.dumps(details,indent=2))
    cols=["cal_f1","oracle_f1","vus_pr","vus_roc","gap","near_oracle_selected","oracle_in_range","tail_cal_std","tail_threshold_cv","val_tail_cv","val_cal_spearman","cal_epoch_near_peak_fraction","oracle_epoch_near_peak_fraction","selected_vs_peak_cal","above_trivial_f1","test_normal_fpr","precision","recall"]
    print(a.loc[shortlist,cols].round(4).to_string())
    print(json.dumps(details,indent=2))


if __name__ == "__main__":main()
