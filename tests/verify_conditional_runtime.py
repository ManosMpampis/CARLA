"""Independent dataset, real training, validation and checkpoint acceptance."""

import argparse
import copy
import json
from pathlib import Path
import random
import tempfile

from conditional_support import close, format_fixtures, output_config, run_cases, tiny_config

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.jepa_dataset import JEPADataset, make_synthetic_series
from models import get_model
from train import check_model_loss, run
from utils.common_config import get_criterion, get_jepa_datasets, get_optimizer, get_scheduler
from utils.trainer import Trainer
from utils.reconstruction_baselines import score_both
from utils.scoring import model_evaluation_options


def data_normalization_and_splits():
    fixture=format_fixtures()
    config=tiny_config("ae",window=24)
    config.update(val_fraction=.25,stride=7)
    details=[]
    for source,collection,fname,channels in (
        ("psm","psm","psm",3),("smd","smd","machine-1-1.txt",3),
        ("tsb_ad","tsb-ad-u",fixture["files"]["tsb-ad-u"][0],1),
        ("tsb_ad","tsb-ad-m",fixture["files"]["tsb-ad-m"][0],3)):
        p={**config,"train_db_name":source,"dataset_collection":collection,
           "fname":fname,"dataset_root":str(fixture[collection])}
        tr,va=get_jepa_datasets(p);te=JEPADataset(p,train=False)
        raw=fixture["raw"][:,:channels]
        test=fixture["test"][:,:channels]
        if source=="smd":raw=raw+1;test=test+1
        expected_mean=raw[:120].astype(np.float64).mean(axis=0)
        expected_std=raw[:120].astype(np.float64).std(axis=0)
        close(tr.mean,expected_mean,atol=1e-6)
        close(tr.std,expected_std,atol=1e-6)
        close(te.mean,tr.mean);close(te.std,tr.std)
        close(tr.series,(raw[:120]-expected_mean)/expected_std,atol=2e-6)
        close(va.series,(raw[120:]-expected_mean)/expected_std,atol=2e-6)
        close(te.series,(test-expected_mean)/expected_std,atol=2e-6)
        close(tr.full_train_series,(raw-expected_mean)/expected_std,atol=2e-6)
        assert len(tr.series)==120 and len(va.series)==40
        assert te.series.shape==(112,channels)
        assert np.array_equal(te.targets,fixture["labels"])
        if source=="tsb_ad":assert te.series_start==160
        details.append({"collection":collection,"train_rows":120,"validation_rows":40,
                        "test_rows":112,"features":channels})
    # Synthetic train and test must reuse exactly the same optimization-row scaler.
    config=tiny_config("ae",window=24)
    config.update(val_fraction=.25)
    raw,_=make_synthetic_series(176,3,seed=17)
    tr,va=get_jepa_datasets(config);te=JEPADataset(config,train=False)
    close(tr.mean,raw[:132].astype(np.float64).mean(axis=0),atol=1e-6)
    close(te.mean,tr.mean);close(te.std,tr.std)
    assert len(tr.series)==132 and len(va.series)==44
    return {"fixtures":str(fixture["base"]),"formats":details,"synthetic_shared_scaler":True}


def short_series_and_pooled_boundaries():
    fixture=format_fixtures()
    p=output_config(tiny_config("ae",window=24),fixture["base"]/"pooled-output")
    p.update(train_db_name="smd",fname="all-one",dataset_root=str(fixture["smd"]),val_fraction=.25)
    tr,va=get_jepa_datasets(p)
    assert len(tr.machine_files)==2
    for split in (tr,va):
        for index in range(len(split)):
            item=split[index];meta=item["meta"]
            i=int(meta["machine_file"].split("-")[-1].split(".")[0])
            raw=fixture["raw"]*i+i
            mean=raw[:120].astype(np.float64).mean(0)
            std=raw[:120].astype(np.float64).std(0)
            expected=(raw[meta["start_idx"]:meta["end_idx"]]-mean)/std
            close(item["ts"],expected,atol=3e-6)
            assert meta["end_idx"] <= (120 if split is tr else 160)
            assert meta["start_idx"] >= (0 if split is tr else 120)
    p=tiny_config("ae",window=48)
    p["synthetic_kwargs"].update(n_steps=64)
    tr,va=get_jepa_datasets(p)
    assert len(tr)>0 and len(va)==0,"short data must not fabricate held-out validation"
    assert len(tr.series)==len(tr.full_train_series)==64
    p["synthetic_kwargs"].update(n_steps=96)
    p["val_fraction"]=.1
    boundary_tr,boundary_val=get_jepa_datasets(p)
    assert len(boundary_tr)==len(boundary_val)==1,"two legal windows permit disjoint validation"
    raw,_=make_synthetic_series(96,3,seed=17)
    close(boundary_tr.mean,raw[:48].astype(np.float64).mean(0),atol=1e-6)
    return {"short_validation_windows":len(va),"two_window_boundary":[len(boundary_tr),len(boundary_val)]}


def sample_weighted_epoch_summaries():
    p=tiny_config("ae",window=24)
    model=get_model(p,input_channels=3)
    criterion=get_criterion(p)
    optimizer=torch.optim.SGD(model.parameters(),lr=0.)
    trainer=Trainer(p,model,criterion,optimizer,None,torch.device("cpu"),None)
    # Nine items give 4+4+1. The final item has deliberately much larger loss.
    values=[{"ts":torch.full((24,3),float(i)/2)} for i in range(9)]
    loader=DataLoader(values,batch_size=4,shuffle=False)
    expected=0.;batch_means=[]
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x=batch["ts"].transpose(1,2)
            value=float(criterion(model(x))["loss"])
            expected+=value*x.size(0);batch_means.append(value)
    expected/=9
    assert abs(np.mean(batch_means)-expected)>0.1,"fixture must distinguish batch and sample means"
    trainer.validate(loader)
    close(trainer.validation_metrics["loss"],expected)
    training=trainer.train_one_epoch(loader,0)
    close(training["loss"],expected)
    return {"expected_sample_mean":expected,"incorrect_batch_mean":float(np.mean(batch_means)),
            "train_mean":training["loss"],"validation_mean":trainer.validation_metrics["loss"]}


def validation_rng_and_preflight():
    p=tiny_config("lewm",window=32)
    p["criterion_kwargs"]["sigreg_kwargs"]["seed"]=29
    model=get_model(p,input_channels=3)
    criterion=get_criterion(p)
    optimizer=get_optimizer(p,model)
    trainer=Trainer(p,model,criterion,optimizer,None,torch.device("cpu"),None)
    values=[{"ts":torch.randn(32,3)} for _ in range(9)]
    loader=DataLoader(values,batch_size=4,shuffle=False)
    original={k:v.detach().clone() for k,v in model.state_dict().items() if torch.is_tensor(v)}
    torch_before=torch.get_rng_state().clone();np_before=np.random.get_state();py_before=random.getstate()
    first=trainer.validate(loader);second=trainer.validate(loader)
    close(first,second,atol=0,rtol=0)
    assert torch.equal(torch_before,torch.get_rng_state()),"validation must not consume train Torch RNG"
    np_after=np.random.get_state()
    assert np_before[0]==np_after[0] and np.array_equal(np_before[1],np_after[1]) and np_before[2:]==np_after[2:]
    assert py_before==random.getstate()
    for key,value in model.state_dict().items():
        if torch.is_tensor(value):close(value,original[key],atol=0,rtol=0)
    first_batch=next(iter(loader))
    torch_before=torch.get_rng_state().clone();np_before=np.random.get_state()
    check_model_loss(model,criterion,first_batch,torch.device("cpu"))
    assert torch.equal(torch_before,torch.get_rng_state()),"preflight must preserve train Torch RNG"
    assert np.array_equal(np_before[1],np.random.get_state()[1])
    return {"repeat_validation_loss":first,"validation_parameters_and_buffers_unchanged":True}


def deterministic_resume():
    base=Path(tempfile.mkdtemp(prefix="lewm-deterministic-resume-"))
    details={}
    for name in ("vae","lewm"):
        config=tiny_config(name)
        if name=="lewm":config["criterion_kwargs"]["sigreg_kwargs"]["seed"]=43
        full=output_config(config,base/f"{name}-full")
        full.update(epochs=3,eval_every=0)
        run(full)
        interrupted=output_config(config,base/f"{name}-interrupted")
        interrupted.update(epochs=1,eval_every=0)
        run(interrupted)
        interrupted["epochs"]=3
        run(interrupted)
        first=torch.load(full["jepa_checkpoint"],weights_only=False)
        second=torch.load(interrupted["jepa_checkpoint"],weights_only=False)
        assert first["epoch"]==second["epoch"]==3
        for key,value in first["model"].items():
            if torch.is_tensor(value):close(value,second["model"][key],atol=0,rtol=0)
        assert first["losses"]==second["losses"],"resume must preserve stochastic losses"
        assert "rng_state" in second
        if name=="lewm":assert "criterion_rng_state" in second
        details[name]={"full":full["jepa_dir"],"resumed":interrupted["jepa_dir"],
                       "final_loss":second["losses"],"bitwise_weights_equal":True}
    return details


def real_data_samples():
    from conditional_support import ROOT
    results={}
    for collection in ("psm","smd","tsb-ad-u","tsb-ad-m"):
        root=ROOT/"datasets"/{"psm":"PSM","smd":"SMD","tsb-ad-u":"TSB-AD-U","tsb-ad-m":"TSB-AD-M"}[collection]
        if not root.is_dir():
            results[collection]={"available":False,"limitation":"real collection not installed; format fixture only"}
            continue
        p=tiny_config("lewm")
        p.update(train_db_name="tsb_ad" if collection.startswith("tsb") else collection,
                 dataset_collection=collection,dataset_root=str(root),fname="psm")
        if collection=="smd":p["fname"]=sorted((root/"train").glob("machine-*.txt"))[0].name
        elif collection.startswith("tsb"):p["fname"]=sorted(root.glob("*.csv"))[0].name
        tr,va=get_jepa_datasets(p);test=JEPADataset(p,train=False)
        batch=torch.from_numpy(np.stack([tr[0]["ts"],tr[len(tr)-1]["ts"]])).transpose(1,2)
        model=get_model(p,input_channels=batch.size(1)).eval()
        loss=get_criterion(p)(model(batch))["loss"]
        scores=model.score(torch.from_numpy(test[0]["ts"]).T.unsqueeze(0))["fused"]
        assert torch.isfinite(loss) and torch.isfinite(scores).all()
        results[collection]={"available":True,"root":str(root.resolve()),"fname":p["fname"],
                             "feature_count":batch.size(1),"train_windows":len(tr),
                             "validation_windows":len(va),"test_windows":len(test),
                             "sample_loss":float(loss.detach()),"score_shape":list(scores.shape)}
    return results


def normal_entry_cuda_amp():
    if not torch.cuda.is_available():return {"available":False}
    base=Path(tempfile.mkdtemp(prefix="lewm-normal-amp-"))
    config=output_config(tiny_config("lewm"),base)
    config.update(epochs=1,eval_every=0,amp=True,device="cuda")
    best=run(config)
    saved=torch.load(config["jepa_checkpoint"],map_location="cpu",weights_only=False)
    assert np.isfinite(best) and "scaler" in saved
    assert all(torch.isfinite(value).all() for value in saved["model"].values() if torch.is_tensor(value))
    return {"available":True,"directory":str(base),"best_validation":best,"scaler_saved":True}


def training_and_score_reload():
    base=Path(tempfile.mkdtemp(prefix="lewm-conditional-training-"))
    results={}
    for name in ("ae","vae","lewm","guided_forecast"):
        p=output_config(tiny_config(name),base/name)
        p.update(epochs=2,eval_every=1,save_every=1)
        best=run(p)
        directory=Path(p["jepa_dir"])
        assert np.isfinite(best)
        history=[json.loads(line) for line in (directory/"epochs.jsonl").read_text().splitlines()]
        assert [row["epoch"] for row in history]==[1,2]
        for row in history:
            assert row["losses"]["validation"] is not None
            assert all(np.isfinite(v) for split in row["losses"].values() for v in split.values())
            assert set(row["evaluation"])=={"window","non_overlap_window","timeseries"}
            for metrics in row["evaluation"].values():
                assert "calibrated" in metrics and "oracle" in metrics
        expected_files={"checkpoint.pth.tar","best_validation_loss.pth.tar"}
        expected_files.update(f"best_test_{mode}_oracle_f1.pth.tar" for mode in
                              ("window","non_overlap_window","timeseries"))
        assert expected_files <= {f.name for f in directory.glob("*.pth.tar")}
        primary=torch.load(directory/"best_validation_loss.pth.tar",weights_only=False)
        assert primary["selection"]["source"]=="validation"
        for mode in ("window","non_overlap_window","timeseries"):
            diagnostic=torch.load(directory/f"best_test_{mode}_oracle_f1.pth.tar",weights_only=False)
            assert diagnostic["selection"]["source"]=="test"
            assert diagnostic["selection"]["threshold_source"]=="oracle"
            metrics=diagnostic["evaluation"][mode]
            close(diagnostic["selection"]["value"],metrics["oracle"]["f1"])
            assert "calibrated" in metrics
        score_config=copy.deepcopy(p);score_config["stage"]="score"
        report=run(score_config)
        first=np.load(directory/"scores.npz")
        arrays={key:first[key].copy() for key in first.files};first.close()
        thresholds=json.loads((directory/"calibration.json").read_text())
        assert thresholds["source"]=="full clean training series"
        trained=get_model(primary["config"],input_channels=3)
        trained.load_state_dict(primary["model"],strict=True)
        trained.eval()
        clean_data=JEPADataset(primary["config"],train=True)
        options=model_evaluation_options(primary["config"],trained)
        window,step=options.pop("wsz"),options.pop("stride")
        for mode in ("window","timeseries","non_overlap_window"):
            local=dict(options)
            stride=step
            if mode=="non_overlap_window":
                span=local.get("check_mask")
                stride=span[1]-span[0] if span is not None else window
                local["include_last_window"]=False
            clean=score_both(trained,clean_data.full_train_series,window,stride,
                              int(p["batch_size"]),torch.device("cpu"),**local)
            distribution=(clean["timeseries_scores"][clean["cover_counts"]>0]
                          if mode=="timeseries" else clean["window_scores"])
            expected_threshold=float(np.quantile(distribution,p["calibration_quantile"]))
            close(thresholds["thresholds"][mode],expected_threshold)
        # Independently reconstruct scored target coverage from exported intervals.
        counts=np.zeros(len(arrays["cover_counts"]),dtype=int)
        for start,end in zip(arrays["starts"],arrays["ends"]):counts[int(start):int(end)]+=1
        assert np.array_equal(counts,arrays["cover_counts"])
        if name=="guided_forecast":
            assert arrays["starts"][0]==32 and np.all(counts[:32]==0)
        else:assert arrays["starts"][0]==0
        run(score_config)
        second=np.load(directory/"scores.npz")
        for key,value in arrays.items():
            assert np.array_equal(value,second[key],equal_nan=True),f"score reload changed {key}"
        assert thresholds==json.loads((directory/"calibration.json").read_text())
        second.close()
        results[name]={"best":best,"directory":str(directory),"history":history,
                       "score_report":report,"calibration":thresholds,
                       "architecture":str(get_model(p,input_channels=3))}
    return {"base":str(base),"runs":results}


def short_training_checkpoint():
    base=Path(tempfile.mkdtemp(prefix="lewm-short-training-"))
    p=output_config(tiny_config("ae",window=48),base)
    p["synthetic_kwargs"].update(n_steps=64)
    p.update(epochs=2)
    best=run(p)
    assert np.isfinite(best)
    assert (base/"best_train_loss.pth.tar").is_file()
    assert not (base/"best_validation_loss.pth.tar").exists()
    primary=torch.load(base/"best_train_loss.pth.tar",weights_only=False)
    assert primary["selection"]["source"]=="train"
    assert not primary["config"]["validation_available"]
    assert primary["losses"].get("validation") in (None,{})
    report=run({**p,"stage":"score"})
    return {"directory":str(base),"best_training":best,"score_report":report}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--case",action="append",help="Run named case(s), otherwise all")
    args=parser.parse_args()
    cases=[("data normalization and splits",data_normalization_and_splits),
           ("short and pooled data",short_series_and_pooled_boundaries),
           ("sample weighted summaries",sample_weighted_epoch_summaries),
           ("validation RNG and preflight",validation_rng_and_preflight),
           ("training and score reload",training_and_score_reload),
           ("short training checkpoint",short_training_checkpoint),
           ("deterministic resume",deterministic_resume),("real data samples",real_data_samples),
           ("normal entry CUDA AMP",normal_entry_cuda_amp)]
    if args.case:cases=[item for item in cases if item[0] in args.case]
    assert cases,"no matching cases"
    records=run_cases(cases,args.output)
    raise SystemExit(any(row["status"]!="passed" for row in records))
