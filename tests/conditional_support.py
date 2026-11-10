"""Small independent acceptance helpers; generated experiments stay in /tmp."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import numpy as np
import pandas as pd

from utils.config import _training_sections, load_experiment_config


def preset(name):
    return _training_sections(load_experiment_config(ROOT / "configs/training" / f"{name}.yml"))


def tiny_config(name="lewm", *, channels=3, window=48, stride=1):
    config = copy.deepcopy(preset(name))
    config.update(wsz=window, stride=8, batch_size=4, num_workers=0,
                  epochs=2, eval_every=1, save_every=1, seed=17, device="cpu",
                  aggregate_output_strategy="all", eval_window_size=4,
                  train_db_name="synthetic", val_db_name="synthetic", fname="synthetic",
                  synthetic_kwargs={"n_steps":176, "test_n_steps":112, "n_channels":channels},
                  val_fraction=0.3)
    if name in ("ae", "vae"):
        config["model_kwargs"].update(enc_channels=[4, 6], enc_strides=[stride, 1],
                                       kernels=[3, 3], norm="none", dropout=0.0)
        if config.get("feature_modules"):
            config["feature_modules"]["encoder"].update(config["model_kwargs"])
        config["recon_kwargs"].update(norm="none", dropout=0.0)
    else:
        modules = config["lewm_modules"]
        modules.pop("backbone", None)
        modules["encoder"].update(enc_channels=[4, 6], enc_strides=[stride, 1],
                                   kernels=[3, 3], norm="none", dropout=0.0)
        if name == "lewm":
            modules["action"]["aux"].update(aux_channels=[4, 4], kernels=[3, 3],
                                               norm="none", dropout=0.0)
            modules["predictor"].update(stem_channels=4, neck_widths=[4, 6, 8],
                                         n_fft=8, hop_length=2, win_length=8, dropout=0.0)
        else:
            modules["action"].update(hidden_dims=[4, 4], dilation=[1, 2],
                                       kernels=3, norm="none", dropout=0.0)
            modules["predictor"].update(qk_channels=4, value_channels=4,
                                         num_heads=2, dropout=0.0)
        config["criterion_kwargs"].setdefault("sigreg_kwargs", {}).update(
            num_slices=4, freq_nodes=5)
    return config


def close(actual, expected, *, atol=1e-6, rtol=1e-5):
    torch.testing.assert_close(torch.as_tensor(actual), torch.as_tensor(expected),
                               atol=atol, rtol=rtol, check_dtype=False)


def output_config(config, directory):
    """Supply the same runtime paths create_config provides, without real results/."""
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    config=copy.deepcopy(config)
    config.update(version="acceptance",experiment_dir=str(directory),jepa_dir=str(directory),
                  jepa_checkpoint=str(directory/"checkpoint.pth.tar"),
                  jepa_model=str(directory/"model.pth.tar"),
                  jepa_model_best=str(directory/"model_best_eval.pth.tar"),
                  calibration_path=str(directory/"calibration.json"),
                  scores_path=str(directory/"scores.npz"),metrics_path=str(directory/"metrics.json"))
    return config


def format_fixtures(base=None):
    """Known train/validation distribution shift and separately labeled test data."""
    base=Path(base or tempfile.mkdtemp(prefix="lewm-conditional-formats-"))
    base.mkdir(parents=True,exist_ok=True)
    t=np.arange(160,dtype=np.float32)
    raw=np.stack([np.sin(t/7),np.cos(t/11),t/90],axis=1).astype(np.float32)
    raw[120:]+=12 # This must not influence optimization-row-fitted statistics.
    tt=np.arange(112,dtype=np.float32)
    test=np.stack([np.sin(tt/7),np.cos(tt/11),tt/90],axis=1).astype(np.float32)
    labels=np.zeros(112,dtype=np.int64)
    labels[38:50]=1; test[38:50,0]+=6
    labels[76:90]=1; test[76:90,1]-=4
    psm=base/"psm";psm.mkdir(exist_ok=True)
    for name,values in (("train",raw),("test",test)):
        frame=pd.DataFrame(values,columns=["sensor_a","sensor_b","sensor_c"])
        frame.insert(0,"timestamp_(min)",np.arange(len(frame)))
        frame.to_csv(psm/f"{name}.csv",index=False)
    pd.DataFrame({"timestamp_(min)":np.arange(len(labels)),"label":labels}).to_csv(
        psm/"test_label.csv",index=False)
    smd=base/"smd"
    for kind in ("train","test","test_label"):
        (smd/kind).mkdir(parents=True,exist_ok=True)
    for i in (1,2):
        np.savetxt(smd/"train"/f"machine-1-{i}.txt",raw*i+i,delimiter=",")
        np.savetxt(smd/"test"/f"machine-1-{i}.txt",test*i+i,delimiter=",")
        np.savetxt(smd/"test_label"/f"machine-1-{i}.txt",labels,fmt="%d")
    files={}
    for collection,channels in (("tsb-ad-u",1),("tsb-ad-m",3)):
        folder=base/collection;folder.mkdir(exist_ok=True)
        selected=[]
        for i in (1,2):
            filename=f"fixture_{i}_tr_160_1st_198.csv"
            values=np.concatenate([raw[:,:channels]*i,test[:,:channels]*i],axis=0)
            frame=pd.DataFrame(values,columns=[f"sensor_{j}" for j in range(channels)])
            frame["Label"]=np.concatenate([np.zeros(160,dtype=int),labels])
            frame.to_csv(folder/filename,index=False)
            selected.append(filename)
        pd.DataFrame({"file_name":selected,"description":["first","second"]}).to_csv(
            folder/"selection.csv",index=False)
        files[collection]=selected
    return {"base":base,"psm":psm,"smd":smd,"tsb-ad-u":base/"tsb-ad-u",
            "tsb-ad-m":base/"tsb-ad-m","files":files,"raw":raw,"test":test,"labels":labels}


def run_cases(cases, output=None):
    torch.set_num_threads(1)
    records = []
    for name, function in cases:
        torch.manual_seed(17)
        try:
            detail = function()
            records.append({"case":name, "status":"passed", "detail":detail})
            print(f"PASS {name}", flush=True)
        except Exception as exc:
            records.append({"case":name, "status":"failed", "error":str(exc),
                            "traceback":traceback.format_exc()})
            print(f"FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, indent=2, default=str) + "\n")
    return records
