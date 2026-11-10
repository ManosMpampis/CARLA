"""Independently verify trained v2/archived encoder reuse and source-free resume."""

import argparse
import copy
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

from conditional_support import ROOT, close, output_config, run_cases, tiny_config

import numpy as np
import torch

from data.jepa_dataset import JEPADataset
from models import get_model
from train import run
from utils.common_config import get_criterion
from utils.model_contract import initialize_feature_source, prepare_feature_source_config


def source_run(name,directory):
    config=output_config(tiny_config(name,stride=2),directory)
    config.update(epochs=2,eval_every=0)
    if name=="ae":
        config["model_kwargs"]["norm"]="batch"
        config["feature_modules"]["encoder"]["norm"]="batch"
    else:config["lewm_modules"]["encoder"]["norm"]="batch"
    run(config)
    path=Path(directory)/"best_validation_loss.pth.tar"
    checkpoint=torch.load(path,weights_only=False)
    model=get_model(checkpoint["config"],input_channels=3)
    model.load_state_dict(checkpoint["model"],strict=True)
    return path,checkpoint,model.eval()


def destination(name,source,role="backbone",freeze=True):
    config=tiny_config(name)
    option={"name":"pretrained","pretrained_from":str(source),"freeze":freeze}
    if name=="guided_forecast":
        config["lewm_modules"]["backbone"]=option
    elif role=="backbone":
        config["feature_modules"]={"backbone":option,"encoder":{"name":"identity"},
                                    "freeze_encoder":False}
    else:
        if not freeze:option.pop("freeze")
        config["feature_modules"]={"backbone":None,"encoder":option}
    return config


def tensor_state(module):
    return {key:value.detach().clone() for key,value in module.state_dict().items() if torch.is_tensor(value)}


def trained_transfer_routes():
    base=Path(tempfile.mkdtemp(prefix="lewm-transfer-acceptance-"))
    sources={name:source_run(name,base/f"source-{name}") for name in ("ae","lewm")}
    x=torch.randn(4,3,48)
    details={}
    routes=[("ae","guided_forecast","backbone",True),
            ("lewm","guided_forecast","backbone",True),
            ("lewm","ae","backbone",True),
            ("lewm","ae","encoder",False),
            ("lewm","ae","encoder",True)]
    for source_name,name,role,freeze in routes:
        source_path,source_checkpoint,source_model=sources[source_name]
        schema=source_checkpoint["feature_schema"]
        config=destination(name,source_path,role,freeze)
        prepared=prepare_feature_source_config(config,schema)
        model=get_model(prepared,input_channels=3)
        initialize_feature_source(model,prepared,schema)
        transferred=getattr(model,role)
        transferred.eval()
        close(transferred(x),source_model.encode(x),atol=0,rtol=0)
        assert transferred.total_stride==2
        if name=="ae":
            assert model.total_stride==2 and model(x)["recon"].shape==x.shape
            if role=="backbone":assert isinstance(model.encoder,torch.nn.Identity)
        before=tensor_state(transferred)
        model.train()
        assert transferred.training is (not freeze)
        optimizer=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=.003)
        output=model(x)
        loss=get_criterion(config)(output)["loss"]
        loss.backward()
        downstream=model.predictor if name=="guided_forecast" else model.decoder
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in downstream.parameters())
        if freeze:
            assert all(p.grad is None for p in transferred.parameters())
        else:
            assert any(p.grad is not None and p.grad.abs().sum()>0 for p in transferred.parameters())
        optimizer.step()
        after=tensor_state(transferred)
        changed=any(not torch.equal(before[key],after[key]) for key in before)
        assert changed is (not freeze),"freeze must preserve weights AND BatchNorm buffers"
        key=f"{source_name}-to-{name}-{role}-{'frozen' if freeze else 'trainable'}"
        # Exercise normal entry, validation/scoring and complete destination snapshots.
        runtime=output_config(config,base/key)
        runtime.update(epochs=2,eval_every=1)
        run(runtime)
        saved=torch.load(Path(runtime["jepa_dir"])/"checkpoint.pth.tar",weights_only=False)
        assert saved["config"].get("_feature_transfer")
        details[key]={"directory":runtime["jepa_dir"],"source":str(source_path),
                      "initial_feature_error":0.,"source_changed_after_step":changed,
                      "frozen":freeze,"effective_stride":model.total_stride}
    # Continue a destination after making the original source unavailable.
    source_path=sources["lewm"][0]
    config=destination("guided_forecast",source_path)
    key="lewm-to-guided_forecast-backbone-frozen"
    runtime=output_config(config,base/key)
    source_hidden=source_path.with_suffix(".temporarily-unavailable")
    source_path.rename(source_hidden)
    try:
        runtime.update(epochs=3,eval_every=1)
        run(runtime)
        report=run({**runtime,"stage":"score"})
    finally:source_hidden.rename(source_path)
    resumed=torch.load(Path(runtime["jepa_dir"])/"checkpoint.pth.tar",weights_only=False)
    assert resumed["epoch"]==3
    return {"base":str(base),"routes":details,"source_free_resume_epoch":3,
            "source_free_score_report":report}


def metadata_rejections():
    base=Path(tempfile.mkdtemp(prefix="lewm-transfer-reject-"))
    source_path,checkpoint,source_model=source_run("lewm",base/"source")
    schema=checkpoint["feature_schema"]
    config=destination("guided_forecast",source_path)
    for mismatch in ("schema","stride","activation"):
        changed=copy.deepcopy(config)
        changed_schema=list(reversed(schema)) if mismatch=="schema" else schema
        if mismatch=="stride":changed["lewm_modules"]["backbone"]["enc_strides"]=[1,1]
        if mismatch=="activation":changed["lewm_modules"]["backbone"]["output_act"]="gelu"
        try:prepare_feature_source_config(changed,changed_schema)
        except ValueError:pass
        else:raise AssertionError(f"incompatible source {mismatch} was accepted")
    prepared=prepare_feature_source_config(config,schema)
    model=get_model(prepared,input_channels=3);initialize_feature_source(model,prepared,schema)
    changed=copy.deepcopy(prepared)
    changed["lewm_modules"]["engine"]["context_ratio"]=1
    other=get_model(changed,input_channels=3)
    try:other.load_state_dict(model.state_dict(),strict=True)
    except ValueError:pass
    else:raise AssertionError("same-shape different forecast task resumed silently")
    return {"source":str(source_path),"rejections":["schema/order","stride","activation","task ratio"]}


def archived_sources():
    base=Path(tempfile.mkdtemp(prefix="lewm-archived-source-"))
    archive=subprocess.check_output(["git","archive","--format=tar",
                                      "22dc54c1a8ac573c198e2517c46ba8915d33c9c7",
                                      "models","losses","utils","configs","data","metrics"],cwd=ROOT)
    source_tree=base/"baseline";source_tree.mkdir()
    tarfile.open(fileobj=io.BytesIO(archive)).extractall(source_tree,filter="data")
    configs={}
    for name in ("ae","lewm"):
        config=tiny_config(name,stride=2)
        config.setdefault("model_kwargs",{}).pop("latent_output",None)
        config.get("lewm_modules",{}).get("encoder",{}).pop("latent_output",None)
        # Baseline builder's AE path uses model_kwargs, predating feature_modules.
        config.pop("feature_modules",None)
        config["model_kwargs"]["in_channels"]=3
        configs[name]=config
    config_path=base/"source-configs.json";config_path.write_text(json.dumps(configs))
    program='''import json,sys,torch,numpy as np
from models import get_model
from utils.common_config import get_criterion
from data.jepa_dataset import make_synthetic_series
torch.set_num_threads(1);torch.manual_seed(39)
configs=json.load(open(sys.argv[1]));folder=sys.argv[2]
for name,config in configs.items():
 model=get_model(config,input_channels=3).train()
 raw,_=make_synthetic_series(176,3,17);mean=raw.astype(np.float64).mean(0);std=raw.astype(np.float64).std(0)
 scaled=((raw-mean)/std).astype(np.float32)
 x=torch.from_numpy(np.stack([scaled[i:i+48].T for i in (0,8,16,24)]));optimizer=torch.optim.Adam(model.parameters(),lr=.003)
 loss=get_criterion(config)(model(x))["loss"];loss.backward();optimizer.step();model.eval()
 probe=torch.randn(2,3,48)
 with torch.no_grad():features=model.encoder(probe)
 torch.save({"model":model.state_dict(),"config":config,"epoch":1,"selection":{"source":"validation","metric":"loss","value":float(loss.detach())},"inference":{"normalization_mean":mean.tolist(),"normalization_std":std.tolist()}},folder+"/"+name+".pth.tar")
 torch.save({"input":probe,"features":features},folder+"/"+name+"-expected.pth")
 print(name,float(loss.detach()))
'''
    process=subprocess.run([str(ROOT/"venv/bin/python"),"-c",program,str(config_path),str(base)],
                            cwd=source_tree,text=True,capture_output=True)
    assert process.returncode==0,process.stdout+process.stderr
    details={}
    for name in ("ae","lewm"):
        source=base/f"{name}.pth.tar"
        expected=torch.load(base/f"{name}-expected.pth",weights_only=False)
        config=destination("guided_forecast",source)
        config["_preprocessing"]=JEPADataset(config,train=True).preprocessing
        prepared=prepare_feature_source_config(config,["channel_0","channel_1","channel_2"])
        model=get_model(prepared,input_channels=3)
        initialize_feature_source(model,prepared,["channel_0","channel_1","channel_2"])
        model.eval()
        actual=model.backbone(expected["input"])
        close(actual,expected["features"],atol=0,rtol=0)
        provenance=prepared["_feature_transfer"]["backbone"]
        assert provenance["normalization_changed"] is True
        assert provenance["destination_preprocessing"]==config["_preprocessing"]
        assert not np.allclose(provenance["preprocessing"]["normalization_mean"],
                              config["_preprocessing"]["mean"])
        details[name]={"checkpoint":str(source),"feature_error":float((actual-expected["features"]).abs().max()),
                       "provenance":provenance}
    return {"base":str(base),"baseline_training_log":process.stdout,"sources":details}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output");parser.add_argument("--case",action="append")
    args=parser.parse_args()
    cases=[("trained transfer routes",trained_transfer_routes),("metadata rejections",metadata_rejections),
           ("archived source feature equality",archived_sources)]
    if args.case:cases=[item for item in cases if item[0] in args.case]
    assert cases,"no matching cases"
    rows=run_cases(cases,args.output)
    raise SystemExit(any(row["status"]!="passed" for row in rows))
