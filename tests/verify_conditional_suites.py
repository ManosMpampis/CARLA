"""Exercise generated PSM/SMD/TSB selections through the real job planner/CLI."""

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile

from conditional_support import ROOT, close, format_fixtures, run_cases, tiny_config

import numpy as np
import pandas as pd
import torch
import yaml

from utils.jobs import plan_jobs, run_jobs


def inputs():
    fixtures=format_fixtures()
    config=tiny_config("ae",window=24)
    config.update(train_db_name="smd",val_db_name="smd",fname="machine-1-1.txt",
                  val_fraction=.25,epochs=1,cpu_threads=1)
    config_file=fixtures["base"]/"ae.yml"
    config_file.write_text(yaml.safe_dump(config))
    env=fixtures["base"]/"env.yml"
    env.write_text(yaml.safe_dump({"root_dir":str(fixtures["base"]/"results")}))
    return fixtures,config_file,env


def selection_matrix_and_preflight():
    fixtures,config,env=inputs()
    cases=[("psm",None,None,1),("smd","machine-1-1.txt",None,1),
           ("smd","all",None,2),("smd","all-one",None,1)]
    for collection in ("tsb-ad-u","tsb-ad-m"):
        cases.extend([(collection,fixtures["files"][collection][0],None,1),
                      (collection,None,fixtures[collection]/"selection.csv",2)])
    evidence=[]
    for collection,fname,file_list,count in cases:
        jobs=plan_jobs(config,env,dataset=collection,dataset_root=fixtures[collection],
                       fname=fname,file_list=file_list,version="selection")
        assert len(jobs)==count
        assert len({p["jepa_dir"] for p in jobs})==count
        assert all(p["dataset_collection"]==collection for p in jobs)
        if collection=="psm":assert jobs[0]["fname"]=="psm"
        if fname=="all":assert [p["fname"] for p in jobs]==["machine-1-1.txt","machine-1-2.txt"]
        if fname=="all-one":assert jobs[0]["fname"]=="all-one" and len(jobs[0]["machine_files"])==2
        if file_list:assert [p["fname"] for p in jobs]==fixtures["files"][collection]
        if collection.startswith("tsb"):
            assert all(collection in str(p["jepa_dir"]) for p in jobs)
            assert all(p["tsb_evaluation_scope"]=="test" for p in jobs)
            expected_channels=1 if collection=="tsb-ad-u" else 3
            assert all(len(p["_feature_schema"])==expected_channels for p in jobs)
        evidence.append({"collection":collection,"selection":str(file_list or fname or "default"),
                         "jobs":[{"fname":p["fname"],"output":p["jepa_dir"]} for p in jobs]})
    # Invalid later list entries must fail before the first training checkpoint.
    list_path=fixtures["base"]/"invalid.csv"
    invalid=[pd.DataFrame({"filename":["bad"]}),
             pd.DataFrame({"file_name":[fixtures["files"]["tsb-ad-u"][0]]*2}),
             pd.DataFrame({"file_name":[fixtures["files"]["tsb-ad-u"][0],"missing.csv"]}),
             pd.DataFrame({"file_name":[""]})]
    for frame in invalid:
        frame.to_csv(list_path,index=False)
        try:
            plan_jobs(config,env,dataset="tsb-ad-u",dataset_root=fixtures["tsb-ad-u"],file_list=list_path)
        except (ValueError,FileNotFoundError):pass
        else:raise AssertionError("invalid manifest passed preflight")
    assert not list(fixtures["base"].rglob("checkpoint.pth.tar"))
    # Custom series can declare a prefix explicitly; unrelated CSV metadata is accepted.
    original=fixtures["tsb-ad-u"]/fixtures["files"]["tsb-ad-u"][0]
    custom=fixtures["tsb-ad-u"]/"custom.csv";custom.write_bytes(original.read_bytes())
    pd.DataFrame({"file_name":[custom.name],"train_end":[160],"note":["explicit"]}).to_csv(list_path,index=False)
    jobs=plan_jobs(config,env,dataset="tsb-ad-u",dataset_root=fixtures["tsb-ad-u"],file_list=list_path)
    assert jobs[0]["tsb_train_end"]==160
    return {"fixtures":str(fixtures["base"]),"matrix":evidence,"invalid_lists_rejected":len(invalid)}


def parallel_failures_and_resume():
    fixtures,config,env=inputs()
    jobs=plan_jobs(config,env,dataset="smd",dataset_root=fixtures["smd"],fname="all",version="parallel")
    victim=fixtures["smd"]/"train"/"machine-1-1.txt"
    hidden=victim.with_suffix(".unavailable")
    victim.rename(hidden)
    try:
        failed=run_jobs(jobs,jobs=2)
    finally:
        hidden.rename(victim)
    by_name={row["fname"]:row for row in failed["jobs"]}
    assert by_name["machine-1-1.txt"]["status"]=="failed"
    assert by_name["machine-1-2.txt"]["status"]=="completed"
    pids={row["pid"] for row in failed["jobs"]}
    assert os.getpid() not in pids and 1<=len(pids)<=2
    complete_checkpoint=Path(jobs[1]["jepa_checkpoint"])
    timestamp=complete_checkpoint.stat().st_mtime_ns
    repaired=run_jobs(jobs,jobs=2)
    assert all(row["status"]=="completed" for row in repaired["jobs"])
    assert repaired["jobs"][1]["skipped_completed"]
    assert complete_checkpoint.stat().st_mtime_ns==timestamp
    # Same seeded job in an independent sequential run should reach identical weights.
    sequential=plan_jobs(config,env,dataset="smd",dataset_root=fixtures["smd"],
                         fname="machine-1-2.txt",version="sequential")
    single=run_jobs(sequential,jobs=1)
    assert single["jobs"][0]["status"]=="completed"
    first=torch.load(complete_checkpoint,weights_only=False)["model"]
    second=torch.load(sequential[0]["jepa_checkpoint"],weights_only=False)["model"]
    for key,value in first.items():
        if torch.is_tensor(value):close(value,second[key],atol=0,rtol=0)
    return {"base":str(fixtures["base"]),"failed_then_repaired":failed,"resumed":repaired,
            "worker_pids":sorted(pids),"sequential_parallel_weights_equal":True}


def pooled_and_tsb_execution():
    fixtures,config,env=inputs()
    evidence={}
    for collection,selection in (("smd","all-one"),("tsb-ad-u","list"),("tsb-ad-m","list")):
        args={"fname":selection} if selection!="list" else {"file_list":fixtures[collection]/"selection.csv"}
        jobs=plan_jobs(config,env,dataset=collection,dataset_root=fixtures[collection],
                       version="formats",**args)
        manifest=run_jobs(jobs,jobs=1)
        assert all(row["status"]=="completed" for row in manifest["jobs"])
        reports=[json.loads(Path(p["metrics_path"]).read_text()) for p in jobs]
        if collection=="smd":
            report=reports[0]
            assert report["machine_count"]==2 and report["aggregation"]=="macro_per_machine"
            assert set(report["machines"])=={"machine-1-1.txt","machine-1-2.txt"}
            thresholds=[]
            for name,item in report["machines"].items():
                assert (Path(jobs[0]["jepa_dir"])/"machines"/name/"metrics.json").is_file()
                thresholds.append(item["evaluation"]["timeseries"]["calibrated"]["threshold"])
            expected=np.mean([item["evaluation"]["timeseries"]["calibrated"]["f1"] for item in report["machines"].values()])
            close(report["evaluation"]["timeseries"]["calibrated"]["f1"],expected)
            calibration=json.loads(Path(jobs[0]["calibration_path"]).read_text())
            assert set(calibration["machines"])==set(report["machines"])
            diagnostic=torch.load(Path(jobs[0]["jepa_dir"])/"best_test_timeseries_oracle_f1.pth.tar",weights_only=False)
            assert diagnostic["selection"]["threshold"] is None
            assert diagnostic["selection"]["threshold_scope"]=="per_machine"
            assert set(diagnostic["evaluation_detail"]["machines"])==set(report["machines"])
        else:
            assert all(item["coverage"]["series_start"]==160 for item in reports)
            assert all(item["coverage"]["total_timesteps"]==112 for item in reports)
            assert all(item["collection"]==collection for item in reports)
        evidence[collection]={"manifest":manifest,"reports":reports}
    return {"base":str(fixtures["base"]),"runs":evidence}


def cli_train_then_score():
    fixtures,config,env=inputs()
    command=[str(ROOT/"venv/bin/python"),str(ROOT/"train.py"),"--config_exp",str(config),
             "--config_env",str(env),"--dataset","psm","--dataset-root",str(fixtures["psm"]),
             "--version","cli","--jobs","1"]
    result=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr
    planned=plan_jobs(config,env,dataset="psm",dataset_root=fixtures["psm"],version="cli")
    path=Path(planned[0]["metrics_path"])
    path.rename(path.with_name("before_score_metrics.json"))
    scored=subprocess.run(command+["--score"],cwd=ROOT,text=True,capture_output=True)
    assert scored.returncode==0,scored.stdout+scored.stderr
    assert path.is_file(),"score CLI must run after a completed training suite"
    old=json.loads(path.read_text())
    assert old["collection"]=="psm"
    first_scores=np.load(path.parent/"scores.npz")["timeseries_scores"].copy()
    # A distinct compatible checkpoint must actually be used by --score_checkpoint.
    override=torch.load(path.parent/"best_validation_loss.pth.tar",weights_only=False)
    override["model"]["decoder.to_input.bias"]+=5
    override_path=path.parent/"diagnostic_override.pth.tar"
    torch.save(override,override_path)
    specific=subprocess.run(command+["--score","--score_checkpoint",str(override_path)],
                             cwd=ROOT,text=True,capture_output=True)
    assert specific.returncode==0,specific.stdout+specific.stderr
    changed_scores=np.load(path.parent/"scores.npz")["timeseries_scores"]
    assert not np.allclose(first_scores,changed_scores),"explicit score checkpoint must change scores"
    # Missing required artifacts must never be accepted as a complete skipped run.
    path.rename(path.with_name("before_artifact_repair_metrics.json"))
    repaired=run_jobs(planned,jobs=1)
    assert path.is_file(),"completed manifest must repair or rerun when output artifact is missing"
    assert repaired["jobs"][0]["status"]=="completed"
    (fixtures["base"]/"cli_train.log").write_text(result.stdout+result.stderr)
    (fixtures["base"]/"cli_score.log").write_text(scored.stdout+scored.stderr)
    return {"base":str(fixtures["base"]),"command":command,"score_report":old}


def matched_source_selection():
    fixtures,_,env=inputs()
    source=tiny_config("lewm",window=24)
    source.update(train_db_name="smd",val_db_name="smd",fname="machine-1-1.txt",
                  val_fraction=.25,epochs=1,eval_every=0)
    source_config=fixtures["base"]/"masked-source.yml"
    source_config.write_text(yaml.safe_dump(source))
    jobs=plan_jobs(source_config,env,dataset="smd",dataset_root=fixtures["smd"],fname="all",version="source")
    assert all(row["status"]=="completed" for row in run_jobs(jobs,jobs=1)["jobs"])
    guided=tiny_config("guided_forecast",window=24)
    guided.update(train_db_name="smd",val_db_name="smd",fname="machine-1-1.txt",epochs=1,eval_every=0)
    guided["lewm_modules"]["backbone"]={"name":"pretrained","source":{
        "framework":"lewm","experiment":"masked_frequency","version":"source"}}
    destination=fixtures["base"]/"guided.yml";destination.write_text(yaml.safe_dump(guided))
    destinations=plan_jobs(destination,env,dataset="smd",dataset_root=fixtures["smd"],fname="all",version="destination")
    selected=[]
    for job in destinations:
        path=Path(job["lewm_modules"]["backbone"]["pretrained_from"])
        assert path.name=="best_validation_loss.pth.tar" and job["fname"] in path.parts
        selected.append(str(path))
    # A matching source without legal validation resolves to its best-training snapshot.
    source["wsz"]=96
    source_config.write_text(yaml.safe_dump(source))
    short=plan_jobs(source_config,env,dataset="smd",dataset_root=fixtures["smd"],
                    fname="machine-1-1.txt",version="source-short")
    assert run_jobs(short,jobs=1)["jobs"][0]["status"]=="completed"
    guided["lewm_modules"]["backbone"]["source"]["version"]="source-short"
    destination.write_text(yaml.safe_dump(guided))
    fallback=plan_jobs(destination,env,dataset="smd",dataset_root=fixtures["smd"],
                       fname="machine-1-1.txt",version="destination-short")
    fallback_path=fallback[0]["lewm_modules"]["backbone"]["pretrained_from"]
    assert Path(fallback_path).name=="best_train_loss.pth.tar"
    return {"base":str(fixtures["base"]),"matched_validation_sources":selected,
            "short_training_fallback":fallback_path}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--case",action="append")
    args=parser.parse_args()
    cases=[("selection matrix and preflight",selection_matrix_and_preflight),
           ("parallel failures and resume",parallel_failures_and_resume),
           ("pooled and TSB execution",pooled_and_tsb_execution),
           ("CLI train then score",cli_train_then_score),("matched source selection",matched_source_selection)]
    if args.case:cases=[item for item in cases if item[0] in args.case]
    assert cases,"no matching cases"
    records=run_cases(cases,args.output)
    raise SystemExit(any(row["status"]!="passed" for row in records))
