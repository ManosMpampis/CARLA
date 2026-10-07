"""Run original LEWM once per machine, then train/score the requested arms."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import yaml

from lewm_cross_attention import main as run_stage
from utils.config import load_experiment_config, model_path as saved_model_path


CONFIG_DIR = Path(__file__).resolve().parent / "configs/lewm_encoder"


def model_path(config, env, version, machine):
    with open(env) as stream:
        root = yaml.safe_load(stream)["root_dir"]
    return saved_model_path(root, config, machine, version)


def run_machine(machine, args):
    encoder_name = args.phase1_experiment
    phase1_path = args.phase1_config or str(CONFIG_DIR / encoder_name / "phase1.yml")
    phase1 = load_experiment_config(phase1_path)
    encoder_name = phase1.get("phase1_experiment", encoder_name)
    phase1.update(framework="lewm_encoder", phase1_experiment=encoder_name,
                  experiment_name=encoder_name)
    phase1_version = args.phase1_version or args.version
    pretrained = model_path(phase1, args.config_env, phase1_version, machine)
    stages = [(phase1_path, phase1_version, {"stage": "pretrain", "framework": "lewm_encoder",
                                           "phase1_experiment": encoder_name,
                                           "experiment_name": encoder_name})]
    if args.phase1_epochs is not None:
        stages[0][2]["epochs"] = args.phase1_epochs
    sources = ("observed", "context") if args.query_source == "both" else (args.query_source,)
    for source in sources:
        for arm in args.arms:
            config_path = str(CONFIG_DIR / "time/cross_attention" / f"{source}_{arm}.yml")
            config = load_experiment_config(config_path)
            # A user-selected phase-one encoder must be reused exactly.
            patch = {"model_kwargs": dict(phase1["model_kwargs"]),
                     "framework": "cross_attention", "phase1_experiment": encoder_name,
                     "phase1_version": phase1_version, "tag_phase1": phase1.get("tag_jepa"),
                     "pretrained_from": pretrained}
            attention = dict(config["cross_attention_kwargs"])
            # Preserve the 64+64 INPUT geometry when the trunk downsamples.
            stride = 1
            for value in patch["model_kwargs"].get("enc_strides", [1]):
                stride *= value
            if 64 % stride:
                raise ValueError("sweep requires encoder stride to divide the 64-step crops")
            attention.update(context_crop=[0, 64 // stride],
                             target_crop=[64 // stride, 128 // stride])
            for key in ("qk_channels", "value_channels", "num_heads"):
                value = getattr(args, key)
                if value is not None:
                    attention[key] = value
            patch["cross_attention_kwargs"] = attention
            train_patch = {**patch, "stage": "phase2"}
            if args.phase2_epochs is not None:
                train_patch["epochs"] = args.phase2_epochs
            stages.append((config_path, args.version, train_patch))
            stages.append((config_path, args.version,
                           {**patch, "stage": "score",
                            "score_checkpoint": model_path({**config, **patch}, args.config_env, args.version, machine)}))
    for config_path, version, patch in stages:
        print(f"{'DRY ' if args.dry_run else ''}{machine}: {patch['stage']} "
              f"{Path(config_path).name} version={version}", flush=True)
        if args.dry_run:
            if patch.get("pretrained_from"):
                print(f"  pretrained_from={patch['pretrained_from']}")
            if patch.get("score_checkpoint"):
                print(f"  score_checkpoint={patch['score_checkpoint']}")
            continue
        run_stage(SimpleNamespace(config_env=args.config_env, config_exp=config_path,
                                  fname=machine, version=version), patch)


def cli():
    parser = argparse.ArgumentParser(description="LEWM -> crop cross attention -> score SMD arms")
    parser.add_argument("--config-env", default="configs/env.yml")
    parser.add_argument("--phase1-config", help="optional existing LEWM phase-one config")
    parser.add_argument("--phase1-experiment", default="time", help="shared encoder experiment name")
    parser.add_argument("--phase1-version", help="reuse a shared encoder run (defaults to --version)")
    parser.add_argument("--version", default="smd_v1")
    parser.add_argument("--fname", default="machine-1-1.txt")
    parser.add_argument("--all", action="store_true", help="run the chain separately for every SMD machine")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--arms", nargs="+", choices=("features", "input", "query"),
                        default=["features", "input", "query"])
    parser.add_argument("--query-source", choices=("observed", "context", "both"), default="observed")
    parser.add_argument("--phase1-epochs", type=int)
    parser.add_argument("--phase2-epochs", type=int)
    parser.add_argument("--qk-channels", type=int)
    parser.add_argument("--value-channels", type=int)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    machines = sorted(p.name for p in Path("datasets/SMD/train").glob("machine-*.txt")) if args.all else [args.fname]
    if args.limit > 0:
        machines = machines[:args.limit]
    if not machines:
        raise ValueError("no SMD machines found")
    for machine in machines:
        run_machine(machine, args)


if __name__ == "__main__":
    cli()
