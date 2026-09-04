import sys
from easydict import EasyDict
from carla_pretext import main as main_pretext
from carla_classification import main as main_classification

fname = sys.argv[1] if len(sys.argv) > 1 else None

VERSION = "./best_models/batch/avgmax-dynamic_margin_by_neg_distance-dynamic_loss_guidance-clamp_only_negative_loss-dynamic_weight_loss"
PRETEXT_CONFIG = "configs/pretext/new_loss/smd/dynamic_reweight_on_distance/dynamic_margin_by_neg_distance-dynamic_loss_guidance-clamp_only_negative_loss-dynamic_weight_loss.yml"
CLASSIFICATION_CONFIG = "configs/classification/two_phase_avgmax.yml"

RES_KWARGS = {
    "in_channels": 38,
    "mid_channels": [4, 8],
    "kernel_sizes": [8, 5, 3],
    "norm_layer_name": "batch",
    "window_size": 256,
    "dropout": True,
}

pretext_patch = EasyDict({
    "res_kwargs": dict(RES_KWARGS),
    "pooling": "avgmax",
})

classification_patch = EasyDict({
    "res_kwargs": dict(RES_KWARGS),
    "pooling": "avgmax",
    "epochs": 1000,
    "update_data": 0,
    "injection_strategy": "original",
})

pretext_args = EasyDict({
    "config_env": "configs/env.yml",
    "config_exp": PRETEXT_CONFIG,
    "fname": fname,
    "version": VERSION,
})
classification_args = EasyDict({
    "config_env": "configs/env.yml",
    "config_exp": CLASSIFICATION_CONFIG,
    "fname": fname,
    "version": VERSION,
})

if __name__ == "__main__":
    assert fname is not None, "usage: python run_avgmax_experiment.py <machine-file>"
    print(f"[{fname}] === Pretext stage (2000 epochs, avgmax pooling) ===", flush=True)
    main_pretext(pretext_args, update_dictionary=pretext_patch)
    print(f"[{fname}] === Classification stage (1000 epochs, two-phase auto, avgmax pooling) ===", flush=True)
    main_classification(classification_args, update_dictionary=classification_patch)
