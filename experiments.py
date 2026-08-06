# %%
import os
from easydict import EasyDict
from pathlib import Path

from carla_pretext import main as main_pretext
from carla_classification import main as main_classification
from evaluation import main as main_evaluation

env = os.environ.copy()
env['PYTHONPATH'] = '/home/manos/Documents/EKETA/HYPER_AI/gits/official_carla/unchanged/CARLA/'
env['PATH'] = f'/usr/local/cuda/bin:{env.get("PATH", "")}'
env['LD_LIBRARY_PATH'] = f'/usr/local/cuda/lib64:{env.get("LD_LIBRARY_PATH", "")}'

all_files = os.listdir(os.path.join('datasets/', 'SMD/train'))
file_list = [file for file in all_files if file.startswith('machine-')]
file_list = sorted(file_list)
print(file_list)

experiment_dir = Path("configs/pretext/new_loss/smd/")
experiment_group = [exp for exp in experiment_dir.iterdir() if (exp.is_dir() and "normalization-strategy" not in str(exp))]
experiments = [[exp for exp in experiment.iterdir()] for experiment in experiment_group]
# %% Test
# version = "test/test1"
# index = file_list.index('machine-1-1.txt')
# for filename in file_list[index:]:
#     print(filename)

#     # Run the pretext script
#     pretext_args = EasyDict({"config_env": "configs/env.yml",
#                     "config_exp": "configs/pretext/new_loss/smd/orig_re_weight_and_clamp_neg_re_weight/original.yml",
#                     "fname": filename,
#                     "version": f"{version}"})
#     main_pretext(pretext_args)
# %% Pretext experiments

version_batch = "batch/"
version_layer = "layer/"
version_instance = "instance/"
version_no_norm = "no_norm/"

# %% Pretext original-dynamic_weight-loss_clamp
# for exp in experiments[0]:
#     version = f"{version_layer}original-dynamic_weight-loss_clamp/{exp.name[:-4]}"
#     index = file_list.index('machine-1-1.txt')
#     index_scip = file_list.index('machine-3-6.txt')
#     if "original.yml" not in str(exp):
#         for filename in file_list[index:]:
#             if exp.name[:-4] == "clamp_only_negative_loss" and filename in file_list[:index_scip]:
#                 continue
#             print(filename)

#             # Run the pretext script
#             pretext_args = EasyDict({"config_env": "configs/env.yml",
#                             "config_exp": str(exp),
#                             "fname": filename,
#                             "version": f"{version}"})
#             main_pretext(pretext_args)

# %% Pretext original-dynamic_weight-loss_clamp
# for exp in experiments[0]:
#     version = f"batch/original-dynamic_weight-loss_clamp/{exp.name[:-4]}"
#     index = file_list.index('machine-1-1.txt')
#     index_scip = file_list.index('machine-3-6.txt')
#     if "original.yml" not in str(exp):
#         for filename in file_list[index:]:
#             if exp.name[:-4] == "clamp_only_negative_loss" and filename in file_list[:index_scip]:
#                 continue
#             print(filename)

#             # Run the pretext script
#             pretext_args = EasyDict({"config_env": "configs/env.yml",
#                             "config_exp": str(exp),
#                             "fname": filename,
#                             "version": f"{version}"})
#             main_pretext(pretext_args)

# %% Pretext ema_loss
# experiment_to_go = Path(f"{experiment_dir}/ema_loss/dynamic_margin_by_ema_loss-dynamic_loss_guidance-clamp_only_negative_loss.yml")
# exp_index = experiments[1].index(experiment_to_go)
# for exp in experiments[1][exp_index:]:
#     version = f"batch/ema_loss/{exp.name[:-4]}"
#     index = file_list.index('machine-1-1.txt')
#     index_scip = file_list.index('machine-1-1.txt')
#     for filename in file_list[index:]:
#         if (exp == experiment_to_go) and (filename in file_list[:index_scip]):
#             continue
#         print(filename)

#         # Run the pretext script
#         pretext_args = EasyDict({"config_env": "configs/env.yml",
#                         "config_exp": str(exp),
#                         "fname": filename,
#                         "version": f"{version}"})
#         main_pretext(pretext_args)

# %% Pretext dynamic_reweight_on_distance
# experiment_to_go = Path(f"{experiment_dir}/dynamic_reweight_on_distance/dynamic_margin_by_neg_distance-clamp_only_negative_loss-dynamic_weight.yml")
# exp_index = experiments[2].index(experiment_to_go)
# for exp in experiments[2][exp_index:]:
#     version = f"batch/dynamic_reweight_on_distance/{exp.name[:-4]}"
#     index = file_list.index('machine-1-1.txt')
#     index_scip = file_list.index('machine-1-1.txt')
#     for filename in file_list[index:]:
#         if (exp == experiment_to_go) and (filename in file_list[:index_scip]):
#             continue
#         print(filename)
#         # Run the pretext script
#         pretext_args = EasyDict({"config_env": "configs/env.yml",
#                         "config_exp": str(exp),
#                         "fname": filename,
#                         "version": f"{version}"})
#         main_pretext(pretext_args)

# %% Pretext normalization
# experiment_to_go = Path(f"{experiment_dir}/dynamic_reweight_on_distance/dynamic_margin_by_neg_distance-clamp_only_negative_loss-dynamic_weight.yml")
# exp_index = 0# experiments[2].index(experiment_to_go)
# experiments = [exp for exp in Path(os.path.join(experiment_dir, "normalization-strategy")).iterdir()]
# for exp in experiments[exp_index:]:
#     for norm in ["batch", "instance"]:
#         version = f"./best_models/{norm}/{exp.name[:-4]}"
#         index = file_list.index('machine-1-1.txt')
#         index_scip = file_list.index('machine-1-1.txt')
#         for filename in file_list[index:]:
#             if (exp == experiment_to_go) and (filename in file_list[:index_scip]):
#                 continue
#             print(filename)
#             # Run the pretext script
#             patch = EasyDict({"res_kwargs": {
#                 "in_channels": 38,
#                 "mid_channels": [4, 8],
#                 "kernel_sizes": [8, 5, 3],
#                 "norm_layer_name":  norm,
#                 "window_size": 256,
#                 "dropout": True
#                 }
#             })
#             pretext_args = EasyDict({"config_env": "configs/env.yml",
#                             "config_exp": str(exp),
#                             "fname": filename,
#                             "version": f"{version}"})
#             main_pretext(pretext_args, update_dictionary=patch)

# index = file_list.index('machine-1-1.txt')
# for filename in file_list[index:]:
#     # Run the classification script
#     classification_args = EasyDict({"config_env": "configs/env.yml",
#                     "config_exp": "configs/classification/batch_common.yml",
#                     "fname": filename,
#                     "version": f"{version_batch}"})
#     main_classification(classification_args)


# %% Pretext normalization
experiment_dir = Path("configs/classification/experiments/")
# experiment_group = [exp for exp in experiment_dir.iterdir() if (exp.is_dir() and "normalization-strategy" not in str(exp))]
experiments = [experiment for experiment in experiment_dir.iterdir()]

pretext_model = "dynamic_margin_by_neg_distance-dynamic_loss_guidance-clamp_only_negative_loss-dynamic_weight_loss" #"original" #"dynamic_margin_by_neg_distance-dynamic_loss_guidance-clamp_only_negative_loss-dynamic_weight_loss"
experiment_to_go = Path(f"{experiment_dir}/dynamic_reweight_on_distance/dynamic_margin_by_neg_distance-clamp_only_negative_loss-dynamic_weight.yml")
exp_index = 0# experiments[2].index(experiment_to_go)
for exp in experiments[exp_index:]:
    for norm in ["batch", "instance"]:
        version = f"./best_models/{norm}/{pretext_model}"
        index = file_list.index('machine-1-1.txt')
        index_scip = file_list.index('machine-1-1.txt')
        for filename in file_list[index:]:
            if (exp == experiment_to_go) and (filename in file_list[:index_scip]):
                continue
            print(filename)
            # Run the pretext script
            patch = EasyDict({"res_kwargs": {
                "in_channels": 38,
                "mid_channels": [4, 8],
                "kernel_sizes": [8, 5, 3],
                "norm_layer_name":  norm,
                "window_size": 256,
                "dropout": True
                },
                "epochs": 1000,
            })
            classification_args = EasyDict({"config_env": "configs/env.yml",
                            "config_exp": str(exp),
                            "fname": filename,
                            "version": f"{version}"})
            main_classification(classification_args, update_dictionary=patch)
# %%
