# %%
import os
from easydict import EasyDict
from pathlib import Path

from carla_lewm_true import main as lewm
from carla_steered import main as steered

env = os.environ.copy()
env['PYTHONPATH'] = '/home/manos/Documents/EKETA/HYPER_AI/gits/official_carla/unchanged/CARLA/'
env['PATH'] = f'/usr/local/cuda/bin:{env.get("PATH", "")}'
env['LD_LIBRARY_PATH'] = f'/usr/local/cuda/lib64:{env.get("LD_LIBRARY_PATH", "")}'

all_files = os.listdir(os.path.join('datasets/', 'SMD/train'))
file_list = [file for file in all_files if file.startswith('machine-')]
file_list = sorted(file_list)
print(file_list)

experiment_dir = Path("configs/jepa/lewm/")
experiment_group = [exp for exp in experiment_dir.iterdir() if (exp.is_dir() and "normalization-strategy" not in str(exp))]
experiments = [exp for exp in experiment_dir.iterdir()]

# %% Pretext normalization

# experiment_to_go = Path(f"{experiment_dir}/dynamic_reweight_on_distance/dynamic_margin_by_neg_distance-clamp_only_negative_loss-dynamic_weight.yml")
# exp_index = 0# experiments[2].index(experiment_to_go)
# for exp in experiments[exp_index:]:
#     for norm in ["batch"]: #, "instance"]:
#         version = f"./lewm/{exp}"
#         index = file_list.index('machine-1-1.txt')
#         index_scip = file_list.index('machine-1-1.txt')
#         for filename in file_list[index:]:
#             if (exp == experiment_to_go) and (filename in file_list[:index_scip]):
#                 continue
#             print(filename)
#             # Run the pretext script
#             patch = EasyDict({"res_kwargs": {}})
#             classification_args = EasyDict({"config_env": "configs/env.yml",
#                             "config_exp": str(exp),
#                             "fname": filename,
#                             "version": f"{version}"})
#             lewm(classification_args, update_dictionary=patch)
# # %%
# experiment_to_go = Path(f"{experiment_dir}/dynamic_reweight_on_distance/dynamic_margin_by_neg_distance-clamp_only_negative_loss-dynamic_weight.yml")
# exp_index = 0# experiments[2].index(experiment_to_go)
# for exp in experiments[exp_index:]:
#     for norm in ["batch"]: #, "instance"]:
#         version = f"./lewm/{exp}_projector_off"
#         index = file_list.index('machine-1-1.txt')
#         index_scip = file_list.index('machine-1-1.txt')
#         for filename in file_list[index:]:
#             if (exp == experiment_to_go) and (filename in file_list[:index_scip]):
#                 continue
#             print(filename)
#             # Run the pretext script
#             patch = EasyDict({
#                 "use_projector": False,
#             })
#             classification_args = EasyDict({"config_env": "configs/env.yml",
#                             "config_exp": str(exp),
#                             "fname": filename,
#                             "version": f"{version}"})
#             lewm(classification_args, update_dictionary=patch)


# experiment_to_go = Path(f"{experiment_dir}/dynamic_reweight_on_distance/dynamic_margin_by_neg_distance-clamp_only_negative_loss-dynamic_weight.yml")
# exp_index = 0# experiments[2].index(experiment_to_go)
# for exp in experiments[exp_index:]:
#     for norm in ["batch"]: #, "instance"]:
#         version = f"./lewm/{exp}"
#         index = file_list.index('machine-1-1.txt')
#         index_scip = file_list.index('machine-1-1.txt')
#         for filename in file_list[index:]:
#             if (exp == experiment_to_go) and (filename in file_list[:index_scip]):
#                 continue
#             print(filename)
#             # Run the pretext script
#             patch = EasyDict({"stage": "score"})
#             classification_args = EasyDict({"config_env": "configs/env.yml",
#                             "config_exp": str(exp),
#                             "fname": filename,
#                             "version": f"{version}"})
#             lewm(classification_args, update_dictionary=patch)
# %%
# experiment_to_go = Path(f"{experiment_dir}/dynamic_reweight_on_distance/dynamic_margin_by_neg_distance-clamp_only_negative_loss-dynamic_weight.yml")
# exp_index = 0# experiments[2].index(experiment_to_go)
# for exp in experiments[exp_index:]:
#     for norm in ["batch"]: #, "instance"]:
#         version = f"./lewm/{exp}_projector_off"
#         index = file_list.index('machine-1-1.txt')
#         index_scip = file_list.index('machine-1-1.txt')
#         for filename in file_list[index:]:
#             if (exp == experiment_to_go) and (filename in file_list[:index_scip]):
#                 continue
#             print(filename)
#             # Run the pretext script
#             patch = EasyDict({
#                 "stage": "score",
#                 "use_projector": False
#             })
#             classification_args = EasyDict({"config_env": "configs/env.yml",
#                             "config_exp": str(exp),
#                             "fname": filename,
#                             "version": f"{version}"})
#             lewm(classification_args, update_dictionary=patch)


experiment_to_go = Path(f"{experiment_dir}/dynamic_reweight_on_distance/dynamic_margin_by_neg_distance-clamp_only_negative_loss-dynamic_weight.yml")
exp_index = 0# experiments[2].index(experiment_to_go)
for exp in experiments[exp_index:]:
    for norm in ["batch"]: #, "instance"]:
        version = f"./lewm/{exp}_projector_off"
        index = file_list.index('machine-1-1.txt')
        index_scip = file_list.index('machine-1-1.txt')
        for filename in file_list[index:]:
            if (exp == experiment_to_go) and (filename in file_list[:index_scip]):
                continue
            print(filename)
            # Run the pretext script
            patch = EasyDict({
                "stage": "score",
                "use_projector": False
            })
            classification_args = EasyDict({"config_env": "configs/env.yml",
                            "config_exp": str(exp),
                            "fname": filename,
                            "version": f"{version}"})
            steered(classification_args, update_dictionary=patch)