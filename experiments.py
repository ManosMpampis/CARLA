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
experiment_group = [exp for exp in experiment_dir.iterdir() if exp.is_dir()]
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
for exp in experiments[0]:
    version = f"batch/original-dynamic_weight-loss_clamp/{exp.name[:-4]}"
    index = file_list.index('machine-1-1.txt')
    if "original.yml" not in str(exp):
        for filename in file_list[index:]:
            print(filename)

            # Run the pretext script
            pretext_args = EasyDict({"config_env": "configs/env.yml",
                            "config_exp": str(exp),
                            "fname": filename,
                            "version": f"{version}"})
            main_pretext(pretext_args)

# %% Pretext ema_loss
for exp in experiments[1]:
    version = f"batch/ema_loss/{exp.name[:-4]}"
    index = file_list.index('machine-1-1.txt')
    if "temp.yml" not in str(exp):
        for filename in file_list[index:]:
            print(filename)

            # Run the pretext script
            pretext_args = EasyDict({"config_env": "configs/env.yml",
                            "config_exp": str(exp),
                            "fname": filename,
                            "version": f"{version}"})
            main_pretext(pretext_args)

# %% Pretext dynamic_reweight_on_distance
for exp in experiments[2]:
    version = f"batch/dynamic_reweight_on_distance/{exp.name[:-4]}"
    index = file_list.index('machine-1-1.txt')
    if "temp.yml" not in str(exp):
        for filename in file_list[index:]:
            print(filename)

            # Run the pretext script
            pretext_args = EasyDict({"config_env": "configs/env.yml",
                            "config_exp": str(exp),
                            "fname": filename,
                            "version": f"{version}"})
            main_pretext(pretext_args)

# index = file_list.index('machine-1-1.txt')
# for filename in file_list[index:]:
#     # Run the classification script
#     classification_args = EasyDict({"config_env": "configs/env.yml",
#                     "config_exp": "configs/classification/batch_common.yml",
#                     "fname": filename,
#                     "version": f"{version_batch}"})
#     main_classification(classification_args)

