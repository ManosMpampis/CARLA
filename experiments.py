# %%
import os
from easydict import EasyDict
import yaml

from carla_pretext import main as main_pretext
from carla_classification import main as main_classification
from evaluation import main as main_evaluation

env = os.environ.copy()
env['PYTHONPATH'] = '/home/manos/Documents/EKETA/HYPER_AI/gits/official_carla/unchanged/CARLA/'
env['PATH'] = f'/usr/local/cuda/bin:{env.get("PATH", "")}'
env['LD_LIBRARY_PATH'] = f'/usr/local/cuda/lib64:{env.get("LD_LIBRARY_PATH", "")}'


# %%
all_files = os.listdir(os.path.join('datasets/', 'SMD/train'))
file_list = [file for file in all_files if file.startswith('machine-')]
file_list = sorted(file_list)
print(file_list)


# %% Test
version = "test/test1"
index = file_list.index('machine-1-1.txt')
for filename in file_list[index:]:
    print(filename)

    # Run the pretext script
    pretext_args = EasyDict({"config_env": "configs/env.yml",
                    "config_exp": "configs/pretext/new_loss/smd/orig_re_weight_and_clamp_neg_re_weight/original.yml",
                    "fname": filename,
                    "version": f"{version}"})
    main_pretext(pretext_args)
# # %% Pretext experiments

# version_batch = "batch_norm/small_eps_re_weight_initialization"
# version_layer = "layer_norm/small_eps_re_weight_initialization"
# version_instance = "instance_norm/small_eps_re_weight_initialization"
# version_no_norm = "no_norm/small_eps_re_weight_initialization"

# index = file_list.index('machine-1-1.txt')
# for filename in file_list[index:]:
#     print(filename)

#     # Run the pretext script
#     pretext_args = EasyDict({"config_env": "configs/env.yml",
#                     "config_exp": "configs/pretext/new_loss/smd/final_lr.yml",
#                     "fname": filename,
#                     "version": f"{version_batch}"})
#     main_pretext(pretext_args)

#     # Run the pretext script
#     pretext_args = EasyDict({"config_env": "configs/env.yml",
#                     "config_exp": "configs/pretext/new_loss/smd/final_lr_layer.yml",
#                     "fname": filename,
#                     "version": f"{version_layer}"})
#     main_pretext(pretext_args)

#     # Run the pretext script
#     pretext_args = EasyDict({"config_env": "configs/env.yml",
#                     "config_exp": "configs/pretext/new_loss/smd/final_lr_layer.yml",
#                     "fname": filename,
#                     "version": f"{version_layer}"})
#     main_pretext(pretext_args)

#     # Run the classification script
#     classification_args = EasyDict({"config_env": "configs/env.yml",
#                     "config_exp": "configs/classification/batch_common.yml",
#                     "fname": filename,
#                     "version": f"{version_batch}"})
#     main_classification(classification_args)

#     # Run the classification script
#     classification_args = EasyDict({"config_env": "configs/env.yml",
#                     "config_exp": "configs/classification/layer_common.yml",
#                     "fname": filename,
#                     "version": f"{version_layer}"})
#     main_classification(classification_args)
