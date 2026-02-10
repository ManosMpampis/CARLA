# %%
import numpy as np
import pandas as pd
import os
import subprocess
from easydict import EasyDict

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

# version = "original_with_logs"
# for filename in file_list: #['machine-1-2.txt']: #file_list: #[index:]:  #['GECCO']: #['machine-2-4.txt']:
#     # if 'real_' in filename:
#     if filename != 'GECCO':
#         print(filename)

#         # Run the pretext script
#         pretext_args = EasyDict({"config_env": "configs/env.yml",
#                         "config_exp": "configs/pretext/carla_pretext_smd_original.yml",
#                         "fname": filename,
#                         "version": f"{version}"})
#         main_pretext(pretext_args)
        
#         # Run the classification script
#         classification_args = EasyDict({"config_env": "configs/env.yml",
#                         "config_exp": "configs/classification/carla_classification_smd_original.yml",
#                         "fname": filename,
#                         "version": f"{version}"})
#         main_classification(classification_args)

# eval_args = EasyDict({"version": f"{version}"})
# main_evaluation(eval_args)

index = file_list.index('machine-1-4.txt')
version = "orig_big_batch/entropy_norm"
for filename in file_list[index:]: #['machine-1-2.txt']: #file_list: #[index:]:  #['GECCO']: #['machine-2-4.txt']:
    # if 'real_' in filename:
    if filename != 'GECCO':
        print(filename)

        # Run the pretext script
        # pretext_args = EasyDict({"config_env": "configs/env.yml",
        #                 "config_exp": "configs/pretext/carla_pretext_smd_new_w.yml",
        #                 "fname": filename,
        #                 "version": f"{version}"})
        # main_pretext(pretext_args)
        
        # Run the classification script
        classification_args = EasyDict({"config_env": "configs/env.yml",
                        "config_exp": "configs/classification/carla_classification_smd_1_entropy.yml",
                        "fname": filename,
                        "version": f"{version}"})
        main_classification(classification_args)

eval_args = EasyDict({"version": f"{version}"})
main_evaluation(eval_args)
