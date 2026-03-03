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
filename = os.path.join('datasets/', 'psm/train')
# %% 
version = "loss/original"
# Run the pretext script
pretext_args = EasyDict({"config_env": "configs/env.yml",
                "config_exp": "configs/pretext/new_loss/psm/orig_re_weight_and_clamp_neg_re_weight/original.yml",
                "fname": filename,
                "version": f"{version}"})
main_pretext(pretext_args)

version = "loss/original"
# Run the pretext script
pretext_args = EasyDict({"config_env": "configs/env.yml",
                "config_exp": "configs/pretext/new_loss/psm/orig_re_weight_and_clamp_neg_re_weight/original.yml",
                "fname": filename,
                "version": f"{version}"})
main_pretext(pretext_args)


version = "loss/original_re_weight"
# Run the pretext script
pretext_args = EasyDict({"config_env": "configs/env.yml",
                "config_exp": "configs/pretext/new_loss/psm/orig_re_weight_and_clamp_neg_re_weight/original_reweight.yml",
                "fname": filename,
                "version": f"{version}"})
main_pretext(pretext_args)

version = "loss/original_neg_clamp"
# Run the pretext script
pretext_args = EasyDict({"config_env": "configs/env.yml",
                "config_exp": "configs/pretext/new_loss/psm/orig_re_weight_and_clamp_neg_re_weight/original_neg_clamp.yml",
                "fname": filename,
                "version": f"{version}"})
main_pretext(pretext_args)

version = "loss/original_neg_clamp_re_weight"
# Run the pretext script
pretext_args = EasyDict({"config_env": "configs/env.yml",
                "config_exp": "configs/pretext/new_loss/psm/orig_re_weight_and_clamp_neg_re_weight/original_neg_clamp_reweight.yml",
                "fname": filename,
                "version": f"{version}"})
main_pretext(pretext_args)

#==================================================================================================================================
version = "loss/original_margin_dist"
# Run the pretext script
pretext_args = EasyDict({"config_env": "configs/env.yml",
                "config_exp": "configs/pretext/new_loss/psm/margin_dist_and_supr_re_weight_clamp_neg_re_weight/original_margin_dist.yml",
                "fname": filename,
                "version": f"{version}"})
main_pretext(pretext_args)

version = "loss/original_margin_dist_re_weight"
# Run the pretext script
pretext_args = EasyDict({"config_env": "configs/env.yml",
                "config_exp": "configs/pretext/new_loss/psm/margin_dist_and_supr_re_weight_clamp_neg_re_weight/original_margin_dist_re_weight.yml",
                "fname": filename,
                "version": f"{version}"})
main_pretext(pretext_args)

version = "loss/original_margin_dist_pos_supr"
# Run the pretext script
pretext_args = EasyDict({"config_env": "configs/env.yml",
                "config_exp": "configs/pretext/new_loss/psm/margin_dist_and_supr_re_weight_clamp_neg_re_weight/original_margin_dist_pos_supr.yml",
                "fname": filename,
                "version": f"{version}"})
main_pretext(pretext_args)

version = "loss/original_margin_dist_pos_supr_re_weight"
# Run the pretext script
pretext_args = EasyDict({"config_env": "configs/env.yml",
                "config_exp": "configs/pretext/new_loss/psm/margin_dist_and_supr_re_weight_clamp_neg_re_weight/original_margin_dist_pos_supr_re_weight.yml",
                "fname": filename,
                "version": f"{version}"})
main_pretext(pretext_args)

version = "loss/original_margin_dist_neg_clamp"
# Run the pretext script
pretext_args = EasyDict({"config_env": "configs/env.yml",
                "config_exp": "configs/pretext/new_loss/psm/margin_dist_and_supr_re_weight_clamp_neg_re_weight/original_margin_dist_neg_clamp.yml",
                "fname": filename,
                "version": f"{version}"})
main_pretext(pretext_args)

version = "loss/original_margin_dist_neg_clamp_re_weight"
# Run the pretext script
pretext_args = EasyDict({"config_env": "configs/env.yml",
                "config_exp": "configs/pretext/new_loss/psm/margin_dist_and_supr_re_weight_clamp_neg_re_weight/original_margin_dist_neg_clamp_re_weight.yml",
                "fname": filename,
                "version": f"{version}"})
main_pretext(pretext_args)

version = "loss/original_margin_dist_pos_supr_neg_clamp"
# Run the pretext script
pretext_args = EasyDict({"config_env": "configs/env.yml",
                "config_exp": "configs/pretext/new_loss/psm/margin_dist_and_supr_re_weight_clamp_neg_re_weight/original_margin_dist_pos_supr_neg_clamp.yml",
                "fname": filename,
                "version": f"{version}"})
main_pretext(pretext_args)

version = "loss/original_margin_dist_pos_supr_neg_clamp_re_weight"
# Run the pretext script
pretext_args = EasyDict({"config_env": "configs/env.yml",
                "config_exp": "configs/pretext/new_loss/psm/margin_dist_and_supr_re_weight_clamp_neg_re_weight/original_margin_dist_pos_supr_neg_clamp_re_weight.yml",
                "fname": filename,
                "version": f"{version}"})
main_pretext(pretext_args)

#==================================================================================================================================
version = "loss/original_margin_ema"
# Run the pretext script
pretext_args = EasyDict({"config_env": "configs/env.yml",
                "config_exp": "configs/pretext/new_loss/psm/margin_ema_and_supr_re_weight_clamp_neg_re_weight/original_margin_ema.yml",
                "fname": filename,
                "version": f"{version}"})
main_pretext(pretext_args)

version = "loss/original_margin_ema_re_weight"
# Run the pretext script
pretext_args = EasyDict({"config_env": "configs/env.yml",
                "config_exp": "configs/pretext/new_loss/psm/margin_ema_and_supr_re_weight_clamp_neg_re_weight/original_margin_ema_re_weight.yml",
                "fname": filename,
                "version": f"{version}"})
main_pretext(pretext_args)

version = "loss/original_margin_ema_pos_supr"
# Run the pretext script
pretext_args = EasyDict({"config_env": "configs/env.yml",
                "config_exp": "configs/pretext/new_loss/psm/margin_ema_and_supr_re_weight_clamp_neg_re_weight/original_margin_ema_pos_supr.yml",
                "fname": filename,
                "version": f"{version}"})
main_pretext(pretext_args)

version = "loss/original_margin_ema_pos_supr_re_weight"
# Run the pretext script
pretext_args = EasyDict({"config_env": "configs/env.yml",
                "config_exp": "configs/pretext/new_loss/psm/margin_ema_and_supr_re_weight_clamp_neg_re_weight/original_margin_ema_pos_supr_re_weight.yml",
                "fname": filename,
                "version": f"{version}"})
main_pretext(pretext_args)

version = "loss/original_margin_ema_neg_clamp"
# Run the pretext script
pretext_args = EasyDict({"config_env": "configs/env.yml",
                "config_exp": "configs/pretext/new_loss/psm/margin_ema_and_supr_re_weight_clamp_neg_re_weight/original_margin_ema_neg_clamp.yml",
                "fname": filename,
                "version": f"{version}"})
main_pretext(pretext_args)

version = "loss/original_margin_ema_neg_clamp_re_weight"
# Run the pretext script
pretext_args = EasyDict({"config_env": "configs/env.yml",
                "config_exp": "configs/pretext/new_loss/psm/margin_ema_and_supr_re_weight_clamp_neg_re_weight/original_margin_ema_neg_clamp_re_weight.yml",
                "fname": filename,
                "version": f"{version}"})
main_pretext(pretext_args)

version = "loss/original_margin_ema_pos_supr_neg_clamp"
# Run the pretext script
pretext_args = EasyDict({"config_env": "configs/env.yml",
                "config_exp": "configs/pretext/new_loss/psm/margin_ema_and_supr_re_weight_clamp_neg_re_weight/original_margin_ema_pos_supr_neg_clamp.yml",
                "fname": filename,
                "version": f"{version}"})
main_pretext(pretext_args)

version = "loss/original_margin_ema_pos_supr_neg_clamp_re_weight"
# Run the pretext script
pretext_args = EasyDict({"config_env": "configs/env.yml",
                "config_exp": "configs/pretext/new_loss/psm/margin_ema_and_supr_re_weight_clamp_neg_re_weight/original_margin_ema_pos_supr_neg_clamp_re_weight.yml",
                "fname": filename,
                "version": f"{version}"})
main_pretext(pretext_args)

# # Run the classification script
# classification_args = EasyDict({"config_env": "configs/env.yml",
#                 "config_exp": "configs/classification/carla_classification_smd_twoB_threeC.yml",
#                 "fname": filename,
#                 "version": f"{version}"})

# main_classification(classification_args)

# with open(classification_args.config_exp, 'r') as stream:
#             config = yaml.safe_load(stream)
# eval_args = EasyDict({"version": f"{version}",
#                     "save_dir": f"{config.get('tag_class', None)}"})
# main_evaluation(eval_args)