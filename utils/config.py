import os
import time
import yaml
from easydict import EasyDict
from utils.utils import mkdir_if_missing as mkdir

def create_config(config_file_env, config_file_exp, fname, version=None, update_dictionary={}):
    # Config for environment path
    with open(config_file_env, 'r') as stream:
        root_dir = yaml.safe_load(stream)['root_dir']
   
    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)
    
    cfg = EasyDict()
   
    # Copy
    for k, v in config.items():
        cfg[k] = v

    for k, v in update_dictionary.items():
        cfg[k] = v
    
    # Set paths for pretext task (These directories are needed in every stage)
    version = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) if version is None else version
    base_dir = os.path.join(root_dir, f'{cfg['train_db_name']}/{version}/{fname}')
    
    pretext_tag = cfg.get('tag_pretext', None)
    cfg['pretext_tag'] = ("_"+pretext_tag) if pretext_tag else ""
    pretext_dir = os.path.join(base_dir, f'pretext{cfg['pretext_tag']}')
    mkdir(base_dir)
    mkdir(pretext_dir)
    cfg['version'] = version
    cfg['experiment_dir'] = base_dir
    cfg['pretext_dir'] = pretext_dir
    cfg['fname'] = fname
    cfg['pretext_checkpoint'] = os.path.join(pretext_dir, 'checkpoint.pth.tar')
    cfg['pretext_checkpoint_last'] = os.path.join(pretext_dir, 'checkpoint_last.pth.tar')
    cfg['pretext_model'] = os.path.join(pretext_dir, 'model.pth.tar')
    cfg['topk_neighbors_train_path'] = os.path.join(pretext_dir, 'topk-train-neighbors.npy')
    cfg['bottomk_neighbors_train_path'] = os.path.join(pretext_dir, 'bottomk-train-neighbors.npy')
    cfg['aug_train_dataset'] = os.path.join(pretext_dir, 'aug_train_dataset.pth')
    cfg['pretext_features_train_path'] = os.path.join(pretext_dir, 'pretext_features_train.npy')
    cfg['pretext_features_test_path'] = os.path.join(pretext_dir, 'pretext_features_test.npy')
    cfg['topk_neighbors_val_path'] = os.path.join(pretext_dir, 'topk-test-neighbors.npy')
    cfg['bottomk_neighbors_val_path'] = os.path.join(pretext_dir, 'bottomk-test-neighbors.npy')
    cfg['bottomk_neighbors_val_path'] = os.path.join(pretext_dir, 'bottomk-test-neighbors.npy')
    cfg['contrastive_dataset'] = os.path.join(pretext_dir, 'con_train_dataset')
    cfg['contrastive_dataloader'] = os.path.join(pretext_dir, 'con_train_dataset.pth')


    if cfg['setup'] == 'jepa':
        jepa_tag = cfg.get('tag_jepa', None)
        cfg['jepa_tag'] = ("_"+jepa_tag) if jepa_tag else ""
        jepa_dir = os.path.join(base_dir, f'jepa{cfg['jepa_tag']}')
        mkdir(base_dir)
        mkdir(jepa_dir)
        cfg['jepa_dir'] = jepa_dir
        cfg['jepa_checkpoint'] = os.path.join(jepa_dir, 'checkpoint.pth.tar')
        cfg['jepa_model'] = os.path.join(jepa_dir, 'model.pth.tar')
        cfg['calibration_path'] = os.path.join(jepa_dir, 'calibration.json')
        cfg['scores_path'] = os.path.join(jepa_dir, 'scores.npz')
        cfg['metrics_path'] = os.path.join(jepa_dir, 'metrics.json')

    if cfg['setup'] in ['classification', 'classification_e2e']:
        classification_tag = cfg.get('tag_class', None)
        cfg['classification_tag'] = ("_"+classification_tag) if classification_tag else ""
        classification_dir = os.path.join(base_dir, f'classification{cfg['classification_tag']}')
        mkdir(base_dir)
        mkdir(classification_dir)
        cfg['classification_dir'] = classification_dir
        cfg['classification_checkpoint'] = os.path.join(classification_dir, 'checkpoint.pth.tar')
        cfg['classification_checkpoint_last'] = os.path.join(classification_dir, 'checkpoint_last.pth.tar')
        cfg['classification_model'] = os.path.join(classification_dir, 'model.pth.tar')
        cfg['classification_trainfeatures'] = os.path.join(classification_dir, 'classification_traintfeatures.csv')
        cfg['classification_trainprobs'] = os.path.join(classification_dir, 'classification_trainprobs.csv')
        cfg['classification_testfeatures'] = os.path.join(classification_dir, 'classification_testtfeatures.csv')
        cfg['classification_testprobs'] = os.path.join(classification_dir, 'classification_testprobs.csv')
        # Evaluation paths
        mkdir(os.path.join(classification_dir, 'best'))
        cfg['eval_train_csl'] = os.path.join(classification_dir, 'best', 'eval_train_cls.csv')
        cfg['eval_train_best'] = os.path.join(classification_dir, 'best', 'eval_train_best.csv')
        cfg['eval_test_cls'] = os.path.join(classification_dir, 'best', 'eval_test_cls.csv')
        cfg['eval_test_best'] = os.path.join(classification_dir, 'best', 'eval_test_best.csv')
        cfg['eval_test_train_th'] = os.path.join(classification_dir, 'best', 'eval_test_train_th.csv')
        cfg['eval_tstest_cls'] = os.path.join(classification_dir, 'best', 'eval_timeseries_cls.csv')
        cfg['eval_tstest_best'] = os.path.join(classification_dir, 'best', 'eval_timeseries_best.csv')
        cfg['eval_tstest_trainth'] = os.path.join(classification_dir, 'best', 'eval_timeseries_train_th.csv')
        mkdir(os.path.join(classification_dir, 'cls'))
        cfg['clseval_train_csl'] = os.path.join(classification_dir, 'cls', 'eval_train_cls.csv')
        cfg['clseval_train_best'] = os.path.join(classification_dir, 'cls', 'eval_train_best.csv')
        cfg['clseval_test_cls'] = os.path.join(classification_dir, 'cls', 'eval_test_cls.csv')
        cfg['clseval_test_best'] = os.path.join(classification_dir, 'cls', 'eval_test_best.csv')
        cfg['clseval_test_train_th'] = os.path.join(classification_dir, 'cls', 'eval_test_train_th.csv')
        cfg['clseval_tstest_cls'] = os.path.join(classification_dir, 'cls', 'eval_timeseries_cls.csv')
        cfg['clseval_tstest_best'] = os.path.join(classification_dir, 'cls', 'eval_timeseries_best.csv')
        cfg['clseval_tstest_trainth'] = os.path.join(classification_dir, 'cls', 'eval_timeseries_train_th.csv')
        

    if "res_kwargs" in cfg:
        cfg["res_kwargs"]["window_size"] = cfg["wsz"]
    return cfg
