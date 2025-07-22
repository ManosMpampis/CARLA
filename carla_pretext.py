import os
import random
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas

from utils.mypath import MyPath
from utils.config import create_config
from utils.common_config import get_criterion, get_model, get_train_dataset,\
                                get_val_dataset,\
                                get_dataloader, get_transformations, get_optimizer,\
                                adjust_learning_rate, inject_sub_anomaly
from utils.evaluate_utils import contrastive_evaluate
from utils.repository import TSRepository
from utils.train_utils import pretext_train
from utils.utils import fill_ts_repository, log


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(4)

def main(args):
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    p = create_config(args.config_env, args.config_exp, args.fname)

    # Initialize logging
    verbose = args.verbose
    file_path=os.path.join(p['experiment_dir'], "pretext.txt") if verbose>=2 else None
    verbose_dict={"verbose": verbose, "file_path": file_path}

    log('CARLA Pretext stage --> ', verbose=verbose, file_path=file_path, color='yellow')

    model = get_model(p)
    model = model.to(device)

    train_transforms = get_transformations(p)

    sanomaly = inject_sub_anomaly(p)
    val_transforms = get_transformations(p)
    val_dataset, train_dataset = get_val_dataset(p, train_transforms, val_transforms, sanomaly)

    train_dataloader = get_dataloader(p, train_dataset, drop_last=True, shuffle=True)
    val_dataloader = get_dataloader(p, val_dataset)
    base_dataloader = get_dataloader(p, train_dataset)

    log(f'Dataset contains {train_dataset}/{val_dataset} train/val samples', verbose=verbose, file_path=file_path)

    criterion = get_criterion(p).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=p['optimizer_kwargs']['lr'])
 
    # Checkpoint
    if os.path.exists(p['pretext_checkpoint']):
        log(f"Restart from checkpoint {p['pretext_checkpoint']}", verbose=verbose, file_path=file_path, color='blue')
        checkpoint = torch.load(p['pretext_checkpoint'], map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        model = model.to(device, non_blocking=True)
        start_epoch = checkpoint['epoch']
        pretext_best_loss = checkpoint['pretext_best_loss'].to(device, non_blocking=True)
        pretext_previous_loss = checkpoint['pretext_previous_loss'].to(device, non_blocking=True)
    else:
        log(f'No checkpoint file at {p["pretext_checkpoint"]}', verbose=verbose, file_path=file_path, color='blue')
        start_epoch = 0
        pretext_best_loss = torch.tensor(float("inf"), device=device)
        pretext_previous_loss = torch.tensor(float(0), device=device)
    
    # Training
    
    for epoch in range(start_epoch, p['epochs']):
        log(f'Epoch {epoch+1}/{p['epochs']}', verbose=verbose, file_path=file_path, color='yellow')
        log('-'*15, verbose=verbose, file_path=file_path, color='yellow')

        lr = adjust_learning_rate(p, optimizer, epoch)
        log(f'Adjusted learning rate to {lr:.5f}', verbose=verbose, file_path=file_path)
        
        # log(f'EPOCH ----> {epoch}', verbose=verbose, file_path=file_path)
        tmp_loss = pretext_train(train_dataloader, model, criterion, optimizer, epoch, pretext_previous_loss)
        
        # Checkpoint
        if tmp_loss <= pretext_best_loss:
            pretext_best_loss = tmp_loss
            torch.save({'model': model.state_dict()}, p['pretext_model'])
            torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                        'epoch': epoch + 1, 'pretext_best_loss': pretext_best_loss,
                        'pretext_previous_loss': pretext_previous_loss},
                        p['pretext_checkpoint'])


    model_checkpoint = torch.load(p['pretext_model'], map_location='cpu')
    model.load_state_dict(model_checkpoint['model'])
    
    # TS Repository
    ts_repository_base = TSRepository(len(train_dataset),
                                      p['model_kwargs']['features_dim'],
                                      p['num_classes'], p['criterion_kwargs']['temperature'])
    ts_repository_base.to(device)

    ts_repository_val = TSRepository(len(val_dataset),
                                     p['model_kwargs']['features_dim'],
                                     p['num_classes'], p['criterion_kwargs']['temperature'])
    ts_repository_val.to(device)

    # Mine the topk nearest neighbors at the very end (Train)
    # These will be served as input to the classification loss.
    log('Fill TS Repository for mining the nearest/furthest neighbors (train) ...', verbose=verbose, file_path=file_path, color='blue')
    ts_repository_aug = TSRepository(len(train_dataset) * 2,
                                     p['model_kwargs']['features_dim'],
                                     p['num_classes'], p['criterion_kwargs']['temperature'])  # need size of repository == 1+num_of_anomalies
    ts_repository_aug.to(device)
    
    fill_ts_repository(p, base_dataloader, model, ts_repository_base, real_aug = True, ts_repository_aug = ts_repository_aug, verbose_dict=verbose_dict)
    out_pre = np.column_stack((ts_repository_base.features.cpu().numpy(), ts_repository_base.targets.cpu().numpy()))

    np.save(p['pretext_features_train_path'], out_pre)
    topk = 10
    log(f'Mine the nearest neighbors (Top-{topk})', verbose=verbose, file_path=file_path)
    kfurtherst, knearest = ts_repository_aug.furthest_nearest_neighbors(topk)
    np.save(p['topk_neighbors_train_path'], knearest)
    np.save(p['bottomk_neighbors_train_path'], kfurtherst)

    # Mine the topk nearest neighbors at the very end (Val)
    # These will be used for validation.
    log('Fill TS Repository for mining the nearest/furthest neighbors (val) ...', verbose=verbose, file_path=file_path, color='blue')

    fill_ts_repository(p, val_dataloader, model, ts_repository_val, real_aug=False, ts_repository_aug=None, verbose_dict=verbose_dict)
    out_pre = np.column_stack((ts_repository_val.features.cpu().numpy(), ts_repository_val.targets.cpu().numpy()))

    np.save(p['pretext_features_test_path'], out_pre)
    topk = 10
    log(f'Mine the nearest and furthest neighbors (Top-{topk})', verbose=verbose, file_path=file_path)
    kfurtherst, knearest = ts_repository_val.furthest_nearest_neighbors(topk)
    np.save(p['topk_neighbors_val_path'], knearest)
    np.save(p['bottomk_neighbors_val_path'], kfurtherst)

 
if __name__ == '__main__':
    # Parser
    parser = argparse.ArgumentParser(description='pretext')
    parser.add_argument('--config_env', help='Config file for the environment', type=str, default='configs/env.yml')
    parser.add_argument('--config_exp', help='Config file for the experiment', type=str, default='configs/pretext/carla_pretext_smd.yml')
    parser.add_argument('--fname', help='Config the file name of Dataset', type=str, default='machine-1-1.txt')
    parser.add_argument('--device', help='Device used to load the model', type=str, choices=['cpu', 'cuda', 'auto'], default='auto')
    parser.add_argument('--verbose', help='Enable logging messages level. 0: No verbose, 1: Terminal infor, 2: Terminal and file', type=int, choices=[0, 1, 2], default=2)
    parser.add_argument('--tensorboard', help='Enable tensorboard logging', type=int, choices=[0, 1], default=1)
    args = parser.parse_args()

    main(args=args)
