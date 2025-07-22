import argparse
import os
import random

import torch
from torch.utils.tensorboard import SummaryWriter 
import pandas
import numpy as np

from utils.mypath import MyPath
from utils.config import create_config
from utils.common_config import get_train_transformations, get_val_transformations,\
                                get_train_dataset, get_train_dataloader, get_aug_train_dataset,\
                                get_val_dataset, get_val_dataloader,\
                                get_optimizer, get_model, get_criterion,\
                                adjust_learning_rate, inject_sub_anomaly
from utils.evaluate_utils import get_predictions, classification_evaluate, pr_evaluate
from utils.train_utils import self_sup_classification_train
from utils.utils import log

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
    file_path=os.path.join(p['experiment_dir'], "classification.txt") if verbose>=2 else None
    verbose_dict={"verbose": verbose, "file_path": file_path}

    log('CARLA Self-supervised Classification stage --> ', verbose=verbose, file_path=file_path, color='yellow')

    # CUDNN
   # torch.backends.cudnn.benchmark = True

    # Data
    log(f"\n- Get dataset and dataloaders for {p['train_db_name']} dataset - timeseries {p['fname']}", verbose=verbose, file_path=file_path, color='green')
    train_transformations = get_train_transformations(p)
    sanomaly = inject_sub_anomaly(p)
    val_transformations = get_val_transformations(p)
    # In the self-supervised state we use as data the triplets with saves anchors of the first stage
    train_dataset = get_aug_train_dataset(p, train_transformations, to_neighbors_dataset = True)
    train_dataloader = get_train_dataloader(p, train_dataset)
    # In order to correctly measure the similarity matrics, all values need to be checkes,
    # during train we have drop_last so we add second dataloader
    tst_dataloader = get_val_dataloader(p, train_dataset)

    if p['train_db_name'] == 'MSL' or p['train_db_name'] == 'SMAP':
        if p['fname'] == 'All':
            with open(os.path.join(MyPath.db_root_dir('msl'), 'labeled_anomalies.csv'), 'r') as file:
                csv_reader = pandas.read_csv(file, delimiter=',')
            data_info = csv_reader[csv_reader['spacecraft'] == p['train_db_name']]
            ii = 0
            for file_name in data_info['chan_id']:
                p['fname'] = file_name
                if ii == 0 :
                    base_dataset = get_train_dataset(p, train_transformations, sanomaly,
                                                     to_neighbors_dataset=True)
                    val_dataset = get_val_dataset(p, val_transformations, sanomaly, True, base_dataset.mean,
                                                  base_dataset.std)
                else:
                    new_base_dataset = get_train_dataset(p, train_transformations, sanomaly,
                                                     to_neighbors_dataset=True)
                    new_val_dataset = get_val_dataset(p, val_transformations, sanomaly, True, new_base_dataset.mean,
                                                  new_base_dataset.std)
                    val_dataset.concat_ds(new_val_dataset)
                    base_dataset.concat_ds(new_base_dataset)
                ii+=1
        else:
            #base_dataset = get_aug_train_dataset(p, train_transformations, to_neighbors_dataset = True)
            info_ds = get_train_dataset(p, train_transformations, sanomaly, to_neighbors_dataset=False)
            val_dataset = get_val_dataset(p, val_transformations, sanomaly, False, info_ds.mean, info_ds.std)

    elif p['train_db_name'] == 'smd' or p['train_db_name'] == 'kpi' or p['train_db_name'] == 'swat' \
        or p['train_db_name'] == 'swan' or p['train_db_name'] == 'wadi':
        base_dataset = get_train_dataset(p, train_transformations, sanomaly, to_augmented_dataset=True)  # used only to mean and std
        dataset_mean = base_dataset.mean
        dataset_std = base_dataset.std
        del base_dataset

        val_dataset = get_val_dataset(p, val_transformations, sanomaly, False, dataset_mean,
                                      dataset_std)
    else:
        raise ValueError('Invalid train dataset {}'.format(p['train_db_name']))

    val_dataloader = get_val_dataloader(p, val_dataset)

    log(f'-- Train samples size: {len(train_dataset)} - Test samples size: {len(val_dataset)}', verbose=verbose, file_path=file_path, color='green')

    # Model
    model = get_model(p, p['pretext_model'])
    model = model.to(device)

    # Optimizer
    optimizer = get_optimizer(p, model, p['update_cluster_head_only'])

    # Warning
    if p['update_cluster_head_only']:
        log('WARNING: classification will only update the cluster head', verbose=verbose, file_path=file_path, color='red')

    # Loss function
    criterion = get_criterion(p)
    criterion = criterion.to(device)

    log('\n- Model initialisation', verbose=verbose, file_path=file_path, color='green')
    # Checkpoint
    if os.path.exists(p['classification_checkpoint']):
        log(f'-- Model initialised from last checkpoint: {p['classification_checkpoint']}', verbose=verbose, file_path=file_path, color='green')
        checkpoint = torch.load(p['classification_checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        best_loss_head = checkpoint['best_loss_head']
        normal_label = checkpoint['normal_label']
    else:
        log(f'-- No checkpoint file at {p["classification_checkpoint"]} -- new model initialised', verbose=verbose, file_path=file_path, color='green')
        start_epoch = 0
        best_loss = 1e4
        best_loss_head = None
        normal_label = 0


    best_f1 = -1 * torch.tensor(float("inf"), device=device)
    log('\n- Training:', verbose=verbose, file_path=file_path, color='blue')
    
    import time
    for epoch in range(start_epoch, p['epochs']):
        log(f'-- Epoch {epoch+1}/{p['epochs']}', verbose=verbose, file_path=file_path, color='blue')

        lr = adjust_learning_rate(p, optimizer, epoch)
        
        # torch.cuda.synchronize()
        # start_time = time.time()

        total_losses, consistency_losses, inconsistency_losses, entropy_losses = \
            self_sup_classification_train(train_dataloader, model, criterion, optimizer, epoch,
                                    p['update_cluster_head_only'], verbose_dict=verbose_dict)
        # torch.cuda.synchronize()
        # end_time = time.time()
        # inference_time = end_time - start_time

        # print(f'Inference time with gpu sync: {inference_time:.4f} sencods')

        # torch.cuda.synchronize()
        # start_time = time.time()
        if (epoch == p['epochs']-1):
            predictions, _ = get_predictions(p, tst_dataloader, model, True, True)
        else:
            predictions = get_predictions(p, tst_dataloader, model, False, False)

        # torch.cuda.synchronize()
        # end_time = time.time()
        # inference_time = end_time - start_time

        # print(f'Get prediction tst time with gpu sync: {inference_time:.4f} sencods')
        # torch.cuda.synchronize()
        # start_time = time.time()
        label_counts = torch.bincount(predictions[0]['predictions'])
        nomral_label = label_counts.argmax()  # majority_label

        classification_stats = classification_evaluate(predictions, **p['criterion_kwargs'])
        lowest_loss_head = classification_stats['lowest_loss_head']
        lowest_loss = classification_stats['lowest_loss']

        # torch.cuda.synchronize()
        # end_time = time.time()
        # inference_time = end_time - start_time

        # print(f'Class evaluate time with gpu sync: {inference_time:.4f} sencods')
        # torch.cuda.synchronize()
        # start_time = time.time()
        predictions = get_predictions(p, val_dataloader, model, False, False)

        rep_f1 = pr_evaluate(predictions, majority_label=nomral_label)
        # torch.cuda.synchronize()
        # end_time = time.time()
        # inference_time = end_time - start_time
        # print(f'Evaluate time with gpu sync: {inference_time:.4f} sencods')
        if rep_f1 > best_f1:
            best_f1 = rep_f1
            # log('New Checkpoint ...', verbose=verbose, file_path=file_path)
            torch.save({'model': model.state_dict(), 'head': best_loss_head, 'normal_label': nomral_label}, p['classification_model'])
            torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                        'epoch': epoch + 1, 'best_loss': best_loss, 'best_loss_head': best_loss_head, 'normal_label': nomral_label},
                       p['classification_checkpoint'])

    model_checkpoint = torch.load(p['classification_model'], map_location='cpu')
    model.load_state_dict(model_checkpoint['model'])
    torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                'epoch': p['epochs'], 'best_loss': best_loss, 'best_loss_head': best_loss_head, 'normal_label': normal_label},
               p['classification_checkpoint'])
    normal_label = model_checkpoint['normal_label']
    tst_dl = get_val_dataloader(p, val_dataset)
    predictions, _ = get_predictions(p, tst_dl, model, True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='classification Loss')
    parser.add_argument('--config_env', help='Location of path config file', type=str, default='configs/env.yml')
    parser.add_argument('--config_exp', help='Location of experiments config file', type=str, default='configs/classification/carla_classification_smd.yml')
    parser.add_argument('--fname', help='Config the file name of Dataset', type=str, default='machine-1-1.txt')
    parser.add_argument('--device', help='Device used to load the model', type=str, choices=['cpu', 'cuda', 'auto'], default='auto')
    parser.add_argument('--verbose', help='Enable logging messages level. 0: No verbose, 1: Terminal infor, 2: Terminal and file', type=int, choices=[0, 1, 2], default=2)
    parser.add_argument('--tensorboard', help='Enable tensorboard logging', type=int, choices=[0, 1], default=1)
    args = parser.parse_args()

    global verbose, file_path
    file_path = None
    verbose = args.verbose

    main(args=args)
