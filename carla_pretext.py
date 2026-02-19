import argparse
import os
import torch
import numpy as np

from utils.config import create_config
from utils.common_config import get_criterion, get_model, get_train_dataset,\
                                get_val_dataset, get_train_dataloader,\
                                get_val_dataloader, get_train_transformations,\
                                get_val_transformations, get_val_transformations1, get_optimizer,\
                                adjust_learning_rate, inject_sub_anomaly
from utils.evaluate_utils import contrastive_evaluate
from utils.repository import TSRepository, fill_ts_repository
from utils.train_utils import pretext_train
from utils.utils import Logger

import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(4)

device = torch.device("cuda")


def main(args):
    p = create_config(args.config_env, args.config_exp, args.fname, args.version)
    logger = Logger(p['version'], verbose=2, file_path=p['pretext_dir'], use_tensorboard=True)
    logger.log('CARLA Pretext stage --> ')

    logger.log_hyperparams(p)
    
    model = get_model(p)
    best_model = None
    logger.add_graph(model, torch.rand(p['res_kwargs']['in_channels'], p['wsz']).unsqueeze(0))
    model = model.to(device)
   
    # CUDNN
    # torch.backends.cudnn.benchmark = True

    train_transforms = get_train_transformations(p)

    sanomaly = inject_sub_anomaly(p)
    val_transforms = get_val_transformations1(p)

    train_dataset = get_train_dataset(p, train_transforms, sanomaly, to_augmented_dataset=True)
    val_dataset = get_val_dataset(p, val_transforms, sanomaly, False, train_dataset.mean,
                                    train_dataset.std)

    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    base_dataloader = get_val_dataloader(p, train_dataset)

    logger.log('Dataset contains {}/{} train/val samples'.format(len(train_dataset), len(val_dataset)))
    
    # TS Repository
   # base_dataset = get_train_dataset(p, train_transforms, panomaly, sanomaly, to_augmented_dataset=True, split='train')

    ts_repository_base = TSRepository(len(train_dataset),
                                      p['model_kwargs']['features_dim'],
                                      p['num_classes'], p['criterion_kwargs']['temperature'])
    ts_repository_base.to(device)
    ts_repository_val = TSRepository(len(val_dataset),
                                     p['model_kwargs']['features_dim'],
                                     p['num_classes'], p['criterion_kwargs']['temperature'])
    ts_repository_val.to(device)

    criterion = get_criterion(p)
    criterion = criterion.to(device)

    # optimizer = get_optimizer(p, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=p['optimizer_kwargs']['lr'])
 
    # Checkpoint
    if os.path.exists(p['pretext_checkpoint']):
        logger.log('Restart from checkpoint {}'.format(p['pretext_checkpoint']))
        checkpoint = torch.load(p['pretext_checkpoint'], map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        start_epoch = checkpoint['epoch']

    else:
        logger.log('No checkpoint file at {}'.format(p['pretext_checkpoint']))
        start_epoch = 0
        model = model.to(device)
    
    # Training
    pretext_best_loss = np.inf
    prev_loss = None
    for epoch in range(start_epoch, p['epochs']):
        logger.log('Epoch %d/%d' %(epoch+1, p['epochs']))
        logger.log('-'*15)

        lr = adjust_learning_rate(p, optimizer, epoch)
        logger.log('Adjusted learning rate to {:.5f}'.format(lr))
        
        # logger.log('EPOCH ----> ', epoch)
        loss_dict = pretext_train(train_dataloader, model, criterion, optimizer, epoch, prev_loss, logger, device=device)
        tmp_loss = loss_dict['loss']
        
        if epoch % 50 == 0 or epoch == p['epochs']-1 or tmp_loss <= pretext_best_loss:
            logger.metrics_summary("Pretext Loss", loss_dict, epoch)
            feats, metadata, evaluation_metrics = contrastive_evaluate(train_dataloader, model, output_metrics=p.get('evaluation_extra_metrics', False))
            logger.add_embedding("Cluster", feats, metadata, epoch)
            logger.metrics_summary("Pretext Evaluation", evaluation_metrics, epoch)

        # Checkpoint
        if tmp_loss <= pretext_best_loss:
            pretext_best_loss = tmp_loss
            best_model = model

    # Save final model
    torch.save(best_model.state_dict(), p['pretext_model'])

    # Mine the topk nearest neighbors at the very end (Train)
    # These will be served as input to the classification loss.
    logger.log('Fill TS Repository for mining the nearest/furthest neighbors (train) ...')
    ts_repository_aug = TSRepository(len(train_dataset) * 2,
                                     p['model_kwargs']['features_dim'],
                                     p['num_classes'], p['criterion_kwargs']['temperature']) #need size of repository == 1+num_of_anomalies
    fill_ts_repository(p, base_dataloader, model, ts_repository_base, real_aug = True, ts_repository_aug = ts_repository_aug)
    # out_pre = np.column_stack((ts_repository_base.features, ts_repository_base.targets))
    out_pre = np.column_stack((ts_repository_base.features.cpu().numpy(), ts_repository_base.targets.cpu().numpy()))

    np.save(p['pretext_features_train_path'], out_pre)
    topk = 10
    logger.log('Mine the nearest neighbors (Top-%d)' %(topk))
    kfurtherst, knearest = ts_repository_aug.furthest_nearest_neighbors(topk)
    np.save(p['topk_neighbors_train_path'], knearest)
    np.save(p['bottomk_neighbors_train_path'], kfurtherst)

    # Mine the topk nearest neighbors at the very end (Val)
    # These will be used for validation.
    logger.log('Fill TS Repository for mining the nearest/furthest neighbors (val) ...')

    fill_ts_repository(p, val_dataloader, model, ts_repository_val, real_aug=False, ts_repository_aug=None)
    # out_pre = np.column_stack((ts_repository_val.features, ts_repository_val.targets))
    out_pre = np.column_stack((ts_repository_val.features.cpu().numpy(), ts_repository_val.targets.cpu().numpy()))

    np.save(p['pretext_features_test_path'], out_pre)
    topk = 10
    logger.log('Mine the nearest and furthest neighbors (Top-%d)' %(topk))
    kfurtherst, knearest = ts_repository_val.furthest_nearest_neighbors(topk)
    np.save(p['topk_neighbors_val_path'], knearest)
    np.save(p['bottomk_neighbors_val_path'], kfurtherst)
    logger.finalize()

 
if __name__ == '__main__':
    # Parser
    parser = argparse.ArgumentParser(description='pretext')
    parser.add_argument('--config_env', help='Config file for the environment')
    parser.add_argument('--config_exp', help='Config file for the experiment')
    parser.add_argument('--fname', help='Config the file name of Dataset')
    parser.add_argument('--version', help='Experiment version', type=str)
    args = parser.parse_args()
    main(args)
