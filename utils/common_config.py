import os
import math
import numpy as np
import pandas
import torch
import torchvision.transforms as transforms

from data.augment import NoiseTransformation, SubAnomaly
from utils.collate import collate_custom
from utils.mypath import MyPath


def get_criterion(p):
    if p['criterion'] == 'pretext':
        from losses.losses import PretextLoss
        criterion = PretextLoss(p['batch_size'], **p['criterion_kwargs'])

    elif p['criterion'] == 'classification':
        from losses.losses import ClassificationLoss
        criterion = ClassificationLoss(**p['criterion_kwargs'])

    else:
        raise ValueError('Invalid criterion {}'.format(p['criterion']))

    return criterion


def get_feature_dimensions_backbone(p):
    if p['backbone'] == 'resnet18':
        return 8

    elif p['backbone'] == 'resnet_ts':
        return 8

    else:
        raise NotImplementedError


def get_model(p, pretrain_path=None):
    # Get backbone
    if p['backbone'] == 'resnet_ts':
        from models.resent_time import resnet_ts
        backbone = resnet_ts(**p['res_kwargs'])

    else:
        raise ValueError('Invalid backbone {}'.format(p['backbone']))

    # Setup
    if p['setup'] in ['pretext']:
        from models.models import ContrastiveModel
        model = ContrastiveModel(backbone, **p['model_kwargs'])

    elif p['setup'] in ['classification']:
        from models.models import ClusteringModel
        model = ClusteringModel(backbone, p['num_classes'], p['num_heads'])

    else:
        raise ValueError('Invalid setup {}'.format(p['setup']))

    # Load pretrained weights
    if pretrain_path is not None and os.path.exists(pretrain_path):
        state = torch.load(pretrain_path, map_location='cpu')

        if p['setup'] == 'classification':  # Weights are supposed to be transfered from contrastive training
            missing = model.load_state_dict(state['model'], strict=False)
            assert (set(missing[1]) == {
                'contrastive_head.0.weight', 'contrastive_head.0.bias',
                'contrastive_head.2.weight', 'contrastive_head.2.bias'}
                    or set(missing[1]) == {
                        'contrastive_head.weight', 'contrastive_head.bias'})

        else:
            raise NotImplementedError

    elif pretrain_path is not None and not os.path.exists(pretrain_path):
        raise ValueError('Path with pre-trained weights does not exist {}'.format(pretrain_path))

    else:
        pass

    return model

def load_pretext_backbone_to_model(p, model, pretrain_path):
    if os.path.exists(pretrain_path):
        state = torch.load(pretrain_path, map_location='cpu')

        if p['setup'] == 'classification':  # Weights are supposed to be transfered from contrastive training
            missing = model.load_state_dict(state['model'], strict=False)
            assert (set(missing[1]) == {
                'contrastive_head.0.weight', 'contrastive_head.0.bias',
                'contrastive_head.2.weight', 'contrastive_head.2.bias'}
                    or set(missing[1]) == {
                        'contrastive_head.weight', 'contrastive_head.bias'})

        else:
            raise NotImplementedError

    else:
        raise ValueError('Path with pre-trained weights does not exist {}'.format(pretrain_path))
    return

def get_train_dataset(p, transform, sanomaly, to_augmented_dataset=False,
                      to_neighbors_dataset=False, split=None, data=None, label=None):
    # Base dataset
    mean, std = 0, 0
    if p['train_db_name'] == 'MSL' or p['train_db_name'] == 'SMAP':
        from data.MSL import MSL
        dataset = MSL(p['fname'], train=True, transform=transform, sanomaly=sanomaly,
                      mean_data=None, std_data=None, wsz=p['window_size'], stride=p['window_stride'])
        mean, std = dataset.get_info()

    elif p['train_db_name'] == 'kpi':
        from data.KPI import KPI
        dataset = KPI(p['fname'], train=True, transform=transform, sanomaly=sanomaly,
                       mean_data=None, std_data=None, wsz=p['window_size'], stride=p['window_stride'])
        mean, std = dataset.get_info()

    elif p['train_db_name'] == 'smd':
        from data.SMD import SMD
        dataset = SMD(p['fname'], train=True, transform=transform, sanomaly=sanomaly,
                      mean_data=None, std_data=None, wsz=p['window_size'], stride=p['window_stride'])
        mean, std = dataset.get_info()

    elif p['train_db_name'] == 'swat':
        from data.SWAT import SWAT
        dataset = SWAT(p['fname'], train=True, transform=transform, sanomaly=sanomaly,
                      mean_data=None, std_data=None, wsz=p['window_size'], stride=p['window_stride'])
        mean, std = dataset.get_info()

    elif p['train_db_name'] == 'wadi':
        from data.WADI import WADI
        dataset = WADI(p['fname'], train=True, transform=transform, sanomaly=sanomaly,
                      mean_data=None, std_data=None, wsz=p['window_size'], stride=p['window_stride'])
        mean, std = dataset.get_info()

    else:
        raise ValueError('Invalid train dataset {}'.format(p['train_db_name']))

    # Wrap into other dataset (__getitem__ changes)
    if to_augmented_dataset:  # Dataset returns a ts and an augmentation of that.
        from data.custom_dataset import AugmentedDataset
        dataset = AugmentedDataset(dataset)

    if to_neighbors_dataset:  # Dataset returns ts and its nearest and furthest neighbors.
        from data.custom_dataset import NeighborsDataset
        nindices = np.load(p['topk_neighbors_train_path'])
        findices = np.load(p['bottomk_neighbors_train_path'])
        dataset = NeighborsDataset(dataset, None, nindices, findices, p)

    dataset.mean = mean
    dataset.std = std
    return dataset


def get_aug_train_dataset(p, transform, to_neighbors_dataset=False):
    dataloader = torch.load(p['contrastive_dataset'], weights_only=False)
    if to_neighbors_dataset:  # Dataset returns a ts and one of its nearest neighbors.
        from data.custom_dataset import NeighborsDataset
        N_indices = np.load(p['topk_neighbors_train_path'])
        F_indices = np.load(p['bottomk_neighbors_train_path'])
        dataset = NeighborsDataset(dataloader.dataset, transform, N_indices, F_indices, p)

    return dataset

def get_val_dataset(p, train_transformations, val_transformations, sanomaly):
    if p['train_db_name'] == 'MSL' or p['train_db_name'] == 'SMAP':
        if p['fname'] == 'All':
            with open(os.path.join(MyPath.db_root_dir('msl'), 'labeled_anomalies.csv'), 'r') as file:
                csv_reader = pandas.read_csv(file, delimiter=',')
            data_info = csv_reader[csv_reader['spacecraft'] == p['train_db_name']]
            ii = 0
            for file_name in data_info['chan_id']:
                p['fname'] = file_name
                if ii == 0 :
                    train_dataset = get_train_dataset(p, train_transformations, sanomaly,
                                                    to_neighbors_dataset=True)
                    val_dataset = _get_val_dataset(p, val_transformations, sanomaly, True, train_dataset.mean,
                                                train_dataset.std)
                else:
                    new_train_dataset = get_train_dataset(p, train_transformations, sanomaly,
                                                    to_neighbors_dataset=True)
                    new_val_dataset = _get_val_dataset(p, val_transformations, sanomaly, True, new_train_dataset.mean,
                                                new_train_dataset.std)
                    val_dataset.concat_ds(new_val_dataset)
                    train_dataset.concat_ds(new_train_dataset)
                ii+=1
        else:
            #base_dataset = get_aug_train_dataset(p, train_transformations, to_neighbors_dataset=True)
            info_ds = get_train_dataset(p, train_transformations, sanomaly, to_neighbors_dataset=False)
            val_dataset = _get_val_dataset(p, val_transformations, sanomaly, False, info_ds.mean, info_ds.std)

    elif p['train_db_name'] == 'smd' or p['train_db_name'] == 'kpi' or p['train_db_name'] == 'swat' \
        or p['train_db_name'] == 'swan' or p['train_db_name'] == 'wadi':
        train_dataset = get_train_dataset(p, train_transformations, sanomaly, to_augmented_dataset=True)  # used only to mean and std
        dataset_mean = train_dataset.mean
        dataset_std = train_dataset.std

        val_dataset = _get_val_dataset(p, val_transformations, sanomaly, False, dataset_mean, dataset_std)
    else:
        raise ValueError('Invalid train dataset {}'.format(p['train_db_name']))
    return val_dataset, train_dataset


def _get_val_dataset(p, transform=None, sanomaly=None, to_neighbors_dataset=False,
                    mean_data=None, std_data=None, data=None, label=None):
    # Base dataset
    if p['val_db_name'] == 'MSL' or p['val_db_name'] == 'SMAP':
        from data.MSL import MSL
        dataset = MSL(p['fname'], train=False, transform=transform, sanomaly=sanomaly,
                      mean_data=mean_data, std_data=std_data, wsz=p['window_size'], stride=p['window_stride'])

    elif p['train_db_name'] == 'kpi':
        from data.KPI import KPI
        dataset = KPI(p['fname'], train=False, transform=transform, sanomaly=sanomaly,
                      mean_data=mean_data, std_data=std_data, wsz=p['window_size'], stride=p['window_stride'])

    elif p['val_db_name'] == 'smd':
        from data.SMD import SMD
        dataset = SMD(p['fname'], train=False, transform=transform, sanomaly=sanomaly,
                      mean_data=mean_data, std_data=std_data, wsz=p['window_size'], stride=p['window_stride'])

    elif p['val_db_name'] == 'swat':
        from data.SWAT import SWAT
        dataset = SWAT(p['fname'], train=False, transform=transform, sanomaly=sanomaly,
                      mean_data=mean_data, std_data=std_data, wsz=p['window_size'], stride=p['window_stride'])

    elif p['val_db_name'] == 'wadi':
        from data.WADI import WADI
        dataset = WADI(p['fname'], train=False, transform=transform, sanomaly=sanomaly,
                      mean_data=mean_data, std_data=std_data, wsz=p['window_size'], stride=p['window_stride'])

    else:
        raise ValueError('Invalid validation dataset {}'.format(p['val_db_name']))

    # Wrap into other dataset (__getitem__ changes) 
    if to_neighbors_dataset:  # Dataset returns a ts and one of its nearest neighbors.
        from data.custom_dataset import NeighborsDataset
        N_indices = np.load(p['topk_neighbors_val_path'])
        F_indices = np.load(p['bottomk_neighbors_val_path'])
        dataset = NeighborsDataset(dataset, transform, N_indices, F_indices, 5)  # Only use 5

    return dataset


def get_dataloader(p, dataset, pin_memory=True, drop_last=False, shuffle=False):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'],
                                       batch_size=p['batch_size'], pin_memory=pin_memory, collate_fn=collate_custom,
                                       drop_last=drop_last, shuffle=shuffle)


def inject_sub_anomaly(p):
    return SubAnomaly(p['anomaly_kwargs']['portion'])


def get_transformations(p):
    if p['augmentation_strategy'] == 'standard':
        # Standard augmentation strategy
        return transforms.Compose([
            transforms.RandomResizedCrop(**p['augmentation_kwargs']['random_resized_crop']),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize'])
        ])

    elif p['augmentation_strategy'] == 'ts':
        return transforms.Compose([
            NoiseTransformation(p['transformation_kwargs']['noise_sigma']),
        ])

    else:
        raise ValueError('Invalid augmentation strategy {}'.format(p['augmentation_strategy']))


def get_optimizer(p, model, cluster_head_only=False):
    if cluster_head_only:  # Only weights in the cluster head will be updated
        for name, param in model.named_parameters():
            if 'cluster_head' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert (len(params) == 2 * p['num_heads'])

    else:
        params = model.parameters()

    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params, **p['optimizer_kwargs'])

    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    return optimizer


def adjust_learning_rate(p, optimizer, epoch):
    lr = p['optimizer_kwargs']['lr']

    if p['scheduler'] == 'cosine':
        eta_min = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p['epochs'])) / 2

    elif p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'constant':
        lr = lr

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
