import os
import math
import numpy as np
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
    # if p['backbone'] == 'resnet18':
    #     return 8

    # elif p['backbone'] == 'resnet_ts':
    #     return 8

    # else:
    #     raise NotImplementedError
    return p['res_kwargs']['mid_channels'][-1]


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
            model.backbone.load_state_dict(state['backbone'])
            # missing = model.load_state_dict(state['model'], strict=False)
            # assert (set(missing[1]) == {
            #     'contrastive_head.0.weight', 'contrastive_head.0.bias',
            #     'contrastive_head.2.weight', 'contrastive_head.2.bias'}
            #         or set(missing[1]) == {
            #             'contrastive_head.weight', 'contrastive_head.bias'})
        else:
            raise NotImplementedError

    elif pretrain_path is not None and not os.path.exists(pretrain_path):
        raise ValueError('Path with pre-trained weights does not exist {}'.format(pretrain_path))

    else:
        pass

    return model

def load_backbone(p, model, pretrain_path):
    if os.path.exists(pretrain_path):
        state = torch.load(pretrain_path, map_location='cpu')

        if p['setup'] == 'classification':  # Weights are supposed to be transfered from contrastive training
            model.backbone.load_state_dict(state['backbone'])
            # missing = model.load_state_dict(state['model'], strict=False)
            # assert (set(missing[1]) == {
            #     'contrastive_head.0.weight', 'contrastive_head.0.bias',
            #     'contrastive_head.2.weight', 'contrastive_head.2.bias'}
            #         or set(missing[1]) == {
            #             'contrastive_head.weight', 'contrastive_head.bias'})

        else:
            raise NotImplementedError

    else:
        raise ValueError('Path with pre-trained weights does not exist {}'.format(pretrain_path))
    return

def _get_dataset(p, file, train, transform=None, sanomaly=None,
                 mean_data=None, std_data=None, logger=None):
    if p['train_db_name'] == 'MSL' or p['train_db_name'] == 'SMAP':
        from data.MSL import MSL
        dataset = MSL(file, train=train, transform=transform, sanomaly=sanomaly,
                      mean_data=mean_data, std_data=std_data,
                      wsz=p['window_size'], stride=p['window_stride'], logger=logger)
        mean, std = dataset.get_info()

    elif p['train_db_name'] == 'kpi':
        from data.KPI import KPI
        dataset = KPI(file, train=train, transform=transform, sanomaly=sanomaly,
                    mean_data=mean_data, std_data=std_data,
                      wsz=p['window_size'], stride=p['window_stride'], logger=logger)
        mean, std = dataset.get_info()

    elif p['train_db_name'] == 'smd':
        from data.SMD import SMD
        dataset = SMD(file, train=train, transform=transform, sanomaly=sanomaly,
                    mean_data=mean_data, std_data=std_data,
                      wsz=p['window_size'], stride=p['window_stride'], logger=logger)
        mean, std = dataset.get_info()

    elif p['train_db_name'] == 'swat':
        from data.SWAT import SWAT
        dataset = SWAT(file, train=train, transform=transform, sanomaly=sanomaly,
                    mean_data=mean_data, std_data=std_data,
                      wsz=p['window_size'], stride=p['window_stride'], logger=logger)
        mean, std = dataset.get_info()

    elif p['train_db_name'] == 'wadi':
        from data.WADI import WADI
        dataset = WADI(file, train=train, transform=transform, sanomaly=sanomaly,
                    mean_data=mean_data, std_data=std_data,
                      wsz=p['window_size'], stride=p['window_stride'], logger=logger)
        mean, std = dataset.get_info()

    else:
        raise ValueError(f'Invalid {"train" if train else "validation"} dataset {p[f'{"train" if train else "val"}_db_name']} of file :{file}')
    return dataset, mean, std

def get_dataset(p, train, transform=None, sanomaly=None, to_augmented_dataset=False,
                to_neighbors_dataset=False, mean_data=None, std_data=None, logger=None):
    if train:
        topk = p['num_neighbors'] if 'num_neighbors' in p.keys() else None
        assert(transform is not None)
        assert(sanomaly is not None)
        assert(mean_data is None)
        assert(std_data is None)
    else:
        topk = 5
        assert(to_augmented_dataset == False)

    if p['fname'].upper() == 'ALL':
        databese_dir = MyPath.db_root_dir(p[f'{"train" if train else "val"}_db_name'].lower())
        all_files = os.listdir(os.path.join(databese_dir, 'train'))
        file_list = [file for file in all_files if file.endswith('.txt')]
        file_list = sorted(file_list)
    else:
        file_list = [p['fname']]

    # Base dataset
    for idx, file in enumerate(file_list):
        if idx == 0:
            dataset, mean, std = _get_dataset(p, file, train, transform, sanomaly,
                                              mean_data, std_data, logger=logger)
            mean = [mean]
            std = [std]
        else:
            new_dataset, new_mean, new_std = _get_dataset(p, file, train, transform, sanomaly,
                                                          mean_data, std_data, logger=logger)
            dataset.concat_ds(new_dataset)
            mean.append(new_mean)
            std.append(new_std)

    dataset.mean = sum(mean)/len(mean)
    dataset.std = sum(std)/len(std)

    # Wrap into other dataset (__getitem__ changes)
    if to_augmented_dataset:  # Dataset returns a ts and an augmentation of that.
        from data.custom_dataset import AugmentedDataset
        preload = p['preload_aug'] if 'preload_aug' in p.keys() else True
        dataset = AugmentedDataset(dataset, preload=preload)

    if to_neighbors_dataset:  # Dataset returns ts and its nearest and furthest neighbors.
        from data.custom_dataset import NeighborsDataset
        nindices = np.load(p[f'topk_neighbors_{"train" if train else "val"}_path'])
        findices = np.load(p[f'bottomk_neighbors_{"train" if train else "val"}_path'])
        neighbor_transform = None if train else transform
        dataset = NeighborsDataset(dataset, neighbor_transform, nindices, findices, topk)

    return dataset

def get_aug_train_dataset(p, transform, to_neighbors_dataset=False):
    dataloader = torch.load(p['contrastive_dataloader'], weights_only=False)
    if to_neighbors_dataset:  # Dataset returns a ts and one of its nearest neighbors.
        from data.custom_dataset import NeighborsDataset
        N_indices = np.load(p['topk_neighbors_train_path'])
        F_indices = np.load(p['bottomk_neighbors_train_path'])
        dataset = NeighborsDataset(dataloader.dataset, transform, N_indices, F_indices, p['num_neighbors'])

    return dataset

def get_dataloader(p, dataset, pin_memory=True, train=False):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'],
                                       batch_size=p['batch_size'], pin_memory=pin_memory, collate_fn=collate_custom,
                                       drop_last=train, shuffle=train)

def inject_sub_anomaly(p, logger=None):
    return SubAnomaly(p['anomaly_kwargs']['portion'], logger=logger)

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
        for param in model.backbone.parameters():
            param.requires_grad = False
        params = model.head.parameters()
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
    warmup_epochs = p['scheduler_kwargs'].get('lr_warmup_epochs', 0)
    if epoch < warmup_epochs:
        lr = lr * (epoch / warmup_epochs)
    else:
        epoch -= warmup_epochs
        if p['scheduler'] == 'cosine':
            eta_min = p['scheduler_kwargs']['lr_eta_min']
            lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p['epochs'])) / 2
        elif p['scheduler'] == 'cosine_restart':
            eta_min = p['scheduler_kwargs']['lr_eta_min']
            cycle_period = p['scheduler_kwargs']['T_period']
            cycle_period_mul = p['scheduler_kwargs'].get('T_mul', 1)

            cycle = 0
            if epoch < cycle_period:
                cycle = 0
            else:
                epoch -= cycle_period
                cycle = 1
                cycle_period *= cycle_period_mul
                while True:
                    if epoch > cycle_period:
                        epoch -= cycle_period
                        cycle += 1
                        cycle_period *= cycle_period_mul
                    else:
                        break

            lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / cycle_period)) / 2
        elif p['scheduler'] == 'step':
            steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
            if steps > 0:
                lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)
        elif p['scheduler'] == 'constant':
            lr = lr
        elif p['scheduler'] == 'linear':
            lr = lr * (1 - epoch / p['epochs'])
        else:
            raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
