import argparse
import os
import random

import torch
from torch.utils.tensorboard import SummaryWriter 
import pandas
import numpy as np

from utils.mypath import MyPath
from utils.config import create_config
from utils.common_config import get_transformations, get_train_dataset,\
                                get_aug_train_dataset,\
                                get_val_dataset, get_dataloader,\
                                get_optimizer, get_model, get_criterion,\
                                adjust_learning_rate, inject_sub_anomaly
from utils.evaluate_utils import get_predictions, classification_evaluate, pr_evaluate
from utils.repository import TSRepository
from utils.train_utils import self_sup_classification_train, pretext_train
from utils.utils import fill_ts_repository, log, Logger

class CARLA:
    def __init__(self, config_env, config_exp, fname, device, verbose, tensorboard):
        """_summary_

        Args:
            config_env (_type_): _description_
            config_exp (_type_): _description_
            fname (_type_): _description_
            device (_type_): _description_
            verbose (_type_): _description_
            tensorboard (_type_): _description_
        """

        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.p = create_config(config_env, config_exp, fname)

        self.verbose = verbose
        self.file_path=os.path.join(self.p['experiment_dir'], "classification.txt") if verbose>=2 else None
        self.verbose_dict={"verbose": verbose, "file_path": self.file_path}

        self.logger = Logger(verbose=verbose, file_path=self.file_path, use_tensorboard=tensorboard)

        self.model = get_model(self.p, self.p['pretext_model'])
        self.model = self.model.to(self.device)

        self.mazority_label = torch.tensor(0, dtype=torch.long, device=self.device)
    
    def train_pretext(self):
        self.logger.log('CARLA Pretext stage --> ')

        # Data
        self.logger.log(f"\n- Get dataset and dataloaders for {self.p['train_db_name']} dataset - timeseries {self.p['fname']}")
        train_transforms = get_transformations(self.p)
        val_transforms = get_transformations(self.p)
        sanomaly = inject_sub_anomaly(self.p)
        
        val_dataset, train_dataset = get_val_dataset(self.p, train_transforms, val_transforms, sanomaly)
        train_dataloader = get_dataloader(self.p, train_dataset, drop_last=True, shuffle=True)
        base_dataloader = get_dataloader(self.p, train_dataset)
        val_dataloader = get_dataloader(self.p, val_dataset)

        self.logger.log(f'Dataset contains {train_dataset}/{val_dataset} train/val samples')

        # Optimizer
        if not hasattr(self, 'optimizer'):
                self.optimizer = get_optimizer(self.p, self.model, False)

        # Loss function
        criterion = get_criterion(self.p)
        criterion = criterion.to(self.device)

        self.logger.log('\n- Model initialisation')
        # Checkpoint
        if os.path.exists(self.p['classification_checkpoint']):
            self.logger.log(f"Restart from checkpoint {self.p['pretext_checkpoint']}")
            self.load(checkpoint=True)
        else:
            self.logger.log(f'-- No checkpoint file at {self.p["pretext_checkpoint"]} -- new model initialised')
            start_epoch = 0
            self.pretext_best_loss = torch.tensor(float("inf"), device=self.device)
            self.pretext_previous_loss = torch.tensor(float(0), device=self.device)

        self.logger.log('\n- Training:')        
        end_epoch = self.p['epochs']
        for epoch in range(start_epoch, end_epoch):
            self.logger.log(f'-- Epoch {epoch+1}/{end_epoch}')
            self.logger.log('-'*15)

            self.logger.log(f'Adjusted learning rate to {lr:.5f}')
            lr = adjust_learning_rate(self.p, self.optimizer, self.epoch)

            tmp_loss = pretext_train(train_dataloader, self.model, criterion, self.optimizer, epoch, self.pretext_previous_loss, self.logger)
            
            # Checkpoint
            if tmp_loss <= self.pretext_best_loss:
                self.pretext_best_loss = tmp_loss
                self.save(type="pretext", checkpoint=True)
                self.save(type="pretext", checkpoint=False)

        self.load(type="pretext", checkpoint=False)
        self.save(type="classification", checkpoint=True)
        self.save(type="classification", checkpoint=False)

        # Make new repository of time series for the second stage.
        

    def train_classification(self):

        self.logger.log('CARLA Self-supervised Classification stage --> ')

        # Data
        self.logger.log(f"\n- Get dataset and dataloaders for {self.p['train_db_name']} dataset - timeseries {self.p['fname']}")
        train_transformations = get_transformations(self.p)
        val_transformations = get_transformations(self.p)
        sanomaly = inject_sub_anomaly(self.p)
        
        # In the self-supervised state we use as data the triplets with saves anchors of the first stage
        train_dataset = get_aug_train_dataset(self.p, train_transformations, to_neighbors_dataset = True)
        train_dataloader = get_dataloader(self.p, train_dataset, drop_last=True, shuffle=True)
        # In order to correctly measure the similarity matrics, all values need to be checkes,
        # during train we have drop_last so we add second dataloader
        tst_dataloader = get_dataloader(self.p, train_dataset)

        val_dataset, artifact = get_val_dataset(self.p, train_transformations, val_transformations, sanomaly)
        del artifact

        val_dataloader = get_dataloader(self.p, val_dataset)

        self.logger.log(f'-- Train samples size: {len(train_dataset)} - Test samples size: {len(val_dataset)}')

        # Optimizer
        if not hasattr(self, 'optimizer'):
                self.optimizer = get_optimizer(self.p, self.model, self.p['update_cluster_head_only'])

        # Warning
        if self.p['update_cluster_head_only']:
            self.logger.log('WARNING: classification will only update the cluster head')

        # Loss function
        criterion = get_criterion(self.p)
        criterion = criterion.to(self.device)

        self.logger.log('\n- Model initialisation')
        # Checkpoint
        if os.path.exists(self.p['classification_checkpoint']):
            self.logger.log(f'-- Model initialised from last checkpoint: {self.p['classification_checkpoint']}')
            self.load(checkpoint=True)
        else:
            self.logger.log(f'-- No checkpoint file at {self.p["classification_checkpoint"]} -- new model initialised')
            start_epoch = 0
            self.mazority_label = torch.tensor(0, dtype=torch.long, device=self.device)

        best_f1 = -1 * torch.tensor(float("inf"), device=self.device)
        self.logger.log('\n- Training:')
        
        # import time
        end_epoch = self.p['epochs']
        for epoch in range(start_epoch, end_epoch):
            self.logger.log(f'-- Epoch {epoch+1}/{end_epoch}')

            lr = adjust_learning_rate(self.p, self.optimizer, self.epoch)
            
            # torch.cuda.synchronize()
            # start_time = time.time()

            self_sup_classification_train(train_dataloader, self.model, criterion, self.optimizer, epoch,
                                          self.p['update_cluster_head_only'], self.logger)
            
            
            # torch.cuda.synchronize()
            # end_time = time.time()
            # inference_time = end_time - start_time

            # print(f'Inference time with gpu sync: {inference_time:.4f} sencods')

            # torch.cuda.synchronize()
            # start_time = time.time()
            if (epoch == self.p['epochs']-1):
                predictions, _ = get_predictions(self.p, tst_dataloader, self.model, True, True)
            else:
                predictions = get_predictions(self.p, tst_dataloader, self.model, False, False)

            # torch.cuda.synchronize()
            # end_time = time.time()
            # inference_time = end_time - start_time

            # print(f'Get prediction tst time with gpu sync: {inference_time:.4f} sencods')
            # torch.cuda.synchronize()
            # start_time = time.time()
            label_counts = torch.bincount(predictions[0]['predictions'])
            self.mazority_label = label_counts.argmax()

            classification_stats = classification_evaluate(predictions, **p['criterion_kwargs'])
            lowest_loss_head = classification_stats['lowest_loss_head']
            lowest_loss = classification_stats['lowest_loss']

            # torch.cuda.synchronize()
            # end_time = time.time()
            # inference_time = end_time - start_time

            # print(f'Class evaluate time with gpu sync: {inference_time:.4f} sencods')
            # torch.cuda.synchronize()
            # start_time = time.time()
            predictions = get_predictions(self.p, val_dataloader, self.model, False, False)

            rep_f1 = pr_evaluate(predictions, majority_label=self.mazority_label)
            # torch.cuda.synchronize()
            # end_time = time.time()
            # inference_time = end_time - start_time
            # print(f'Evaluate time with gpu sync: {inference_time:.4f} sencods')
            if rep_f1 > best_f1:
                best_f1 = rep_f1
                # log('New Checkpoint ...', verbose=verbose, file_path=file_path)
                self.save(type="classification", checkpoint=True)
                self.save(type="classification", checkpoint=False)

        self.load(type="classification", checkpoint=False)
        self.save(type="classification", checkpoint=True)
        self.save(type="classification", checkpoint=False)

        # find final prediction in train set.
        predictions, _ = get_predictions(self.p, tst_dataloader, self.model, True, True)
        # and in val set
        predictions, _ = get_predictions(self.p, val_dataloader, self.model, True, False)

    def test(self):
        pass

    def eval(self):
        pass

    def inference(self):
        pass

    def load(self, path=None, type="classification", checkpoint=False):
        if path is None:
            assert type in ["classification", "pretext"]
            key = f"{type}_{"checkpoint" if checkpoint else "model"}"
            path = self.p[key]
        
        self.logger.log(f'-- Model initialised from {"last checkpoint" if checkpoint else "model path"}: {path}')
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])

        if "optimizer" in checkpoint.keys(): 
            if not hasattr(self, 'optimizer'):
                self.optimizer = get_optimizer(self.p, self.model, self.p['update_cluster_head_only'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = checkpoint['epoch']
            if type == "classification":
                self.mazority_label = checkpoint['normal_label']
            if type == "pretext":
                self.pretext_best_loss = checkpoint['pretext_best_loss'].to(self.device, non_blocking=True)
                self.pretext_previous_loss = checkpoint['pretext_previous_loss'].to(self.device, non_blocking=True)

    def save(self, path=None, dictionary=None, type="classification", checkpoint=False):
        assert type in ["classification", "pretext"]
        if path is None:
            key = f"{type}_{"checkpoint" if checkpoint else "model"}"
            path = self.p[key]
        
        
        if dictionary is None:
            if type == "classification":
                dictionary = {{'model': self.model.state_dict(), 'self.mazority_label': self.nomral_label}}
            else:
                dictionary = {{'model': self.model.state_dict()}}
            
            if checkpoint:
                dictionary['optimizer'] = self.optimizer.state_dict()
                dictionary['epoch'] = self.epoch

                if type == "pretext":
                    dictionary['pretext_best_loss'] = self.pretext_best_loss
                    dictionary['pretext_previous_loss'] = self.pretext_previous_loss
                
        torch.save(dictionary, path)
        return

    def export(self):
        pass

    def makeTSRepository(self, train_dataset, base_dataloader, val_dataset,topk=10):
        ts_repository_aug = TSRepository(len(train_dataset) * 2, self.p['model_kwargs']['features_dim'],
                                         self.p['num_classes'], self.p['criterion_kwargs']['temperature'])  # need size of repository == 1+num_of_anomalies
        ts_repository_base = TSRepository(len(train_dataset), self.p['model_kwargs']['features_dim'],
                                          self.p['num_classes'], self.p['criterion_kwargs']['temperature'])
        ts_repository_val = TSRepository(len(val_dataset), self.p['model_kwargs']['features_dim'],
                                         self.p['num_classes'], self.p['criterion_kwargs']['temperature'])
        ts_repository_aug.to(self.device, non_blocking=True)
        ts_repository_base.to(self.device, non_blocking=True)
        ts_repository_val.to(self.device, non_blocking=True)
        
        # Mine the topk nearest neighbors at the very end (Train)
        # These will be served as input to the classification loss.
        self.logger.log('Fill TS Repository for mining the nearest/furthest neighbors (train) ...')    
        fill_ts_repository(self.p, base_dataloader, self.model, ts_repository_base, real_aug = True, ts_repository_aug = ts_repository_aug)
        out_pre = np.column_stack((ts_repository_base.features.cpu().numpy(), ts_repository_base.targets.cpu().numpy()))

        self.logger.log(f'Mine the nearest neighbors (Top-{topk})')
        kfurtherst, knearest = ts_repository_aug.furthest_nearest_neighbors(topk)  # hear we mine from aug and not base
        
        np.save(self.p['pretext_features_train_path'], out_pre)
        np.save(self.p['topk_neighbors_train_path'], knearest)
        np.save(self.p['bottomk_neighbors_train_path'], kfurtherst)

        # Mine the topk nearest neighbors at the very end (Val)
        # These will be used for validation.
        self.logger.log('Fill TS Repository for mining the nearest/furthest neighbors (val) ...')    
        fill_ts_repository(self.p, val_dataset, self.model, ts_repository_val, real_aug = True, ts_repository_aug = ts_repository_aug)
        out_pre = np.column_stack((ts_repository_val.features.cpu().numpy(), ts_repository_val.targets.cpu().numpy()))

        self.logger.log(f'Mine the nearest neighbors (Top-{topk})')
        kfurtherst, knearest = ts_repository_val.furthest_nearest_neighbors(topk)  # hear we mine from val
        
        np.save(self.p['pretext_features_test_path'], out_pre)
        np.save(self.p['topk_neighbors_val_path'], knearest)
        np.save(self.p['bottomk_neighbors_val_path'], kfurtherst)