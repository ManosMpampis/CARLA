import os
import random

import torch
from torchmetrics.functional import precision_recall_curve, confusion_matrix
import numpy as np

from utils.config import create_config
from utils.common_config import get_criterion, get_model, get_train_dataset,\
                                get_val_dataset, get_train_dataloader,\
                                get_val_dataloader, get_train_transformations,\
                                get_val_transformations, get_optimizer,\
                                adjust_learning_rate, inject_sub_anomaly
from utils.evaluate_utils import get_predictions, classification_evaluate, contrastive_evaluate
from utils.repository import TSRepository, fill_ts_repository
from utils.train_utils import self_sup_classification_train, pretext_train
from utils.utils import Logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(4)



class CARLA:
    def __init__(self, config_env, config_exp, fname, device, verbose, tensorboard, version=None):
        self.p = create_config(config_env, config_exp, fname, version)

        # Init logger
        file_path = self.p["classification_dir"] if self.p["setup"] == "classification" else self.p["pretext_dir"]
        self.logger = Logger(self.p['version'], verbose=verbose, file_path=file_path, use_tensorboard=tensorboard)

        # Init device
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
        else:
            self.device = torch.device(device, index=0)

        self.autocast_available = ...

        
        self.model = get_model(self.p)

        self.logger.add_graph(self.model, torch.rand(self.p['res_kwargs']['in_channels'], self.p['window_size']).unsqueeze(0))

        self.model.to(self.device)
        self.majority_label = ...        

    def train_pretext(self):
        self.logger.log('CARLA Pretext stage --> ')

        # Init transforms
        train_transforms = get_train_transformations(self.p)
        sanomaly = inject_sub_anomaly(self.p)
        
        # Init dataset
        self.logger.log(f'\n- Get dataset for {self.p['train_db_name']} dataset - timeseries {self.p['fname']}.')
        train_dataset = get_train_dataset(self.p, train_transforms, sanomaly, to_augmented_dataset=True)
        train_dataloader = get_train_dataloader(self.p, train_dataset)
        self.logger.log(f'Dataset contains {len(train_dataset)} train samples')

        # Init train optimizer, loss, AMP
        # Check if user has already load checkpoint
        if not hasattr(self, 'optimizer'):
            self.optimizer = get_optimizer(self.p, self.model)
        criterion = get_criterion(self.p)
        criterion = criterion.to(self.device)

        self.scaler = ...

        self.logger.log('\n- Model Initialisation')
        # Checkpoint
        if os.path.exists(f"{self.p['pretext_checkpoint_last']}"):
            self.logger.log(f"Restart from checkpoint {self.p['pretext_checkpoint_last']}")
            self.load(type="pretext", checkpoint=True, tag="last")
        elif os.path.exists(self.p['pretext_checkpoint']):
            self.logger.log(f"Restart from checkpoint {self.p['pretext_checkpoint']}")
            self.load(type="pretext", checkpoint=True)
        else:
            self.logger.log(f'-- No checkpoint file at {self.p["pretext_checkpoint"]} -- new model initialised')
            self.start_epoch = 0
            self.epoch = 0
            self.pretext_best_loss = float("inf")
            self.pretext_previous_loss = torch.tensor(float(0), device=self.device)
        
        self.logger.log('\n- Training:')        
        end_epoch = self.p['epochs']
        for epoch in range(self.start_epoch, self.p['epochs']):
            self.epoch = epoch
            self.logger.log(f'-- Epoch {epoch+1}/{end_epoch}')
            self.logger.log('-'*15)

            lr = adjust_learning_rate(self.p, self.optimizer, epoch)
            self.logger.log(f'Adjusted learning rate to {lr:.5f}')
            
            # Train epoch
            tmp_loss_dict = pretext_train(train_dataloader, self.model, criterion, self.optimizer, epoch, self.pretext_previous_loss, device=self.device)
            self.logger.metrics_summary("Pretext Loss", tmp_loss_dict, self.epoch)

            # Evaluation
            feats, metadata, evaluation_metrics = contrastive_evaluate(train_dataloader, self.model)
            self.logger.add_embedding("Cluster", feats, metadata, self.epoch)
            self.logger.metrics_summary("Pretext Evaluation", evaluation_metrics, self.epoch)

            # Checkpoint
            if tmp_loss_dict['loss'] <= self.pretext_best_loss:
                self.pretext_best_loss = tmp_loss_dict['loss']
                self.save(type="pretext", checkpoint=True)
                self.save(type="pretext", checkpoint=False)

        #Save last checkpoint in order to restart training
        self.save(type="pretext", checkpoint=True, tag='last')

        #Load best checkpoint
        self.load(type="pretext", checkpoint=True)
        self.logger.log(f'Best Model saved from epoch: {self.epoch}')

        self.makeTSRepository(train_dataset, sanomaly)

    def train_classification(self):
        pass

    @torch.no_grad()
    def evaluate_classification(self):
        pass

    @torch.no_grad()
    def inference(self):
        pass
    
    def load(self, path=None, type="classification", checkpoint=False, tag=None):
        if path is None:
            assert type in ["classification", "pretext"]
            key = f"{type}_{"checkpoint" if checkpoint else "model"}"
            path = self.p[key]

            tag = f"_{tag}" if tag else ""
            path = f"{path[:-8]}{tag}{path[-8:]}"
        
        self.logger.log(f'-- Model initialised from {"last checkpoint" if checkpoint else "model path"}: {path}')
        dictionary = torch.load(path, map_location='cpu')
        self.model.backbone.load_state_dict(dictionary['backbone'])
        self.model.head.load_state_dict(dictionary['head'])

        if checkpoint: 
            if not hasattr(self, 'optimizer'):
                self.optimizer = get_optimizer(self.p, self.model, self.p['update_cluster_head_only'])
            self.optimizer.load_state_dict(dictionary['optimizer'])
            self.start_epoch = dictionary['epoch']
            self.epoch = dictionary['epoch']
            if type == "classification":
                self.majority_label = dictionary['majority_label']
            if type == "pretext":
                self.pretext_best_loss = dictionary['pretext_best_loss']
                self.pretext_previous_loss = dictionary['pretext_previous_loss'].to(self.device, non_blocking=True)

    def save(self, path=None, dictionary=None, type="classification", checkpoint=False, tag=None):
        assert type in ["classification", "pretext"]
        if path is None:
            key = f"{type}_{"checkpoint" if checkpoint else "model"}"
            path = self.p[key]

            tag = f"_{tag}" if tag else ""
            path = f"{path[:-8]}{tag}{path[-8:]}"
        
        
        if dictionary is None:
            dictionary = {'backbone': self.model.backbone.state_dict(), 'head': self.model.head.state_dict()}
            if type == "classification":
                dictionary['majority_label'] = self.majority_label

            if checkpoint:
                dictionary['optimizer'] = self.optimizer.state_dict()
                dictionary['epoch'] = self.epoch + 1

                if type == "pretext":
                    dictionary['pretext_best_loss'] = self.pretext_best_loss
                    dictionary['pretext_previous_loss'] = self.pretext_previous_loss
                
        torch.save(dictionary, path)
        return

    def export(self):
        pass

    def makeTSRepository(self, train_dataset, sanomaly, topk=10):
        memmory_efficient = self.p['fname'].upper() == 'ALL'
        use_fneighbors = self.p.get('use_fneighbors_in_repository', False)

        base_dataloader = get_val_dataloader(self.p, train_dataset)
        
        # TS Repository
        ts_repository_base = TSRepository(len(train_dataset), self.p['model_kwargs']['features_dim'])
        ts_repository_aug = TSRepository(len(train_dataset) * 2, self.p['model_kwargs']['features_dim'])
        ts_repository_base.to(self.device)
        
        # Mine the topk nearest neighbors at the very end (Train)
        # These will be served as input to the classification loss.
        self.logger.log('Fill TS Repository for mining the nearest/furthest neighbors (train) ...')
        
        fill_ts_repository(self.p, base_dataloader, self.model, ts_repository_base, real_aug = True, ts_repository_aug = ts_repository_aug)
        
        
        self.logger.log('Mine the nearest neighbors (Top-%d)' %(topk))
        kfurtherst, knearest = ts_repository_aug.furthest_nearest_neighbors(topk)

        np.save(self.p['topk_neighbors_train_path'], knearest)
        np.save(self.p['bottomk_neighbors_train_path'], kfurtherst)

        # Mine the topk nearest neighbors at the very end (Val)
        # These will be used for validation.
        self.logger.log('Fill TS Repository for mining the nearest/furthest neighbors (val) ...')

        val_transforms = get_val_transformations(self.p)
        val_dataset = get_val_dataset(self.p, val_transforms, sanomaly, False, train_dataset.mean, train_dataset.std)
        val_dataloader = get_val_dataloader(self.p, val_dataset)
        ts_repository_val = TSRepository(len(val_dataset), self.p['model_kwargs']['features_dim'])
        ts_repository_val.to(self.device)

        fill_ts_repository(self.p, val_dataloader, self.model, ts_repository_val, real_aug=False, ts_repository_aug=None)

        self.logger.log('Mine the nearest and furthest neighbors (Top-%d)' %(topk))
        kfurtherst, knearest = ts_repository_val.furthest_nearest_neighbors(topk)
        np.save(self.p['topk_neighbors_val_path'], knearest)
        np.save(self.p['bottomk_neighbors_val_path'], kfurtherst)
