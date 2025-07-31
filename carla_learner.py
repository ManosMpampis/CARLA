import os
import random

import torch
from torchmetrics.functional import precision_recall_curve
from torchmetrics.functional.classification import confusion_matrix
from sklearn import metrics
import numpy as np

from utils.config import create_config
from utils.common_config import get_transformations, get_aug_train_dataset,\
                                get_val_dataset, get_dataset, get_dataloader,\
                                get_optimizer, get_model, load_pretext_backbone_to_model,\
                                get_criterion, adjust_learning_rate, inject_sub_anomaly
from utils.evaluate_utils import get_predictions, classification_evaluate, pr_evaluate
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

        self.p = create_config(config_env, config_exp, fname, version)
        self.version = self.p['version']

        self.verbose = verbose
        self.file_path=self.p['experiment_dir']

        self.logger = Logger(self.p['version'], verbose=verbose, file_path=self.file_path,
                             use_tensorboard=tensorboard, file_name=self.p['setup'])

        self.model = get_model(self.p)
        self.logger.add_graph(self.model, torch.rand(self.p['res_kwargs']['in_channels'], self.p['window_size']).unsqueeze(0))
        
        self.model = self.model.to(self.device)

        self.majority_label = torch.tensor(0, dtype=torch.long, device=self.device)
    
    def train_pretext(self):
        self.logger.log('CARLA Pretext stage --> ')
        self.logger.log_hyperparams(self.p)

        # Data
        self.logger.log(f"\n- Get dataset and dataloaders for {self.p['train_db_name']} dataset - timeseries {self.p['fname']}")
        train_transforms = get_transformations(self.p)
        val_transforms = get_transformations(self.p)
        sanomaly = inject_sub_anomaly(self.p)
        
        train_dataset = get_dataset(self.p, train=True, transform=train_transforms, sanomaly=sanomaly, to_augmented_dataset=True)  # used only to mean and std
        
        # val_dataset, train_dataset = get_val_dataset(self.p, train_transforms, val_transforms, sanomaly)
        train_dataloader = get_dataloader(self.p, train_dataset, drop_last=True, shuffle=True)

        self.logger.log(f'Dataset contains {len(train_dataset)} train samples')

        # Optimizer
        if not hasattr(self, 'optimizer'):
                self.optimizer = get_optimizer(self.p, self.model, False)

        # Loss function
        criterion = get_criterion(self.p)
        criterion = criterion.to(self.device)

        self.logger.log('\n- Model initialisation')
        # Checkpoint
        if os.path.exists(self.p['pretext_checkpoint']):
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
        for epoch in range(self.start_epoch, end_epoch):
            self.epoch = epoch
            self.logger.log(f'-- Epoch {epoch+1}/{end_epoch}')
            self.logger.log('-'*15)

            lr = adjust_learning_rate(self.p, self.optimizer, self.epoch)
            self.logger.log(f'Adjusted learning rate to {lr:.5f}')
            self.logger.scalar_summary('Train Pretex', 'learning_rate', lr, self.epoch)


            tmp_loss_dict = pretext_train(train_dataloader, self.model, criterion, self.optimizer, self.epoch, self.pretext_previous_loss, self.logger)
            for loss, value in tmp_loss_dict.items():
                self.logger.scalar_summary("Train Pretex", loss, value, self.epoch)

            # Checkpoint
            if tmp_loss_dict['Total Loss'] <= self.pretext_best_loss:
                self.pretext_best_loss = tmp_loss_dict['Total Loss']
                self.save(type="pretext", checkpoint=True)
                self.save(type="pretext", checkpoint=False)

        self.load(type="pretext", checkpoint=False)
        self.save(type="pretext", checkpoint=True)
        self.save(type="pretext", checkpoint=False)
        self.makeTSRepository(train_dataset, val_transforms, sanomaly)
        # Make new repository of time series for the second stage.
        
    def train_classification(self):
        self.logger.log('CARLA Self-supervised Classification stage --> ')
        self.logger.log_hyperparams(self.p)

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
            load_pretext_backbone_to_model(self.p, self.model, self.p['pretext_model'])
            self.start_epoch = 0
            self.majority_label = torch.tensor(0, dtype=torch.long, device=self.device)

        best_f1 = -1 * torch.tensor(float("inf"), device=self.device)
        self.logger.log('\n- Training:')
        
        # import time
        end_epoch = self.p['epochs']
        for epoch in range(self.start_epoch, end_epoch):
            self.epoch = epoch
            self.logger.log(f'-- Epoch {epoch+1}/{end_epoch}')

            lr = adjust_learning_rate(self.p, self.optimizer, self.epoch)
            self.logger.scalar_summary('Train Classification', 'learning_rate', lr, self.epoch)

            tmp_loss_dict = self_sup_classification_train(train_dataloader, self.model, criterion, self.optimizer, self.epoch,
                                          self.p['update_cluster_head_only'], self.logger)
            for loss, value in tmp_loss_dict.items():
                self.logger.scalar_summary("Train Classification", loss, value, self.epoch)

            if (epoch == self.p['epochs']-1):
                predictions, _ = get_predictions(self.p, tst_dataloader, self.model, True, True)
            else:
                predictions = get_predictions(self.p, tst_dataloader, self.model, False, False)

            label_counts = torch.bincount(predictions['predictions'])
            self.majority_label = label_counts.argmax()

            classification_losses = classification_evaluate(predictions, **self.p['criterion_kwargs'])

            metrics = self.eval_classification(val_dataloader)
            rep_f1 = metrics["rep_f1"]

            for metric, value in metrics.items():
                self.logger.scalar_summary("Train Val Dataset Evaluation", metric, value, self.epoch)
            
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
    
    @torch.no_grad()
    def eval_classification(self, dataloader, train=False, class_names=None):
        experiment = "Train" if train else "Val"

        predictions = get_predictions(self.p, dataloader, self.model)

        targets = predictions['targets']
        probs = predictions['probabilities']

        scores = 1-probs[:, self.majority_label]

        precision, recall, thresholds = precision_recall_curve(scores, targets, task='binary')
        
        try:
            f1_score = 2*precision*recall / (precision+recall)
            if torch.isnan(f1_score).any():
                f1_score = torch.nan_to_num(f1_score)
                self.logger.log('f1: Nan --> 0')     
        except ZeroDivisionError:
            f1_score = [0.0]
            self.logger.log('f1: 0 --> 0')

        best_f1_index = torch.argmax(f1_score)

        rep_f1 = f1_score[best_f1_index]

        if class_names=='Anom':
            best_threshold = thresholds[best_f1_index]
            anomalies = [1 if s >= best_threshold else 0 for s in scores]
            best_tn, best_fp, best_fn, best_tp = confusion_matrix(targets, anomalies).ravel()
            self.logger.log(f"Anomalies --> TP: {best_tp}, TN: {best_tn}, FN: {best_fn}, FP: {best_fp}")
            self.logger.log(f"Mazority label: {self.majority_label}")
            self.logger.log(metrics.classification_report(targets, anomalies))
            return {"rep_f1": rep_f1, "best_tp": best_tp, "best_tn": best_tn, "best_fn": best_fn, "best_fp": best_fp}

        return {"rep_f1": rep_f1}
    
    @torch.no_grad()
    def inference(self, ts):
        output = self.model(ts, forward_pass='return_all')
        prediction = torch.argmax(output, dim=1)
        
        probs = torch.nn.functional.softmax(output, dim=1)
        anomalus_score = 1 - probs[:,   self.majority_label]
        self.logger.log(f"Prediction is {'anomaly' if prediction == self.majority_label else 'normal'}\n\
                        With anomalus score: {anomalus_score.item()}")
        return prediction

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
            self.epoch = checkpoint['epoch']
            if type == "classification":
                self.majority_label = checkpoint['normal_label']
            if type == "pretext":
                self.pretext_best_loss = checkpoint['pretext_best_loss']
                self.pretext_previous_loss = checkpoint['pretext_previous_loss'].to(self.device, non_blocking=True)

    def save(self, path=None, dictionary=None, type="classification", checkpoint=False):
        assert type in ["classification", "pretext"]
        if path is None:
            key = f"{type}_{"checkpoint" if checkpoint else "model"}"
            path = self.p[key]
        
        
        if dictionary is None:
            if type == "classification":
                dictionary = {'model': self.model.state_dict(), 'majority_label': self.majority_label}
            else:
                dictionary = {'model': self.model.state_dict()}
            
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

    def makeTSRepository(self, dataset, val_transformations, sanomaly, topk=10):
        base_dataloader = get_dataloader(self.p, dataset)
        ts_repository_aug = TSRepository(len(dataset), self.p['model_kwargs']['features_dim'],
                                         self.p['num_classes'], self.p['criterion_kwargs']['temperature'], need_fneighbors=True)  # need size of repository == 1+num_of_anomalies
        ts_repository_base = TSRepository(len(dataset), self.p['model_kwargs']['features_dim'],
                                          self.p['num_classes'], self.p['criterion_kwargs']['temperature'])
        # ts_repository_val = TSRepository(len(val_dataset), self.p['model_kwargs']['features_dim'],
        #                                  self.p['num_classes'], self.p['criterion_kwargs']['temperature'])
        # ts_repository_aug.to(self.device, non_blocking=True)
        # ts_repository_base.to(self.device, non_blocking=True)
        # ts_repository_val.to(self.device, non_blocking=True)
        
        # Mine the topk nearest neighbors at the very end (Train)
        # These will be served as input to the classification loss.
        self.logger.log('Fill TS Repository for mining the nearest/furthest neighbors (train) ...')    
        fill_ts_repository(self.p, base_dataloader, self.model, ts_repository_base, real_aug = True, ts_repository_aug = ts_repository_aug, logger=self.logger)
        out_pre = np.column_stack((ts_repository_base.features.cpu().numpy(), ts_repository_base.targets.cpu().numpy()))

        self.logger.log(f'Mine the nearest neighbors (Top-{topk})')
        kfurtherst, knearest = ts_repository_aug.furthest_nearest_neighbors(topk, use_algorithm=1)  # hear we mine from aug and not base

        np.save(self.p['pretext_features_train_path'], out_pre)
        np.save(self.p['topk_neighbors_train_path'], knearest)
        np.save(self.p['bottomk_neighbors_train_path'], kfurtherst)
        mean, std = dataset.get_info()

        del dataset
        del ts_repository_aug
        del ts_repository_base

        dataset = get_dataset(self.p, train=False, transform=val_transformations, sanomaly=sanomaly,
                                  to_augmented_dataset=False, mean_data=mean, std_data=std)
        base_dataloader = get_dataloader(self.p, dataset)
        ts_repository_base = TSRepository(len(dataset), self.p['model_kwargs']['features_dim'],
                                          self.p['num_classes'], self.p['criterion_kwargs']['temperature'])
        # Mine the topk nearest neighbors at the very end (Val)
        # These will be used for validation.
        self.logger.log('Fill TS Repository for mining the nearest/furthest neighbors (val) ...')

        fill_ts_repository(self.p, base_dataloader, self.model, ts_repository_base, real_aug = False, ts_repository_aug = None, logger=self.logger)
        out_pre = np.column_stack((ts_repository_base.features.cpu().numpy(), ts_repository_base.targets.cpu().numpy()))

        self.logger.log(f'Mine the nearest neighbors (Top-{topk})')
        kfurtherst, knearest = ts_repository_base.furthest_nearest_neighbors(topk, use_algorithm=0)  # hear we mine from val
        
        np.save(self.p['pretext_features_test_path'], out_pre)
        np.save(self.p['topk_neighbors_val_path'], knearest)
        np.save(self.p['bottomk_neighbors_val_path'], kfurtherst)
    
    def close(self):
        self.logger.finalize()