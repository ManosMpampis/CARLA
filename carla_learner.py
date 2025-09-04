import os
import random

import torch
from torchmetrics.functional import precision_recall_curve, confusion_matrix, Kmeans
import numpy as np

from utils.config import create_config
from utils.common_config import get_transformations, get_aug_train_dataset,\
                                get_dataset, get_dataloader,\
                                get_optimizer, get_model, load_backbone,\
                                get_criterion, adjust_learning_rate, inject_sub_anomaly
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
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
        else:
            self.device = torch.device(device, index=0)

        self.autocast_available = torch.amp.autocast_mode.is_autocast_available(self.device.type)

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
        sanomaly = inject_sub_anomaly(self.p, self.logger)
        
        train_dataset = get_dataset(self.p, train=True, transform=train_transforms,
                                    sanomaly=sanomaly, to_augmented_dataset=True,
                                    logger=self.logger)  # used only to mean and std
        
        # val_dataset, train_dataset = get_val_dataset(self.p, train_transforms, val_transforms, sanomaly)
        train_dataloader = get_dataloader(self.p, train_dataset, train=True)

        self.logger.log(f'Dataset contains {len(train_dataset)} train samples')

        # Optimizer
        if not hasattr(self, 'optimizer'):
            self.optimizer = get_optimizer(self.p, self.model, False)

        # Automatic Mixed Precision
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=(self.p.get('amp', False) and self.autocast_available))
        
        # Loss function
        criterion = get_criterion(self.p)
        criterion = criterion.to(self.device)

        self.logger.log('\n- Model initialisation')
        # Checkpoint
        if os.path.exists(f"{self.p['pretext_checkpoint_last']}"):
            self.logger.log(f"Restart from checkpoint {self.p['pretext_checkpoint']}")
            self.load(type="pretext", checkpoint=True, tag='last')
        elif os.path.exists(f"{self.p['pretext_checkpoint']}"):
            self.logger.log(f"Restart from checkpoint {self.p['pretext_checkpoint']}")
            self.load(type="pretext", checkpoint=True)
        else:
            self.logger.log(f'-- No checkpoint file at {self.p["pretext_checkpoint"]} -- new model initialised')
            self.start_epoch = 0
            self.epoch = 0
            self.pretext_best_loss = float("inf")
            self.pretext_previous_loss = torch.tensor(float(0), device=self.device)

            feats, metadata = contrastive_evaluate(train_dataloader, self.model)
            self.logger.add_embedding("Cluster", feats, metadata, "Pretext Initial")

        self.logger.log('\n- Training:')        
        end_epoch = self.p['epochs']
        for epoch in range(self.start_epoch, end_epoch):
            self.epoch = epoch
            self.logger.log(f'-- Epoch {epoch+1}/{end_epoch}')
            self.logger.log('-'*15)

            lr = adjust_learning_rate(self.p, self.optimizer, self.epoch)
            self.logger.log(f'Adjusted learning rate to {lr:.5f}')
            self.logger.scalar_summary('Train Pretex', 'learning_rate', lr, self.epoch)

            tmp_loss_dict = pretext_train(train_dataloader, self.model, 
                                          criterion, self.optimizer, self.scaler,
                                          self.epoch, self.pretext_previous_loss, self.logger)
            for loss, value in tmp_loss_dict.items():
                self.logger.scalar_summary("Train Pretex", loss, value, self.epoch)

            feats, metadata, evaluation_metrics = contrastive_evaluate(train_dataloader, self.model)
            self.logger.add_embedding("Cluster", feats, metadata, self.epoch)

            for metric, value in evaluation_metrics.items():
                self.logger.scalar_summary("Train Pretext Evaluation", metric, value, self.epoch)

            # Checkpoint
            if tmp_loss_dict['Total Loss'] <= self.pretext_best_loss:
                self.pretext_best_loss = tmp_loss_dict['Total Loss']
                self.save(type="pretext", checkpoint=True)
                self.save(type="pretext", checkpoint=False)

        #Save last checkpoint in order to restart training
        self.save(type="pretext", checkpoint=True, tag='last')

        #Load best checkpoint
        self.load(type="pretext", checkpoint=True)
        self.logger.log(f'Best Model saved from epoch: {self.epoch}')

        self.makeTSRepository(train_dataset, val_transforms, sanomaly)
        # Make new repository of time series for the second stage.
        
    def train_classification(self):
        self.logger.log('CARLA Self-supervised Classification stage --> ')
        self.logger.log_hyperparams(self.p)

        # Data
        self.logger.log(f"\n- Get dataset and dataloaders for {self.p['train_db_name']} dataset - timeseries {self.p['fname']}")
        train_transformations = get_transformations(self.p)
        val_transformations = get_transformations(self.p)
        sanomaly = inject_sub_anomaly(self.p, self.logger)
        
        # In the self-supervised state we use as data the triplets with saves anchors of the first stage
        train_dataset = get_aug_train_dataset(self.p, train_transformations, to_neighbors_dataset = True)
        train_dataloader = get_dataloader(self.p, train_dataset, train=True)
        # In order to correctly measure the similarity matrics, all values need to be checkes,
        # during train we have drop_last so we add second dataloader
        tst_dataloader = get_dataloader(self.p, train_dataset)
        
        dataset_mean, dataset_std = train_dataset.get_info()

        val_dataset = get_dataset(self.p, train=False, transform=val_transformations, sanomaly=sanomaly,
                                  to_augmented_dataset=False, mean_data=dataset_mean, std_data=dataset_std, logger=self.logger)

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

        # Automatic Mixed Precision
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=(self.p.get('amp', False) and self.autocast_available))
        
        self.logger.log('\n- Model initialisation')
        # Checkpoint
        if os.path.exists(f"{self.p['classification_checkpoint_last']}"):
            self.logger.log(f'-- Model initialised from last checkpoint: {self.p['classification_checkpoint_last']}')
            self.load(checkpoint=True, tag='last')
        if os.path.exists(f"{self.p['classification_checkpoint']}"):
            self.logger.log(f'-- Model initialised from last checkpoint: {self.p['classification_checkpoint']}')
            self.load(checkpoint=True)
        else:
            self.logger.log(f'-- No checkpoint file at {self.p["classification_checkpoint"]} -- new model initialised')
            load_backbone(self.p, self.model, self.p['pretext_model'])
            assert(next(self.model.backbone.parameters()).device.type == self.device.type)
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

            tmp_loss_dict = self_sup_classification_train(train_dataloader, self.model,
                                                          criterion, self.optimizer, self.scaler,
                                                          self.epoch, self.logger)
            for loss, value in tmp_loss_dict.items():
                self.logger.scalar_summary("Train Classification", loss, value, self.epoch)

            predictions, metrics = self.eval_classification(tst_dataloader)  # Kanonika ginete sto tst kai oxi sto val
            rep_f1 = metrics["rep_f1"]

            classification_losses = classification_evaluate(predictions, **self.p['criterion_kwargs'])

            for metric, value in metrics.items():
                self.logger.scalar_summary("Train Val Dataset Evaluation", metric, value, self.epoch)
            
            if rep_f1 > best_f1:
                best_f1 = rep_f1
                # self.logger.log('New Checkpoint ...', verbose=verbose, file_path=file_path)
                self.save(type="classification", checkpoint=True)
                self.save(type="classification", checkpoint=False)

        self.save(type="classification", checkpoint=True, tag='_last')

        self.load(type="classification", checkpoint=True)
        self.logger.log(f'Best Model saved from epoch: {self.epoch}')
        # find final prediction in train set.
        _, report = self.eval_classification(tst_dataloader, True, True)
        # and in val set
        _, report = self.eval_classification(val_dataloader, False, True)
    
    @torch.no_grad()
    def eval_classification(self, dataloader, train=False, save_outputs=False):
        experiment = "Train" if train else "Val"

        predictions = get_predictions(self.p, dataloader, self.model, save_outputs, is_training=train)

        label_counts = torch.bincount(predictions['predictions'])
        if train:
            self.majority_label = label_counts.argmax()

        targets = predictions['targets']
        probs = predictions['probabilities']

        # find anomalites as classification task
        anomalies = [1 if p == self.majority_label else 0 for p in predictions['predictions']]
        classification_confusion_matrix = confusion_matrix(targets, anomalies)
        tp = classification_confusion_matrix[1, 1]
        tn = classification_confusion_matrix[0, 0]
        fp = classification_confusion_matrix[0, 1]
        fn = classification_confusion_matrix[1, 0]
        eval_report = {"cls_tp": tp, "cls_tn": tn, "cls_fp": fp, "cls_fn": fn}
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        eval_report['cls_precision'] = precision
        eval_report['cls_recall'] = recall
        eval_report['cls_f1'] = f1

        # find anomalities with anomaly score
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
        best_threshold = thresholds[best_f1_index]

        eval_report['best_f1'] = rep_f1
        eval_report['best_threshold'] = best_threshold

        anomalies = [1 if s >= best_threshold else 0 for s in scores]
        best_tn, best_fp, best_fn, best_tp = confusion_matrix(targets, anomalies).ravel()
        
        eval_report['best_tp'] = best_tp
        eval_report['best_tn'] = best_tn
        eval_report['best_fn'] = best_fn
        eval_report['best_fp'] = best_fp

        self.logger.log(f"{experiment} Set Metrics\nAnomalies --> TP: {best_tp}, TN: {best_tn}, FN: {best_fn}, FP: {best_fp}\nMajority label: {self.majority_label}")
        return predictions, eval_report
    
    @torch.no_grad()
    def inference(self, ts):
        output = self.model(ts)
        prediction = torch.argmax(output, dim=1)
        
        probs = torch.nn.functional.softmax(output, dim=1)
        anomalus_score = 1 - probs[:,   self.majority_label]
        self.logger.log(f"Prediction is {'anomaly' if prediction == self.majority_label else 'normal'}\n\
                        With anomalus score: {anomalus_score.item()}")
        return prediction

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

    def makeTSRepository(self, dataset, val_transformations, sanomaly, topk=10):
        memmory_efficient = self.p['fname'].upper() == 'ALL'
        use_fneighbors = self.p.get('use_fneighbors_in_repository', False)
        base_dataloader = get_dataloader(self.p, dataset)

        aug_length = len(dataset) if self.p.get('use_fneighbors_in_repository', False) else len(dataset) * 2
        ts_repository_aug = TSRepository(aug_length, self.p['model_kwargs']['features_dim'], use_fneighbors=use_fneighbors)  # need size of repository == 1+num_of_anomalies
        ts_repository = TSRepository(len(dataset), self.p['model_kwargs']['features_dim'])

        # Mine the topk nearest neighbors at the very end (Train)
        # These will be served as input to the classification loss.
        self.logger.log('Fill TS Repository for mining the nearest/furthest neighbors (train) ...')    
        
        fill_ts_repository(self.p, base_dataloader, self.model, ts_repository, 
                           real_aug = True, ts_repository_aug = ts_repository_aug, 
                           use_fneighbors=use_fneighbors, logger=self.logger)
        
        out_pre = np.column_stack((ts_repository.features.cpu().numpy(), ts_repository.targets.cpu().numpy()))

        self.logger.log(f'Mine the nearest neighbors (Top-{topk})')
        kfurtherst, knearest = ts_repository_aug.furthest_nearest_neighbors(topk, use_original_algorithm=False, memory_efficient=memmory_efficient)  # hear we mine from aug and not base

        np.save(self.p['pretext_features_train_path'], out_pre)
        np.save(self.p['topk_neighbors_train_path'], knearest)
        np.save(self.p['bottomk_neighbors_train_path'], kfurtherst)
        mean, std = dataset.get_info()

        del dataset
        del ts_repository_aug
        del ts_repository

        dataset = get_dataset(self.p, train=False, transform=val_transformations, sanomaly=sanomaly,
                                  to_augmented_dataset=False, mean_data=mean, std_data=std, logger=self.logger)
        # Use the same variable to save some memory
        base_dataloader = get_dataloader(self.p, dataset)
        ts_repository = TSRepository(len(dataset), self.p['model_kwargs']['features_dim'],
                                          self.p['num_classes'], self.p['criterion_kwargs']['temperature'])
        # Mine the topk nearest neighbors at the very end (Val)
        # These will be used for validation.
        self.logger.log('Fill TS Repository for mining the nearest/furthest neighbors (val) ...')

        fill_ts_repository(self.p, base_dataloader, self.model, ts_repository,
                           real_aug = False, ts_repository_aug = None,
                           use_fneighbors=use_fneighbors,logger=self.logger)
        
        out_pre = np.column_stack((ts_repository.features.cpu().numpy(), ts_repository.targets.cpu().numpy()))

        self.logger.log(f'Mine the nearest neighbors (Top-{topk})')
        kfurtherst, knearest = ts_repository.furthest_nearest_neighbors(topk, use_original_algorithm=False, memory_efficient=memmory_efficient)  # hear we mine from val
        
        np.save(self.p['pretext_features_test_path'], out_pre)
        np.save(self.p['topk_neighbors_val_path'], knearest)
        np.save(self.p['bottomk_neighbors_val_path'], kfurtherst)
    
    def close(self):
        self.logger.finalize()