import os
import torch
from termcolor import colored
from torchmetrics.functional import roc_auc_score, average_precision, confusion_matrix, precision_recall_curve
from utils.config import create_config
from utils.common_config import get_criterion, get_model, get_train_dataset, get_val_dataset, get_train_dataloader, get_val_dataloader, get_train_transformations, get_val_transformations1, get_optimizer, adjust_learning_rate, inject_sub_anomaly
from utils.evaluate_utils import contrastive_evaluate, get_predictions, classification_evaluate, pr_evaluate
from utils.repository import TSRepository, fill_ts_repository
from utils.train_utils import pretext_train as pt_train, self_sup_classification_train as cls_train
import matplotlib.pyplot as plt


from utils.utils import Logger

class CarlaLearner:
    def __init__(self, config_env, config_exp, fname, version=None, device=None, verbose=1, tensorboard=True):
        self.config = create_config(config_env, config_exp, fname, version)
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.logger = Logger(self.config.get('version', 'default'), verbose=verbose, file_path="./", use_tensorboard=tensorboard)

    def save(self, path=None, type="classification", checkpoint=False, tag=None):
        assert type in ["classification", "pretext"]
        p = self.config
        key = f"{type}_{'checkpoint' if checkpoint else 'model'}"
        if path is None:
            path = p.get(key, None)
            if tag:
                path = f"{path[:-8]}_{tag}{path[-8:]}"
        dictionary = {'model': self.model.module.state_dict()} if hasattr(self.model, 'module') else {'model': self.model.state_dict()}
        if checkpoint:
            dictionary['optimizer'] = self.optimizer.state_dict()
            dictionary['epoch'] = getattr(self, 'epoch', 0)
            dictionary['best_f1'] = getattr(self, 'best_f1', None)
            dictionary['pretext_best_loss'] = getattr(self, 'pretext_best_loss', None)
            dictionary['majority_label'] = getattr(self, 'majority_label', None)
        torch.save(dictionary, path)
        self.logger.log(f"Model saved to {path}")

    def load(self, path=None, type="classification", checkpoint=False, tag=None):
        assert type in ["classification", "pretext"]
        p = self.config
        key = f"{type}_{'checkpoint' if checkpoint else 'model'}"
        if path is None:
            path = p.get(key, None)
            if tag:
                path = f"{path[:-8]}_{tag}{path[-8:]}"
        dictionary = torch.load(path, map_location=self.device)
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(dictionary['model'])
        else:
            self.model.load_state_dict(dictionary['model'])
        if checkpoint and 'optimizer' in dictionary:
            self.optimizer.load_state_dict(dictionary['optimizer'])
        # Restore training state variables
        if checkpoint:
            self.epoch = dictionary.get('epoch', 0)
            self.best_f1 = dictionary.get('best_f1', None)
            self.pretext_best_loss = dictionary.get('pretext_best_loss', None)
            self.majority_label = dictionary.get('majority_label', None)
        self.logger.log(f"Model loaded from {path}")

    def pretext_train(self):
        p = self.config
        train_transforms = get_train_transformations(p)
        sanomaly = inject_sub_anomaly(p)
        train_dataset = get_train_dataset(p, train_transforms, sanomaly, to_augmented_dataset=True)
        train_dataloader = get_train_dataloader(p, train_dataset)
        self.logger.log(f"Pretext dataset size: {len(train_dataset)}")
        self.model = get_model(p).to(self.device)
        criterion = get_criterion(p).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=p['optimizer_kwargs']['lr'])
        # Check for checkpoint
        checkpoint_path = p.get('pretext_checkpoint', None)
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load(path=checkpoint_path, type="pretext", checkpoint=True)
            start_epoch = self.epoch
        else:
            start_epoch = 0
            self.epoch = 0
            self.pretext_best_loss = torch.tensor(float('inf'))
        best_model = None
        for epoch in range(start_epoch, p['epochs']):
            self.epoch = epoch
            self.logger.log(f'Pretext Epoch {epoch+1}/{p["epochs"]}')
            lr = adjust_learning_rate(p, self.optimizer, epoch)
            tmp_loss = pt_train(train_dataloader, self.model, criterion, self.optimizer, epoch, None, device=self.device)['loss']
            self.logger.scalar_summary("Pretext", "Loss", tmp_loss, epoch)
            if tmp_loss <= self.pretext_best_loss:
                self.pretext_best_loss = tmp_loss
                self.save(type="pretext", checkpoint=True)

        self.save(type="pretext", checkpoint=True, tag="last")

        self.load(type="pretext", checkpoint=True)
        self.save(type="pretext", checkpoint=False)
        self.logger.log("Pretext training finished.")

    def classification_train(self):
        p = self.config
        train_transforms = get_train_transformations(p)
        sanomaly = inject_sub_anomaly(p)
        train_dataset = get_train_dataset(p, train_transforms, sanomaly, to_augmented_dataset=True)
        train_dataloader = get_train_dataloader(p, train_dataset)
        val_transforms = get_val_transformations1(p)
        val_dataset = get_val_dataset(p, val_transforms, sanomaly, False, train_dataset.mean, train_dataset.std)
        val_dataloader = get_val_dataloader(p, val_dataset)
        self.logger.log(f"Classification train size: {len(train_dataset)}, val size: {len(val_dataset)}")
        self.model = get_model(p, p['pretext_model'])
        self.model = torch.nn.DataParallel(self.model).to(self.device)
        criterion = get_criterion(p).to(self.device)
        self.optimizer = get_optimizer(p, self.model, p.get('update_cluster_head_only', False))
        # Check for checkpoint
        checkpoint_path = p.get('classification_checkpoint', None)
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load(path=checkpoint_path, type="classification", checkpoint=True)
            start_epoch = self
        else:
            start_epoch = 0
            self.epoch = 0
            self.best_f1 = -torch.tensor(float('inf'))
            self.majority_label = None
        

        for epoch in range(start_epoch, p['epochs']):
            self.epoch = epoch
            self.logger.log(f'Classification Epoch {epoch+1}/{p["epochs"]}')
            lr = adjust_learning_rate(p, self.optimizer, epoch)
            cls_train(train_dataloader, self.model, criterion, self.optimizer, epoch, p.get('update_cluster_head_only', False))
            predictions = get_predictions(p, val_dataloader, self.model, False, False)
            rep_f1 = pr_evaluate(predictions, compute_confusion_matrix=False)
            self.logger.scalar_summary("Classification", "F1", rep_f1, epoch)
            # Calculate majority_label from predictions
            pred_labels = predictions[0]['predictions'].cpu()
            counts = torch.bincount(pred_labels, return_counts=True)
            self.majority_label = ...
            if rep_f1 > self.best_f1:
                best_f1 = rep_f1
                self.best_f1 = best_f1
                self.save(type="classification", checkpoint=True)
        self.save(type="classification", checkpoint=True, tag="last")
        self.load(type="classification", checkpoint=True)
        self.save(type="classification", checkpoint=False)
        self.logger.log("Classification training finished.")


    def evaluate_classification(self, train_dataset, sanomaly):
        p = self.config
        val_transforms = get_val_transformations1(p)
        val_dataset = get_val_dataset(p, val_transforms, sanomaly, False, train_dataset.mean, train_dataset.std)
        val_dataloader = get_val_dataloader(p, val_dataset)
        predictions, _ = get_predictions(p, val_dataloader, self.model, True)
        y_true = torch.cat([item['labels'].cpu() for item in predictions[0]['results']])
        y_scores = torch.cat([item['scores'].cpu() for item in predictions[0]['results']])
        roc_auc = roc_auc_score(y_scores, y_true.int())
        pr_auc = average_precision(y_scores, y_true.int())
        self.logger.log(f'AU-ROC: {roc_auc.item()}, AU-PR: {pr_auc.item()}')
        precision, recall, thresholds = precision_recall_curve(y_scores, y_true.int())
        plt.figure()
        plt.plot(recall.numpy(), precision.numpy(), label=f'PR curve (area = {pr_auc.item():.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.show()
        # ROC curve
        # torchmetrics does not return fpr/tpr directly, so we skip ROC plot
        y_pred = (y_scores > 0.5).int()
        cm = confusion_matrix(y_pred, y_true.int(), num_classes=2)
        tn = cm[0,0].item()
        fp = cm[0,1].item()
        fn = cm[1,0].item()
        tp = cm[1,1].item()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        self.logger.log(f'Precision: {prec}, Recall: {rec}, F1: {f1}')
