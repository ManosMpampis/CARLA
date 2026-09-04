import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
import pandas as pd

from utils.utils import find_target

""" 
    AugmentedDataset
    Returns a ts together with an augmentation.
"""


class AugmentedDataset(Dataset):
    def __init__(self, dataset):
        super(AugmentedDataset, self).__init__()
        self.samples = [
            {} for _ in range(len(dataset))
        ]  # Initialized with empty dictionaries
        self.ts_org = [
            np.empty(dataset.data[0].shape) for _ in range(len(dataset))
        ]  # Initialized
        self.targets = [
            np.empty(dataset.targets[0].shape) for _ in range(len(dataset))
        ]  # Initialized
        self.ts_w_augment = [
            np.empty(dataset.data[0].shape) for _ in range(len(dataset))
        ]  # Initialized
        self.ts_ss_augment = [
            np.empty(dataset.data[0].shape) for _ in range(len(dataset))
        ]  # Initialized
        self.ts_ss_mask = [
            np.empty(dataset.data[0].shape[0], dtype=np.float32) for _ in range(len(dataset))
        ]  # float32 [T]: 1.0 where the synthetic sub-anomaly was injected
        self.meta = [
            {} for _ in range(len(dataset))
        ]  # Initialized
        transform = dataset.transform
        sanomaly = dataset.sanomaly
        self.scaler = dataset.scaler
        dataset.transform = None
        self.dataset = dataset

        if isinstance(transform, dict):
            self.ts_transform = transform["standard"]
            self.augmentation_transform = transform["augment"]
        else:
            self.ts_transform = transform
            self.augmentation_transform = transform
            self.subseq_anomaly = sanomaly

        self.create_pairs()

    def create_pairs(self):
        for index in range(len(self.dataset)):
            item = self.dataset.__getitem__(index)
            ts_org = item["ts_org"]
            ts_trg = item["target"]

            # Get random neighbor from windows before time step T
            if index > 10:
                rand_nei = np.random.randint(index - 10, index)
                sample_nei = self.dataset.__getitem__(rand_nei)
                ts_w_augment = sample_nei["ts_org"]
            else:
                ts_w_augment = self.augmentation_transform(ts_org)

            ts_ss_augment = self.subseq_anomaly(ts_org)
            # The injector records which timesteps it modified (see SubAnomaly.last_mask);
            # used as the target of the auxiliary localization head.
            last_mask = getattr(self.subseq_anomaly, "last_mask", None)
            ss_mask = (
                np.asarray(last_mask, dtype=np.float32)
                if last_mask is not None
                else np.zeros(ts_org.shape[0], dtype=np.float32)
            )
            # During inference we do not know what inputs are anomalies and how they derived. So we normalize everything the same way.
            self.ts_org[index] = self.scaler.transform(ts_org)
            self.ts_w_augment[index] = self.scaler.transform(ts_w_augment)
            self.ts_ss_augment[index] = self.scaler.transform(ts_ss_augment)
            self.ts_ss_mask[index] = ss_mask
            self.targets[index] = ts_trg
            self.meta[index] = item["meta"]

            self.samples[index] = {
                "ts_org": self.ts_org[index],
                "ts_w_augment": self.ts_w_augment[index],
                "ts_ss_augment": self.ts_ss_augment[index],
                "target": ts_trg,
                "meta": item["meta"],
            }

    def __len__(self):
        return len(self.dataset)

    def concat_ds(self, new_ds):
        self.dataset.data = np.concatenate(
            (self.dataset.data, new_ds.dataset.data), axis=0
        )
        self.dataset.targets = np.concatenate(
            (self.dataset.targets, new_ds.dataset.targets), axis=0
        )

    def __getitem__(self, index):
        return self.samples[index]

class DynamicNeighbors(Dataset):
    def __init__(self, original_dataset, p, data_number=None):
        super(DynamicNeighbors, self).__init__()
        self.classes = ['Normal', 'Anomaly', 'Noise', 'Point', 'Subseq', 'Subseq2']
        self.dataset = original_dataset

        self.targets = self.dataset.targets
        self.data = self.dataset.ts_org

        self.ts_w_augment = self.dataset.ts_w_augment
        self.ts_ss_augment = self.dataset.ts_ss_augment
        self.ts_ss_mask = getattr(self.dataset, "ts_ss_mask", None)
        self.meta = self.dataset.meta
        if data_number is not None:
            self.data = self.data[:data_number]
            self.targets = self.targets[:data_number]
            self.ts_w_augment = self.ts_w_augment[:data_number]
            self.ts_ss_augment = self.ts_ss_augment[:data_number]
            if self.ts_ss_mask is not None:
                self.ts_ss_mask = self.ts_ss_mask[:data_number]
            
        self.topk = p["num_neighbors"]
        self.k_furthest_nneighbours = np.zeros((len(self.data), self.topk))
        self.k_nearest_fneighbours = np.zeros((len(self.data), self.topk))
    
    @torch.no_grad()
    def predict_and_update(self, model, loader, p, epoch=0, update=0):
        model.eval()
        predictions = []
        probs = []
        targets = []

        device = next(model.parameters()).device
        
        data_features = torch.tensor([]).to(device)
        ts_w_augment_features = torch.tensor([]).to(device)
        ts_ss_augment_features = torch.tensor([])

        for i, batch in enumerate(loader): 
            ts_org = batch['ts_org'].to(device, non_blocking=True)
            ts_w_augment = batch['ts_w_augment'].to(device, non_blocking=True)
            ts_ss_augment = batch['ts_ss_augment'].to(device, non_blocking=True)
            ts_label = batch["target"]
            if ts_org.ndim == 3:
                b, w, h = ts_org.shape
            else:
                b, w = ts_org.shape
                h = 1

            output = model(ts_org.reshape(b, h, w), forward_pass="return_all")
            data_features = torch.cat((data_features, output["features"]), dim=0)
            predictions.append(torch.argmax(output["output"], dim=1))
            probs.append(F.softmax(output["output"], dim=1) if output["output"].size(1) > 1 else F.sigmoid(output["output"]))
            targets.append(ts_label)

            output = model(ts_w_augment.reshape(b, h, w), forward_pass="return_all")
            ts_w_augment_features = torch.cat((ts_w_augment_features, output["features"]), dim=0)
            predictions.append(torch.argmax(output["output"], dim=1))
            probs.append(F.softmax(output["output"], dim=1) if output["output"].size(1) > 1 else F.sigmoid(output["output"]))
            targets.append(torch.ones_like(ts_label)*2)
            
            output = model(ts_ss_augment.reshape(b, h, w), forward_pass="return_all")
            ts_ss_augment_features = torch.cat((ts_ss_augment_features, output["features"].to("cpu")), dim=0)
            predictions.append(torch.argmax(output["output"], dim=1))
            probs.append(F.softmax(output["output"], dim=1) if output["output"].size(1) > 1 else F.sigmoid(output["output"]))
            targets.append(torch.ones_like(ts_label)*4)

        if (update > 0) and (epoch % update == 0):
            # Compute pairwise distances
            distances = torch.cdist(data_features, ts_w_augment_features).to("cpu")
            del ts_w_augment_features

            # Find indices of k furthest near-neighbors for each feature
            _, furthest_indices = distances.topk(self.topk, largest=True, dim=1)
            self.k_furthest_nneighbours = furthest_indices[:, :].cpu().numpy()

            distances = torch.cdist(data_features, ts_ss_augment_features.to(device)).to("cpu")
            del ts_ss_augment_features

            # Find indices of k nearest far-neighbors for each feature
            _, nearest_indices = distances.topk(self.topk, largest=False, dim=1)
            self.k_nearest_fneighbours = nearest_indices[:, :].cpu().numpy()

        predictions = torch.cat(predictions, dim=0).cpu()
        probs = torch.cat(probs, dim=0).cpu()
        targets = torch.cat(targets, dim=0)

        prob_np = np.array(probs)
        phdr = [str(x) for x in range(prob_np.shape[1])] + ["Class"]
        # prob_np = np.hstack((prob_np, np.array(targets)[np.newaxis].T))
        final_targets = find_target(targets)
        prob_np = np.hstack((prob_np, final_targets[np.newaxis].T))
        
        prob_df = pd.DataFrame(prob_np, columns=phdr)
        prob_df.to_csv(
                p["classification_trainprobs"], index=False, header=True, sep=","
        )
        
        return {"predictions": predictions, "probabilities": probs, "targets": targets}


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'ts': ts, 'target': index of target class, 'meta': dict}
        """
        # dict: {"ts_org": ts, "ts_w_augment": ts_near_neightbor, "ts_ss_augment": ts_far_neightbor, "target": 0 if normal else 1}
        ts = self.data[index]
        ts_w_augment = self.ts_w_augment[index]
        ts_ss_augment = self.ts_ss_augment[index]
        meta = self.meta[index]
        if len(self.targets) > 0:
            target = self.targets[index]
        else:
            target = torch.zeros(ts.shape[0], dtype=torch.long)

        return {'ts_org': ts, 'target': target, 'ts_w_augment': ts_w_augment, 'ts_ss_augment': ts_ss_augment, 'meta': meta}

    def get_ts(self, index):
        ts = self.data[index]
        return ts

    def __len__(self):
        return len(self.data)
    
class ContrustiveDataset(Dataset):
    def __init__(self, dataset, transform, p):
        super(ContrustiveDataset, self).__init__()

        if isinstance(transform, dict):
            self.anchor_transform = transform["standard"]
            self.neighbor_transform = transform["augment"]
        else:
            self.anchor_transform = transform
            self.neighbor_transform = transform

        self.dataset = dataset
        
        # self.dataset.k_furthest_nneighbours # furthest near-neighbor indices (np.array  [len(dataset) x k])
        # self.dataset.k_nearest_fneighbours # Nearest further-neighbor indices (np.array  [len(dataset) x k])

        self.NNeighbor = self.dataset.ts_w_augment
        self.FNeighbor = self.dataset.ts_ss_augment
        # Injection mask per FNeighbor window (parallel to ts_ss_augment)
        self.FNeighbor_mask = getattr(self.dataset, "ts_ss_mask", None)

        self.mean = 0
        self.std = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        anchor = self.dataset.__getitem__(index)

        NN_index = np.random.choice(self.dataset.k_furthest_nneighbours[index], 1)[0]
        NNeighbor = self.NNeighbor.__getitem__(NN_index)
        FN_index = np.random.choice(self.dataset.k_nearest_fneighbours[index], 1)[0]
        FNeighbor = self.FNeighbor.__getitem__(FN_index)
        if self.FNeighbor_mask is not None:
            FNeighbor_mask = np.asarray(self.FNeighbor_mask[FN_index], dtype=np.float32)
        else:
            FNeighbor_mask = np.zeros(FNeighbor.shape[0], dtype=np.float32)

        return {"anchor": anchor["ts_org"], "NNeighbor": NNeighbor, "FNeighbor": FNeighbor,
                "FNeighbor_mask": FNeighbor_mask, "target": anchor["target"], "meta": anchor["meta"]}

    def concat_ds(self, new_ds):
        self.dataset.data = np.concatenate(
            (self.dataset.data, new_ds.dataset.data), axis=0
        )
        self.dataset.targets = np.concatenate(
            (self.dataset.targets, new_ds.dataset.targets), axis=0
        )
