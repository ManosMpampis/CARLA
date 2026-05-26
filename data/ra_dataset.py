import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset


class SaveAugmentedDataset(Dataset):
    def __init__(self, data, target, all_data=None, all_targets=None):
        super(SaveAugmentedDataset, self).__init__()
        self.classes = ['Normal', 'Anomaly', 'Noise', 'Point', 'Subseq', 'Subseq2']
        self.targets = target
        self.data = data
        self.all_data = all_data
        self.all_targets = all_targets

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'ts': ts, 'target': index of target class, 'meta': dict}
        """
        ts = self.data[index]
        if len(self.targets) > 0:
            target = int(self.targets[index])
            class_name = self.classes[target]
        else:
            target = 0
            class_name = ''

        ts_size = (ts.shape[0])

        out = {'ts_org': ts, 'target': target, 'meta': {'ts_size': ts_size, 'index': index, 'class_name': class_name}}

        return out

    def get_ts(self, index):
        ts = self.data[index]
        return ts

    def __len__(self):
        return len(self.data)
    

class DynamicNeighbors(Dataset):
    def __init__(self, original_dataset, p, data_number=None):
        super(DynamicNeighbors, self).__init__()
        self.classes = ['Normal', 'Anomaly', 'Noise', 'Point', 'Subseq', 'Subseq2']
        self.dataset = original_dataset

        self.targets = self.dataset.targets
        self.data = self.dataset.ts_org

        self.ts_w_augment = self.dataset.ts_w_augment
        self.ts_ss_augment = self.dataset.ts_ss_augment

        if data_number is not None:
            self.data = self.data[:data_number]
            self.targets = self.targets[:data_number]
            self.ts_w_augment = self.ts_w_augment[:data_number]
            self.ts_ss_augment = self.ts_ss_augment[:data_number]
            
        self.topk = p["num_neighbors"]
        self.k_furthest_nneighbours = np.zeros((len(self.data), self.topk))
        self.k_nearest_fneighbours = np.zeros((len(self.data), self.topk))
    
    @torch.no_grad()
    def predict_and_update(self, model, loader, p, update=True):
        model.eval()
        predictions = []
        probs = []
        targets = []

        device = next(model.parameters()).device
        
        data_features = torch.tensor([]).to(device)
        ts_w_augment_features = torch.tensor([]).to(device)
        ts_ss_augment_features = torch.tensor([]).to(device)

        for i, batch in enumerate(loader): 
            ts_org = batch['ts_org'].to(device, non_blocking=True)
            if ts_org.ndim == 3:
                b, w, h = ts_org.shape
            else:
                b, w = ts_org.shape
                h = 1

            output = model(ts_org.reshape(b, h, w), forward_pass="return_all") #TODO: output features
            data_features = torch.cat((data_features, output["features"]), dim=0)
            predictions.append(torch.argmax(output["output"], dim=1))
            probs.append(F.softmax(output["output"], dim=1) if output["output"].size(1) > 1 else F.sigmoid(output["output"]))
            targets.append(batch["target"].to(device))


            ts_w_augment = batch['ts_w_augment'].to(device, non_blocking=True)
            output = model(ts_w_augment.reshape(b, h, w), forward_pass="return_all") #TODO: output features
            ts_w_augment_features = torch.cat((ts_w_augment_features, output["features"]), dim=0)
            predictions.append(torch.argmax(output["output"], dim=1))
            probs.append(F.softmax(output["output"], dim=1) if output["output"].size(1) > 1 else F.sigmoid(output["output"]))
            targets.append(torch.LongTensor([2]*ts_w_augment.shape[0]).to(device, non_blocking=True))

            ts_ss_augment = batch['ts_ss_augment'].to(device, non_blocking=True) #cuda
            output = model(ts_ss_augment.reshape(b, h, w), forward_pass="return_all")
            ts_ss_augment_features = torch.cat((ts_ss_augment_features, output["features"]), dim=0)
            predictions.append(torch.argmax(output["output"], dim=1))
            probs.append(F.softmax(output["output"], dim=1) if output["output"].size(1) > 1 else F.sigmoid(output["output"]))
            targets.append(torch.LongTensor([4]*ts_ss_augment.shape[0]).to(device, non_blocking=True))

        if update:
            # Compute pairwise distances
            distances = torch.cdist(data_features, ts_w_augment_features)
            # Find indices of k furthest near-neighbors for each feature
            _, furthest_indices = distances.topk(self.topk, largest=True, dim=1)
            self.k_furthest_nneighbours = furthest_indices[:, :].cpu().numpy()

            distances = torch.cdist(data_features, ts_ss_augment_features)
            # Find indices of k nearest far-neighbors for each feature
            _, nearest_indices = distances.topk(self.topk, largest=False, dim=1)
            self.k_nearest_fneighbours = nearest_indices[:, :].cpu().numpy()

        predictions = torch.cat(predictions, dim=0).cpu()
        probs = torch.cat(probs, dim=0).cpu()
        targets = torch.cat(targets, dim=0)

        prob_np = np.array(probs)
        phdr = [str(x) for x in range(prob_np.shape[1])] + ["Class"]
        # prob_np = np.hstack((prob_np, np.array(targets)[np.newaxis].T))
        prob_np = np.hstack((prob_np, np.array(targets.cpu().numpy())[np.newaxis].T))
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
        if len(self.targets) > 0:
            target = int(self.targets[index])
            class_name = self.classes[target]
        else:
            target = 0
            class_name = ''

        ts_size = (ts.shape[0])

        out = {'ts_org': ts, 'target': target, 'ts_w_augment': ts_w_augment, 'ts_ss_augment': ts_ss_augment, 'meta': {'ts_size': ts_size, 'index': index, 'class_name': class_name}}

        return out

    def get_ts(self, index):
        ts = self.data[index]
        return ts

    def __len__(self):
        return len(self.data)