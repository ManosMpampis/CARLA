import os
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
import faiss

from data.ra_dataset import SaveAugmentedDataset
from utils.utils import EmptyLogger

class TSRepository(object):
    def __init__(self, n, dim, num_classes, temperature, need_fneighbors=False):
        self.n = n
        self.dim = dim 
        self.features = torch.empty(self.n, self.dim, dtype=torch.float)
        self.targets = torch.empty(self.n, dtype=torch.long)
        self.ptr = 0
        self.device = 'cpu'
        self.K = 100
        self.temperature = temperature
        self.C = num_classes

        if need_fneighbors:
            self.ffeatures = torch.empty(self.n, self.dim, dtype=torch.float)
            self.f_target = torch.empty(self.n, dtype=torch.long)
            self.ptr_f = 0

    def weighted_knn(self, predictions):
        # perform weighted knn
        retrieval_one_hot = torch.zeros(self.K, self.C, device=self.device)
        batchSize = predictions.shape[0]
        correlation = torch.matmul(predictions, self.features.t())
        yd, yi = correlation.topk(self.K, dim=1, largest=True, sorted=True)
        candidates = self.targets.view(1,-1).expand(batchSize, -1)
        retrieval = torch.gather(candidates, 1, yi)
        retrieval_one_hot.resize_(batchSize * self.K, self.C).zero_()
        retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
        yd_transform = yd.clone().div_(self.temperature).exp_()
        probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , self.C), 
                          yd_transform.view(batchSize, -1, 1)), 1)
        _, class_preds = probs.sort(1, True)
        class_pred = class_preds[:, 0]

        return class_pred

    def knn(self, predictions):
        # perform knn
        correlation = torch.matmul(predictions, self.features.t())
        sample_pred = torch.argmax(correlation, dim=1)
        class_pred = torch.index_select(self.targets, 0, sample_pred)
        return class_pred

    def mine_nearest_neighbors(self, topk, calculate_accuracy=True):
        # mine the topk nearest neighbors for every sample
        features = self.features.cpu().numpy()
        knn_model = NearestNeighbors(n_neighbors=features.shape[0],
                                 algorithm='brute',
                                 n_jobs=-1)
        knn_model.fit(features)

        distances, indices = knn_model.kneighbors(features, return_distance=True)
        k_furthest_neighbours = []
        k_nearest_neighbours = []
        for i in range(features.shape[0]):
            # sort the neighbours based on their distance to the point
            sorted_indices = np.argsort(distances[i])
            # get the k furthest neighbours for each point
            k_furthest_neighbours.append(indices[i][sorted_indices[-topk:]])
            k_nearest_neighbours.append(indices[i][sorted_indices[1:topk+1]])

        # evaluate 
        if calculate_accuracy:
            targets = self.targets.cpu().numpy()
            neighbor_targets = np.take(targets, indices[:,1:], axis=0) # Exclude sample itself for eval
            anchor_targets = np.repeat(targets.reshape(-1,1), topk, axis=1)
            accuracy = np.mean(neighbor_targets == anchor_targets)

            return k_furthest_neighbours, k_nearest_neighbours, accuracy
        
        else:
            return k_furthest_neighbours, k_nearest_neighbours

    def furthest_nearest_neighbors(self, topk, use_algorithm=0):
        features = self.features

        # # Compute pairwise distances
        # distances = torch.cdist(features, features)
        #
        # # Find indices of k nearest neighbors for each feature
        # _, nearest_indices = distances.topk(topk + 1, largest=False, dim=1)
        # k_nearest_neighbours = nearest_indices[:, 1:]  # exclude self as nearest neighbor
        #
        # # Find indices of k furthest neighbors for each feature
        # _, furthest_indices = distances.topk(topk, largest=True, dim=1)
        # k_furthest_neighbours = furthest_indices[:, :]

        # index = nmslib.init(method='hnsw', space='12')
        # index.addDataPointBatch(features)
        # index.createIndex({'post':2}, prin_progress = True)
        # ids , distances = index.knnQueryBatch(features, k=len(features), num_threads=4)
        #
        # k_furthest_neighbours = ids[:, -1:]
        # k_nearest_neighbours = ids[:, 1:]

        if use_algorithm == 0:
            d = features.shape[1]
            index = faiss.IndexFlatL2(d)
            # index.add(features)
            index.add(features.cpu().numpy())  # CUDA

            xq = np.random.random(d)
            _, ids = index.search(xq.reshape(1, -1).astype(np.float32), len(features))
            sz = ids.shape[1]
            k_furthest_neighbours = ids.reshape(sz, 1)[::-1]
            k_nearest_neighbours = ids[:, :].reshape(sz, 1)
        
        # Mine interpatation if we do not use data labeling
        elif use_algorithm == 1:
            d = self.features.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(self.features.cpu().numpy())

            _, k_nearest_neighbours = index.search(self.features.cpu().numpy(), len(self.features))
            _, k_furthest_neighbours = index.search(self.ffeatures.cpu().numpy(), len(self.features))
            k_furthest_neighbours = k_furthest_neighbours[:, ::-1]

        # Mine interpatation if we use data labeling
        elif use_algorithm == 2:
            d = self.features.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(self.nneighbors.cpu().numpy())

            _, k_nearest_neighbours = index.search(self.features.cpu().numpy(), len(self.nneighbors))

            index = faiss.IndexFlatL2(d)
            index.add(self.fneighbors.cpu().numpy())

            _, k_furthest_neighbours = index.search(self.features.cpu().numpy(), len(self.features))[::-1]
        return k_furthest_neighbours, k_nearest_neighbours


    def reset(self):
        self.ptr = 0
        if hasattr(self, 'ptr_f'):
            self.ptr_f = 0

    def add_ffneighbors(self):
        if not hasattr(self, 'ffeatures'):
            self.ffeatures = torch.empty(self.n, self.dim, dtype=torch.float)
            self.f_target = torch.empty(self.n, dtype=torch.long)
            self.ptr_f = 0

    def resize(self, sz):
        self.n = sz * self.n
        self.features = torch.empty(self.n, self.dim, dtype=torch.float)
        self.targets = torch.empty(self.n, dtype=torch.long)
        if hasattr(self, 'ffeatures'):
            self.ffeatures = torch.empty(self.n, self.dim, dtype=torch.float)
            self.f_target = torch.empty(self.n, dtype=torch.long)
        
    def update(self, features, targets):
        b = features.size(0)
        
        assert(b + self.ptr <= self.n)
        
        self.features[self.ptr:self.ptr+b].copy_(features.detach())
        if not torch.is_tensor(targets): targets = torch.from_numpy(targets)
        self.targets[self.ptr:self.ptr+b].copy_(targets.detach())
        self.ptr += b

    def update_fneighbors(self, ffeatures, f_target):
        b = ffeatures.size(0)

        assert(b + self.ptr_f <= self.n)

        self.ffeatures[self.ptr_f:self.ptr_f+b].copy_(ffeatures.detach())
        if not torch.is_tensor(f_target): f_target = torch.from_numpy(f_target)
        self.f_target[self.ptr_f:self.ptr_f+b].copy_(f_target.detach())
        self.ptr_f += b

    def to(self, device, non_blocking=True):
        self.features = self.features.to(device, non_blocking=non_blocking)
        self.targets = self.targets.to(device, non_blocking=non_blocking)
        self.device = device

    def cpu(self):
        self.to(torch.device('cpu'))

    def cuda(self):
        self.to(torch.device('cuda'))

@torch.no_grad()
def fill_ts_repository(p, loader, model, ts_repository, real_aug=False, ts_repository_aug=None, logger=None):
    """_summary_

    Args:
        p (dict): configuration dictionary
        loader (torch.utils.data.DataLoader): Dataset loader to be used to generate near and far neighbors
        model (torch.nn.Module): Trained model to use for filling time serires repository
        ts_repository (TSRepository): Time series repository to be filled.
        real_aug (bool, optional): Determines if the method save the new values to the Time Series Repositorys. Defaults to False.
        ts_repository_aug (TSRepository, optional): Time series repository filled with original anchors and negative neighbors. Defaults to None.
        device (torch.device, optional): Device to use for torch. Defaults to torch.device("cpu").
        verbose_dict (dict): Verbose variables for logging.
    """
    logger = EmptyLogger() if logger is None else logger
    
    model.eval()
    device = next(model.parameters()).device

    ts_repository.reset()
    if ts_repository_aug != None: ts_repository_aug.reset()
    if real_aug: ts_repository.add_ffneighbors()  # ts_repository.resize(3)

    con_data = torch.tensor([], device="cpu")
    con_target = torch.tensor([], device="cpu")
    con_fneighbors = torch.tensor([], device="cpu")
    con_f_target = torch.tensor([], device="cpu")
    for i, batch in enumerate(loader): 
        ts_org = batch['ts_org'].to(device, non_blocking=True)
        targets = batch['target'].cpu()  # .to(device, non_blocking=True)
        if ts_org.ndim == 3:
            b, w, h = ts_org.shape
        else:
            b, w = ts_org.shape
            h = 1

        output = model(ts_org.reshape(b, h, w)).cpu()
        ts_repository.update(output, targets)
        if ts_repository_aug != None: ts_repository_aug.update(output, targets)
        if i % 100 == 0 or i == len(loader) - 1:
            logger.log(f'Fill TS Repository [{i+1}/{len(loader)}]')

        if real_aug:
            con_data = torch.cat((con_data, ts_org.cpu()), dim=0)
            con_target = torch.cat((con_target, targets), dim=0)

            # TODO: ts_w_augment do not needed because it effectively the same as ts_org
            # ts_w_augment = batch['ts_w_augment'].to(device, non_blocking=True)
            # targets = torch.tensor([1]*ts_w_augment.shape[0], dtype=torch.long, device="cpu")

            # output = model(ts_w_augment.reshape(b, h, w)).cpu()
            # ts_repository.update(output, targets)
            # ts_repository_aug.update(output, targets)


            ts_ss_augment = batch['ts_ss_augment'].to(device, non_blocking=True)
            targets = torch.tensor([4]*ts_ss_augment.shape[0], dtype=torch.long, device="cpu")
            output = model(ts_ss_augment.reshape(b, h, w)).cpu()

            con_fneighbors = torch.cat((con_fneighbors, ts_ss_augment.cpu()), dim=0)
            con_f_target = torch.cat((con_f_target, targets), dim=0)

            ts_repository.update_fneighbors(output, targets)
            ts_repository_aug.update_fneighbors(output, targets)
            
    #         if (i % 50 == 0 and i > 0) or (i == len(loader) - 1):
    #             filename = f"{p['contrastive_dataset']}_{i}.pth"
    #             con_dataset = SaveAugmentedDataset(con_data, con_target, filename=filename)
    #             del con_dataset
    #             con_data = torch.tensor([], device="cpu")
    #             con_target = torch.tensor([], device="cpu")
    # del loader.dataset
    # del loader

    if real_aug:
        # con_dataset = SaveAugmentedDataset(data=torch.tensor([], device="cpu"), target=torch.tensor([], device="cpu"), filename=None)

        # dataset_files = os.listdir(p['pretext_dir'] )
        # dataset_files_list = [file for file in dataset_files if file.startswith('con_train_dataset')]
        # for filename in dataset_files_list:
        #     temp_dataset = SaveAugmentedDataset(data=torch.tensor([], device="cpu"), target=torch.tensor([], device="cpu"), filename=None)
        #     temp_dataset.load_from_file(os.path.join(p['pretext_dir'], filename))
        #     con_dataset.concat_ds(temp_dataset)
        #     del temp_dataset

        con_dataset = SaveAugmentedDataset(con_data, con_target, con_fneighbors, con_f_target)
        con_loader = torch.utils.data.DataLoader(con_dataset, num_workers=p['num_workers'],
                                                 batch_size=p['batch_size'], pin_memory=True,
                                                 drop_last=False, shuffle=False)
        torch.save(con_loader, p['contrastive_dataloader'])