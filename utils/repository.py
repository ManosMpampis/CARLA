import os
import numpy as np
import torch
import faiss

from data.ra_dataset import SaveAugmentedDataset
from utils.utils import EmptyLogger

class TSRepository(object):
    def __init__(self, n, dim, use_fneighbors=False):
        self.n = n
        self.dim = dim 
        self.features = torch.empty(self.n, self.dim, dtype=torch.float)
        self.targets = torch.empty(self.n, dtype=torch.long)
        self.ptr = 0
        self.device = 'cpu'
        self.use_fneighbors = use_fneighbors

        if use_fneighbors:
            self.add_fneighbors()

    def furthest_nearest_neighbors(self, topk, use_original_algorithm=True, memory_efficient=False):
        """Find the furthest nearest neighbors of each feature."""

        if use_original_algorithm:
            # faiss way, respecting clustering output 
            # !!! features contain negative windows !!!
            d = self.features.shape[1]
            index = faiss.IndexFlatL2(d)
            # index.add(features)
            index.add(self.features.cpu())  # CUDA

            # xq = np.random.random(d)
            # _, ids = index.search(xq.reshape(1, -1).astype(np.float32), len(features))
            # sz = ids.shape[1]
            # k_furthest_neighbours = ids.reshape(sz, 1)[::-1]
            # k_nearest_neighbours = ids[:, :].reshape(sz, 1)
            _, k_nearest_neighbours = index.search(self.features.cpu(), len(self.features))
            k_furthest_neighbours = k_nearest_neighbours[:, -1:]
        
        else:
            topk = topk if topk is not None else self.features.shape[0]
            if not self.use_fneighbors:
                # Pytorch way, respecting model output to make far neighbors be the furthest 
                # !!! features contain negative windows !!!
                if memory_efficient:
                # Slow but checkes only one feature at the time
                    k_nearest_neighbours = []
                    k_furthest_neighbours = []
                    for feature in self.features:
                        dist = torch.cdist(feature.unsqueeze(0), self.features).squeeze(0)
                        topk_d = torch.topk(dist, topk+1, largest=False).indices[1:]
                        k_nearest_neighbours.append(topk_d.cpu().numpy())
                        # hear we find the bottomk but use the same variable for memory efficiency
                        topk_d = torch.topk(dist, topk).indices
                        k_furthest_neighbours.append(topk_d.cpu().numpy())
                    k_nearest_neighbours = np.array(k_nearest_neighbours)
                    k_furthest_neighbours = np.array(k_furthest_neighbours)
                else:
                # Faster but very heavy on memory
                    dist = torch.cdist(self.features, self.features)
                    k_nearest_neighbours = torch.topk(k_nearest_neighbours, topk+1, largest=False).indices.cpu().numpy()[:, 1:]
                    k_furthest_neighbours = torch.topk(k_furthest_neighbours, topk).indices.cpu().numpy()

            else:
                # Pytorch way, using knowing labbeling
                if memory_efficient:
                    k_nearest_neighbours = []
                    k_furthest_neighbours = []
                    for feature in self.features:
                        dist = torch.cdist(feature.unsqueeze(0), self.features).squeeze(0)
                        topk_d = torch.topk(dist, topk+1, largest=False).indices[1:]
                        k_nearest_neighbours.append(topk_d.cpu().numpy())

                        # hear we find the bottomk but use the same variable for memory efficiency
                        dist = torch.cdist(feature.unsqueeze(0), self.ffeatures).squeeze(0)
                        topk_d = torch.topk(dist, topk).indices
                        k_furthest_neighbours.append(topk_d.cpu().numpy())
                    k_nearest_neighbours = np.array(k_nearest_neighbours)
                    k_furthest_neighbours = np.array(k_furthest_neighbours)
                else:
                # Faster but very heavy on memory
                    k_nearest_neighbours = torch.cdist(self.features, self.features)
                    k_nearest_neighbours = torch.topk(k_nearest_neighbours, topk+1, largest=False).indices.cpu().numpy()[:, 1:]
                    k_furthest_neighbours = torch.cdist(self.features, self.ffeatures)
                    k_furthest_neighbours = torch.topk(k_furthest_neighbours, topk).indices.cpu().numpy()
        return k_furthest_neighbours, k_nearest_neighbours


    def reset(self):
        self.ptr = 0
        if self.use_fneighbors:
            self.ptr_f = 0

    def add_fneighbors(self):
        if not hasattr(self, 'ffeatures'):
            self.ffeatures = torch.empty(self.n, self.dim, dtype=torch.float)
            self.f_target = torch.empty(self.n, dtype=torch.long)
            self.ptr_f = 0

    def resize(self, sz):
        self.n = sz * self.n
        self.features = torch.empty(self.n, self.dim, dtype=torch.float)
        self.targets = torch.empty(self.n, dtype=torch.long)
        if self.use_fneighbors:
            self.ffeatures = torch.empty(self.n, self.dim, dtype=torch.float)
            self.f_target = torch.empty(self.n, dtype=torch.long)
        
    def update(self, features, targets, is_fneighbors=False):
        if is_fneighbors and self.use_fneighbors:
            self.update_fneighbors()
        
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
        if self.use_fneighbors:
            self.ffeatures = self.ffeatures.to(device, non_blocking=non_blocking)
            self.f_target = self.f_target.to(device, non_blocking=non_blocking)
        self.device = device

    def cpu(self):
        self.to(torch.device('cpu'))

    def cuda(self):
        self.to(torch.device('cuda'))

@torch.no_grad()
def fill_ts_repository(p, loader, model, ts_repository, real_aug=False, ts_repository_aug=None, use_fneighbors=False, logger=None):
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
    if real_aug: ts_repository.resize(3)

    con_data = torch.tensor([], device="cpu")
    con_target = torch.tensor([], device="cpu")

    # Only if we use different lists for far neighbors
    con_fneighbors = torch.tensor([], device="cpu") if use_fneighbors else None
    con_f_target = torch.tensor([], device="cpu") if use_fneighbors else None
    
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

            ts_w_augment = batch['ts_w_augment'].to(device, non_blocking=True)
            targets = torch.tensor([2]*ts_w_augment.shape[0], dtype=torch.long, device="cpu")

            output = model(ts_w_augment.reshape(b, h, w)).cpu()
            ts_repository.update(output, targets)
            # ts_repository_aug.update(output, targets)


            ts_ss_augment = batch['ts_ss_augment'].to(device, non_blocking=True)
            targets = torch.tensor([4]*ts_ss_augment.shape[0], dtype=torch.long, device="cpu")
            output = model(ts_ss_augment.reshape(b, h, w)).cpu()

            if use_fneighbors:
                con_fneighbors = torch.cat((con_fneighbors, ts_ss_augment.cpu()), dim=0)
                con_f_target = torch.cat((con_f_target, targets), dim=0)
            else:
                con_data = torch.cat((con_data, ts_ss_augment.cpu()), dim=0)
                con_target = torch.cat((con_target, targets), dim=0)

            ts_repository.update(output, targets)
            ts_repository_aug.update(output, targets, is_fneighbors=True)
            
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