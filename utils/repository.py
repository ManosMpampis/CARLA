import torch
from data.ra_dataset import SaveAugmentedDataset

class TSRepository(object):
    def __init__(self, n, model_kwargs, res_kwargs):
        self.n = n
        self.head = model_kwargs['head']
        if model_kwargs['head'] == 'tcl':
            self.dim = res_kwargs['mid_channels'][-1]
        else:
            self.dim = model_kwargs['features_dim']
        self.features = torch.FloatTensor(self.n, self.dim)
        self.targets = torch.LongTensor(self.n)
        self.ptr = 0
        self.device = 'cpu'

    def furthest_nearest_neighbors(self, topk):
        features = self.features

        # Compute pairwise distances
        distances = torch.cdist(features, features)
        
        # Find indices of k nearest neighbors for each feature
        _, nearest_indices = distances.topk(topk + 1, largest=False, dim=1)
        k_nearest_neighbours = nearest_indices[:, 1:].cpu().numpy()  # exclude self as nearest neighbor
        
        # Find indices of k furthest neighbors for each feature
        _, furthest_indices = distances.topk(topk, largest=True, dim=1)
        k_furthest_neighbours = furthest_indices[:, :].cpu().numpy()

        return k_furthest_neighbours, k_nearest_neighbours


    def reset(self):
        self.ptr = 0

    def resize(self, sz):
        self.n = sz * self.n
        self.features = torch.FloatTensor(self.n, self.dim)
        self.targets = torch.LongTensor(self.n)
        
    def update(self, features, targets):
        b = features.size(0)
        
        assert(b + self.ptr <= self.n)
        features = features.mean(dim=-1) if features.ndim == 3 else features
        self.features[self.ptr:self.ptr+b].copy_(features.detach())
        if not torch.is_tensor(targets): targets = torch.from_numpy(targets)
        self.targets[self.ptr:self.ptr+b].copy_(targets.detach())
        self.ptr += b

    def to(self, device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.device = device

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda:0')

@torch.no_grad()
def fill_ts_repository(p, loader, model, ts_repository, real_aug=False, ts_repository_aug=None):
    model.eval()
    ts_repository.reset()
    device = next(model.parameters()).device
    if ts_repository_aug != None: ts_repository_aug.reset()
    if real_aug:
        ts_repository.resize(3)

    con_data = torch.tensor([]).to(device)
    con_target = torch.tensor([]).to(device)
    for i, batch in enumerate(loader): 
        ts_org = batch['ts_org'].to(device, non_blocking=True) #cuda
        targets = batch['target'].to(device, non_blocking=True)
        if ts_org.ndim == 3:
            b, w, h = ts_org.shape
        else:
            b, w = ts_org.shape
            h = 1

        # ts_org = torch.from_numpy(ts_org).float(). #cuda
        output = model(ts_org.reshape(b, h, w))
        ts_repository.update(output, targets)
        if ts_repository_aug != None: ts_repository_aug.update(output, targets)
        if i % 100 == 0:
            print('Fill TS Repository [%d/%d]' %(i, len(loader)))

        if real_aug:
            ts_w_augment = batch['ts_w_augment'].to(device, non_blocking=True) #cuda
            ts_ss_augment = batch['ts_ss_augment'].to(device, non_blocking=True) #cuda
            targets = torch.LongTensor([2]*ts_w_augment.shape[0]).to(device, non_blocking=True)
            targets = torch.LongTensor([4]*ts_ss_augment.shape[0]).to(device, non_blocking=True)

            con_data = torch.cat((con_data, ts_org), dim=0)
            # con_target = torch.cat((con_target, torch.from_numpy(targets).float()), dim=0)
            con_target = torch.cat((con_target, targets), dim=0) #cuda

            
            # ts_w_augment = torch.from_numpy(ts_w_augment).float() #cuda
            output = model(ts_w_augment.reshape(b, h, w))
            ts_repository.update(output, targets)
            # ts_repository_aug.update(output, targets)
            
            # ts_ss_augment = torch.from_numpy(ts_ss_augment).float() #cuda
            con_data = torch.cat((con_data, ts_ss_augment), dim=0)
            con_target = torch.cat((con_target, targets), dim=0)
            output = model(ts_ss_augment.reshape(b, h, w))
            ts_repository.update(output, targets)
            ts_repository_aug.update(output, targets)


    if real_aug:
        con_dataset = SaveAugmentedDataset(con_data, con_target)
        con_loader = torch.utils.data.DataLoader(con_dataset, num_workers=p['num_workers'],
                                                 batch_size=p['batch_size'], pin_memory=True,
                                                 drop_last=False, shuffle=False)
        torch.save(con_loader, p['contrastive_dataset'])