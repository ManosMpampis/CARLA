
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda")

class MeanLayer(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(dim=self.dim)
    
class NormalizationLayer(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, dim = self.dim)
    
class ContrastiveModel(nn.Module):
    def __init__(self, backbone, head='mlp', features_dim=128):
        super(ContrastiveModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_output_dim = backbone['dim'][-1]
        self.head = head
 
        if head == 'linear':
            self.contrastive_head = nn.Sequential(
                MeanLayer(dim=-1),
                nn.Linear(self.backbone_dim, features_dim),
                NormalizationLayer(dim=1))

        elif head == 'mlp':
            self.contrastive_head = nn.Sequential(
                    MeanLayer(dim=-1),
                    nn.Linear(self.backbone_output_dim, self.backbone_output_dim),
                    nn.ReLU(), nn.Linear(self.backbone_output_dim, features_dim),
                    NormalizationLayer(dim=1))
        
        elif head == 'tcl':
            self.contrastive_head = nn.Identity()
        else:
            raise ValueError('Invalid head {}'.format(head))

    def forward(self, x):
        features = self.backbone(x)

        return self.contrastive_head(features)


class ClusteringModel(nn.Module):
    def __init__(self, backbone, nclusters, nheads=1):
        super(ClusteringModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_output_dim = backbone['dim'][-1]
        self.nheads = nheads
        assert(isinstance(self.nheads, int))
        assert(self.nheads > 0)
        self.cluster_head = nn.ModuleList([nn.Linear(self.backbone_output_dim, nclusters) for _ in range(self.nheads)])

    def forward(self, x, forward_pass='default'):
        if forward_pass == 'default':
            features = self.backbone(x)
            out = [cluster_head(features.mean(dim=-1)) for cluster_head in self.cluster_head]

        elif forward_pass == 'backbone':
            out = self.backbone(x)

        elif forward_pass == 'head':
            out = [cluster_head(x.mean(dim=-1)) for cluster_head in self.cluster_head]

        elif forward_pass == 'return_all':
            features = self.backbone(x)
            out = {'features': features, 'output': [cluster_head(features.mean(dim=-1)) for cluster_head in self.cluster_head]}
        
        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))        

        return out
