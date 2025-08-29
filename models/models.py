import torch.nn as nn
import torch.nn.functional as F

class ContrastiveModel(nn.Module):
    def __init__(self, backbone, head='mlp', features_dim=128):
        super(ContrastiveModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.head = head
 
        if head == 'linear':
            self.head = nn.Linear(self.backbone_dim, features_dim)

        elif head == 'mlp':
            self.head = nn.Sequential(
                    nn.Linear(self.backbone_dim, self.backbone_dim),
                    nn.ReLU(), nn.Linear(self.backbone_dim, features_dim))
        else:
            raise ValueError('Invalid head {}'.format(head))

    def forward(self, x):
        features = self.backbone(x)
        features = self.head(features)
        features = F.normalize(features, dim = 1)
        return features


class ClusteringModel(nn.Module):
    def __init__(self, backbone, nclusters, nheads=1):
        super(ClusteringModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.nheads = nheads
        assert(isinstance(self.nheads, int))
        assert(self.nheads > 0)
        assert(self.nheads == 1)
        self.head = nn.Linear(self.backbone_dim, nclusters)
    
    def forward(self, x, forward_pass='default'):
        features = self.backbone(x)
        if forward_pass == 'default':
            return self.head(features)
        elif forward_pass == 'return_all':
            return {'features': features, 'output': self.head(features)}
        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))
