import torch
import torch.nn as nn
import torch.nn.functional as F
from .convolutions import _init_weights


class MeanLayer(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(dim=self.dim)


class PoolLayer(nn.Module):
    """Temporal pooling for per-timestep features [b, dim, T] -> [b, out_dim].

    mean:   global context (default)
    max:    strongest local activation
    avgmax: concat of both, keeps global context AND fine details
    """

    def __init__(self, mode="mean"):
        super().__init__()
        self.mode = mode

    @property
    def out_multiplier(self):
        return 2 if self.mode == "avgmax" else 1

    def forward(self, x):
        if self.mode == "max":
            return x.max(dim=-1).values
        if self.mode == "avgmax":
            return torch.cat([x.mean(dim=-1), x.max(dim=-1).values], dim=-1)
        return x.mean(dim=-1)


class NormalizationLayer(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, dim=self.dim)


class ContrastiveModel(nn.Module):
    def __init__(self, backbone, head="mlp", features_dim=128, pooling="mean"):
        super(ContrastiveModel, self).__init__()
        self.backbone = backbone["backbone"]
        self.backbone_output_dim = backbone["dim"][-1]
        self.head = head
        self.pooling = PoolLayer(pooling)
        pooled_dim = self.backbone_output_dim * self.pooling.out_multiplier

        if head == "linear":
            self.contrastive_head = nn.Sequential(
                self.pooling,
                nn.Linear(pooled_dim, features_dim),
                NormalizationLayer(dim=1),
            )

        elif head == "mlp":
            self.contrastive_head = nn.Sequential(
                self.pooling,
                nn.Linear(pooled_dim, self.backbone_output_dim),
                nn.ReLU(),
                nn.Linear(self.backbone_output_dim, features_dim),
                NormalizationLayer(dim=1),
            )

        elif head == "tcl":
            self.contrastive_head = nn.Identity()
        else:
            raise ValueError("Invalid head {}".format(head))
        
        # Initialize all weights and biases in the contrastive head
        self._init_head_weights()
    
    def _init_head_weights(self):
        """Initialize weights for linear layers in the head."""
        for module in self.contrastive_head.modules():
            if isinstance(module, nn.Linear):
                _init_weights(module)

    def forward(self, x):
        features = self.backbone(x)

        return self.contrastive_head(features)


class ClusteringModel(nn.Module):
    def __init__(self, backbone, nclusters, nheads=1, localization_head=False, pooling="mean"):
        super(ClusteringModel, self).__init__()
        self.backbone = backbone["backbone"]
        self.backbone_output_dim = backbone["dim"][-1]
        self.pooling = pooling
        head_dim = self.backbone_output_dim * (2 if pooling == "avgmax" else 1)
        self.cluster_head = nn.Linear(head_dim, nclusters)
        self.nclusters = nclusters

        self.pooling_layer = PoolLayer(pooling)
        # Auxiliary per-timestep localization head: predicts WHERE in the
        # window a sub-anomaly occurs. Trained with the BCE localization term
        # on the FNeighbor branch (criterion_kwargs: localization_weight > 0).
        self.localization_head = (
            nn.Conv1d(self.backbone_output_dim, 1, kernel_size=1)
            if localization_head
            else None
        )

        # Initialize all weights and biases in the cluster head
        _init_weights(self.cluster_head)
        if self.localization_head is not None:
            _init_weights(self.localization_head)


    def forward(self, x, forward_pass="default"):
        if forward_pass == "default":
            features = self.backbone(x)
            out = self.cluster_head(self.pooling_layer(features))
        elif forward_pass == "backbone":
            out = self.backbone(x)
        elif forward_pass == "head":
            features = x.mean(dim=-1)
            out = {
                "features": features,
                "output": self.cluster_head(self.pooling_layer(features)),
            }
        elif forward_pass == "return_all":
            features = self.backbone(x)
            out = {
                "features": self.pooling_layer(features),
                "output": self.cluster_head(self.pooling_layer(features)),
            }
            if self.localization_head is not None:
                out["localization_logits"] = self.localization_head(features).squeeze(1)
        else:
            raise ValueError("Invalid forward pass {}".format(forward_pass))
        return out

class ClassificationModel(nn.Module):
    def __init__(self, clusteringModel, classes=2):
        super(ClassificationModel, self).__init__()
        self.backbone = clusteringModel.backbone
        self.cluster_head = clusteringModel.cluster_head
        self.classification_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(clusteringModel.nclusters, classes)
        )
        
        # Initialize the classification head weights and biases
        for module in self.classification_head.modules():
            if isinstance(module, nn.Linear):
                _init_weights(module)

    def forward(self, x, forward_pass="default"):
        if forward_pass == "default":
            x = self.backbone(x)
            x = self.cluster_head(x.mean(dim=-1))
            out = self.classification_head(x)
        elif forward_pass == "backbone":
            out = self.backbone(x)
        elif forward_pass == "cluster":
            out = self.cluster_head(x.mean(dim=-1))
        elif forward_pass == "head":
            features = x.mean(dim=-1)
            cluster = self.cluster_head(features)
            out = {
                "features": features,
                "cluster": cluster,
                "output": self.classification_head(cluster),
            }
        elif forward_pass == "return_all":
            features = self.backbone(x)
            cluster = self.cluster_head(features.mean(dim=-1))
            out = {
                "features": features.mean(dim=-1),
                "cluster": cluster,
                "output": self.classification_head(cluster),
            }
        else:
            raise ValueError("Invalid forward pass {}".format(forward_pass))
        return out
