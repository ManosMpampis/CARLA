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
        return F.normalize(x, dim=self.dim)


class ContrastiveModel(nn.Module):
    def __init__(self, backbone, head="mlp", features_dim=128):
        super(ContrastiveModel, self).__init__()
        self.backbone = backbone["backbone"]
        self.backbone_output_dim = backbone["dim"][-1]
        self.head = head

        if head == "linear":
            self.contrastive_head = nn.Sequential(
                MeanLayer(dim=-1),
                nn.Linear(self.backbone_dim, features_dim),
                NormalizationLayer(dim=1),
            )

        elif head == "mlp":
            self.contrastive_head = nn.Sequential(
                MeanLayer(dim=-1),
                nn.Linear(self.backbone_output_dim, self.backbone_output_dim),
                nn.ReLU(),
                nn.Linear(self.backbone_output_dim, features_dim),
                NormalizationLayer(dim=1),
            )

        elif head == "tcl":
            self.contrastive_head = nn.Identity()
        else:
            raise ValueError("Invalid head {}".format(head))

    def forward(self, x):
        features = self.backbone(x)

        return self.contrastive_head(features)


class ClusteringModel(nn.Module):
    def __init__(self, backbone, nclusters, nheads=1):
        super(ClusteringModel, self).__init__()
        self.backbone = backbone["backbone"]
        self.backbone_output_dim = backbone["dim"][-1]
        self.cluster_head = nn.Linear(self.backbone_output_dim, nclusters)
        self.nclusters = nclusters

    def forward(self, x, forward_pass="default"):
        if forward_pass == "default":
            features = self.backbone(x)
            out = self.cluster_head(features.mean(dim=-1))
        elif forward_pass == "backbone":
            out = self.backbone(x)
        elif forward_pass == "head":
            out = self.cluster_head(x.mean(dim=-1))
        elif forward_pass == "return_all":
            features = self.backbone(x)
            out = {
                "features": features.mean(dim=-1),
                "output": self.cluster_head(features.mean(dim=-1)),
            }
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
            out = self.classification_head(x)
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
