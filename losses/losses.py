import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

EPS=1e-8


def entropy(x, input_as_probabilities):
    """ 
    Helper function to compute the entropy over the batch 

    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    if input_as_probabilities:
        x_ =  torch.clamp(x, min = EPS)
        b =  x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim = 1) * F.log_softmax(x, dim = 1)

    if len(b.size()) == 2: # Sample-wise entropy
        return -b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))


class ClassificationLoss(nn.Module):
    def __init__(self, entropy_weight = 2.0, inconsistency_weight=1.0, consistency_weight=1.0, entropy_norm=False):
        super(ClassificationLoss, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        self.bce = nn.BCELoss()
        self.entropy_weight = entropy_weight 
        self.inconsistency_weight = inconsistency_weight
        self.consistency_weight = consistency_weight
        self.entropy_norm = entropy_norm

    def forward(self, anchors, nneighbors, fneighbors):
        """
        input:
            - anchors: logits for anchor ts w/ shape [b, num_classes]
            - k nearest neighbors: logits for neighbor ts w/ shape [b, num_classes]
            - k furthest neighbors: logits for neighbor ts w/ shape [b, num_classes]

        output:
            - Loss
        """
        b, n = anchors.size()
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(nneighbors)
        negatives_prob = self.softmax(fneighbors)
       
        # Similarity in output space
        similarity = torch.bmm(anchors_prob.view(b, 1, n), positives_prob.view(b, n, 1)).squeeze()
        ones = torch.ones_like(similarity)
        consistency_loss = self.bce(similarity, ones)

        # DiSimilarity in output space
        negsimilarity = torch.bmm(anchors_prob.view(b, 1, n), negatives_prob.view(b, n, 1)).squeeze()
        zeros = torch.zeros_like(negsimilarity)
        inconsistency_loss = self.bce(negsimilarity, zeros)
        
        # Entropy loss
        if self.entropy_norm:
            entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities = True)/torch.log(torch.tensor(n)) # Normalize to 1
        else:
            entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities = True)
        #-torch.sum(anchors_prob * torch.log(anchors_prob + 1e-12), dim=-1).mean() #

        # Total loss
        total_loss = (self.consistency_weight*consistency_loss) - (self.entropy_weight * entropy_loss) + (self.inconsistency_weight * inconsistency_loss)

        return total_loss, consistency_loss, inconsistency_loss, entropy_loss


class PretextLoss(nn.Module):
    # Based on the implementation of SupContrast
    def __init__(self, bs, temperature, pos_weight=1, neg_weight=1, initial_margin=1.0, margin_constant=False,
                 adjust_factor=0.1, ema_alpha=0, min_improvement=0.0001, orig_margin=False, hard_neg=True):
        super(PretextLoss, self).__init__()
        self.temperature = temperature
        self.bs = bs

        self.margin = initial_margin
        self.initial_margin = initial_margin
        self.max_margin = 5.0
        self.margin_constant = margin_constant
        self.hard_neg = hard_neg
        
        self.adjust_factor = adjust_factor
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        
        self.min_improvement = min_improvement
        self.ema_alpha = ema_alpha
        self.prev_ema_loss = None
        self.orig_margin = orig_margin

        self.previous_loss = None

    def orig_forward(self, features):
        """
        input:
            - features: hidden feature representation of shape [b, 3, dim]

        output:
            - loss: loss computed according to pretext triplet loss
        """
        features_org, features_pos, features_subseq = torch.split(features, self.bs, dim=0)

        # Normalize features for stable distance computation
        anchor = F.normalize(features_org, dim=-1)
        positive = F.normalize(features_pos, dim=-1)
        negative = F.normalize(features_subseq, dim=-1)

        # self.margin = 5
        if self.previous_loss is not None:
            self.margin = max(0.01, self.margin - self.adjust_factor * self.previous_loss)

        positive_distance = torch.sum((anchor - positive) ** 2, dim=-1) / self.temperature
        negative_distance = torch.sum((anchor - negative) ** 2, dim=-1) / self.temperature

        hard_negative_distance = torch.sum(torch.pow(anchor.unsqueeze(1) - negative, 2), dim=-1) / self.temperature
        hard_negative_distance = torch.min(hard_negative_distance, dim=1)[0]

        n_d = hard_negative_distance if self.hard_neg else negative_distance
        loss = torch.clamp(self.margin + positive_distance - n_d, min=0.0)
        # clamped_distance = torch.clamp(self.margin + positive_distance - negative_distance, min=0.0)
        # loss = torch.sum(clamped_distance, dim=1)
        loss = torch.mean(loss)
        positive_d_loss = torch.mean(positive_distance)
        negative_d_loss = torch.mean(negative_distance)
        hard_negative_d_loss = torch.mean(hard_negative_distance)

        self.previous_loss = loss.item()
        return {"loss": loss, "positive_d_loss": positive_d_loss, "negative_d_loss": negative_d_loss, "hard_negative_d_loss": hard_negative_d_loss}
    
    def update_margin(self,  new_margin):
        if new_margin is None:
            return
        self.margin = new_margin.to(self.device) if isinstance(new_margin, torch.Tensor) else torch.tensor(new_margin).to(self.device)

    def forward(self, features):
        """
        input:
            - features: hidden feature representation of shape [b, 3, dim]

        output:
            - loss: loss computed according to pretext triplet loss
        """
        if self.orig_margin:
            return self.orig_forward(features)
        features_org, features_pos, features_subseq = torch.split(features, self.bs, dim=0)

        # Normalize features for stable distance computation
        anchor = F.normalize(features_org, dim=-1)
        positive = F.normalize(features_pos, dim=-1)
        negative = F.normalize(features_subseq, dim=-1)

        negative_distance = torch.sum((anchor - negative) ** 2, dim=-1) / self.temperature
        
        hard_negative_distance = torch.sum(torch.pow(anchor.unsqueeze(1) - negative, 2), dim=-1) / self.temperature
        hard_negative_distance = torch.min(hard_negative_distance, dim=1)[0]
        
        n_d = hard_negative_distance if self.hard_neg else negative_distance

        pos_supression_weight = 1 - ((self.margin - (self.neg_weight * torch.mean(n_d))) / self.margin)
        pos_supression_weight = torch.clamp(pos_supression_weight, min=0, max=self.neg_weight)
        positive_distance = torch.sum((anchor - positive) ** 2, dim=-1) / self.temperature
        
        clamp_neg_loss = torch.clamp((self.neg_weight * n_d), max=self.margin)
        
        loss = (self.pos_weight * positive_distance * pos_supression_weight) - clamp_neg_loss
        # loss = torch.clamp((self.pos_weight * positive_distance * pos_supression_weight) - clamp_neg_loss, min=-self.margin)
        loss = torch.mean(loss)

        # Use ema and update margin for the next batch
        if self.previous_loss is None:
            ema_loss = torch.mean(loss)
        else:
            ema_loss = torch.mean((1.0 - self.ema_alpha) * loss + self.ema_alpha * self.previous_loss)

        if not self.margin_constant:
            improvement = (ema_loss - self.prev_ema_loss) / max(self.prev_ema_loss, EPS) if self.prev_ema_loss is not None else ema_loss
            if improvement > self.min_improvement:
                self.update_margin(torch.clamp(torch.tensor(self.margin * (1 + self.adjust_factor)), min=self.initial_margin, max=self.max_margin).item())
            else:
                self.update_margin(self.initial_margin)
        self.prev_ema_loss = ema_loss

        positive_d_loss = torch.mean(positive_distance)
        negative_d_loss = torch.mean(negative_distance)
        hard_negative_d_loss = torch.mean(hard_negative_distance)

        self.previous_loss = loss.item()
        return {"loss": loss, "positive_d_loss": positive_d_loss, "negative_d_loss": negative_d_loss, "hard_negative_d_loss": hard_negative_d_loss}



