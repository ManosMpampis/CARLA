import torch
import numpy as np
from torch import Tensor

from utils.utils import AverageMeter, ProgressMeter


def pretext_train(train_loader, model, criterion, optimizer, epoch, prev_loss, logger=None):
    """ 
    Train epoch w/ pretext-Loss
    """
    total_l = AverageMeter('Total Loss', ':.4e')
    positive_l = AverageMeter('Positive Distance Loss', ':.4e')
    negative_l = AverageMeter('Negative Distance Loss', ':.4e')
    margin_l = AverageMeter('Margin', ':.4e')

    progress = ProgressMeter(len(train_loader),
        [total_l, positive_l, negative_l, margin_l],
        prefix="Epoch: [{}]".format(epoch+1),
        logger=logger)

    model.train()
    device = next(model.parameters()).device
    for i, batch in enumerate(train_loader):
        ts_org = batch['ts_org'].float().to(device, non_blocking=True)
        ts_w_augmented = batch['ts_w_augment'].float().to(device, non_blocking=True)
        ts_ss_augmented = batch['ts_ss_augment'].float().to(device, non_blocking=True)

        if ts_org.ndim == 3:
            b, w, h = ts_org.shape
        else:
            b, w = ts_org.shape
            h =1

        input_: Tensor = torch.cat([ts_org, ts_w_augmented, ts_ss_augmented], dim=0).view(b * 3, h, w)

        optimizer.zero_grad()

        output = model(input_)
        
        loss, positive_distance, hard_negative_distance = criterion(output, prev_loss)

        loss.backward()
        optimizer.step()

        margin = criterion.margin

        total_l.update(loss.item())
        positive_l.update(positive_distance.item())
        negative_l.update(hard_negative_distance.item())
        margin_l.update(margin)
        prev_loss = loss.item()
        
        if i % 10 == 0:
            progress.display(i)

    return {'Total Loss': total_l.avg, 'Positive Distance': positive_l.avg, 'Hard Negative Distance': negative_l.avg, 'margin': margin_l.avg}


def self_sup_classification_train(train_loader, model, criterion, optimizer, epoch, update_cluster_head_only=False, logger=None):
    """ 
    Train epoch w/ classification-Loss
    """
    total_losses = AverageMeter('Total Loss', ':.4e')
    consistency_losses = AverageMeter('Consistency Loss', ':.4e')
    inconsistency_losses = AverageMeter('Inconsistency Loss', ':.4e')
    entropy_losses = AverageMeter('Entropy', ':.4e')
    progress = ProgressMeter(len(train_loader),
        [total_losses, consistency_losses, inconsistency_losses, entropy_losses],
        prefix="Epoch: [{}]".format(epoch+1),
        logger=logger)

    if update_cluster_head_only:
        model.eval() # No need to update BN
    else:
        model.train() # Update BN
    device = next(model.parameters()).device

    for i, batch in enumerate(train_loader):
        # Forward pass
        anchors = batch['anchor'].to(device, non_blocking=True)
        nneighbors = batch['NNeighbor'].to(device, non_blocking=True)
        fneighbors = batch['FNeighbor'].to(device, non_blocking=True)

        if anchors.ndim == 3:
            b, w, h = anchors.shape
        else:
            b, w = anchors.shape
            h =1

        anchors = anchors.reshape(b, h, w)
        nneighbors = nneighbors.reshape(b, h, w)
        fneighbors = fneighbors.reshape(b, h, w)

        optimizer.zero_grad()

        if update_cluster_head_only: # Only calculate gradient for backprop of linear layer
            with torch.no_grad():
                anchors_features = model(anchors, forward_pass='backbone')
                nneighbors_features = model(nneighbors, forward_pass='backbone')
                fneighbors_features = model(fneighbors, forward_pass='backbone')

            anchors_output = model(anchors_features, forward_pass='head')
            nneighbors_output = model(nneighbors_features, forward_pass='head')
            fneighbors_output = model(fneighbors_features, forward_pass='head')

        else: # Calculate gradient for backprop of complete network
            anchors_output = model(anchors)
            nneighbors_output = model(nneighbors)
            fneighbors_output = model(fneighbors)

        # Loss for every head
        total_loss, consistency_loss, inconsistency_loss, entropy_loss = [], [], [], []
        
        total_loss, consistency_loss, inconsistency_loss, entropy_loss = criterion(anchors_output, nneighbors_output, fneighbors_output)

        # Aggregate losses and check for NaN
        assert(not torch.isnan(total_loss))
        assert(not torch.isnan(consistency_loss))
        assert(not torch.isnan(inconsistency_loss))
        assert(not torch.isnan(entropy_loss))

        total_losses.update(total_loss.item())
        consistency_losses.update(consistency_loss.item())
        inconsistency_losses.update(inconsistency_loss.item())
        entropy_losses.update(entropy_loss.item())

        assert total_loss.requires_grad, "Total loss does not require grad!"

        total_loss.backward()
        optimizer.step()

        if i % 100 == 0:
            progress.display(i)
    progress.display(i+1)
    return {"Total Loss": total_losses.avg, "Consistency Loss":consistency_losses.avg, "Incosistency Loss": inconsistency_losses.avg, "Entropy Loss": entropy_losses.avg}
