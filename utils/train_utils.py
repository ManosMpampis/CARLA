import torch
from torch import Tensor

from utils.utils import AverageMeter, ProgressMeter

device = torch.device("cuda")


def pretext_train(
    train_loader, model, criterion, optimizer, epoch, logger, device="cuda"
):

    avg_meters = {"meter_Margin": AverageMeter("Margin", ":.4e")}
    progress = ProgressMeter(
        len(train_loader),
        list(avg_meters.values()),
        logger,
        prefix="Epoch: [{}]".format(epoch + 1),
    )

    model.to(device)
    model.train()

    for i, batch in enumerate(train_loader):
        ts_org = batch["ts_org"].float().to(device, non_blocking=True)
        ts_w_augmented = batch["ts_w_augment"].float().to(device, non_blocking=True)
        ts_ss_augmented = batch["ts_ss_augment"].float().to(device, non_blocking=True)

        if ts_org.ndim == 3:
            b, w, h = ts_org.shape
        else:
            b, w = ts_org.shape
            h = 1

        # input_: Tensor = torch.cat(
        #     [ts_org, ts_w_augmented, ts_ss_augmented], dim=0
        # ).view(b * 3, h, w)

        optimizer.zero_grad()

        anch_out = model(ts_org.view(b, h, w))
        nn_out = model(ts_w_augmented.view(b, h, w))
        model.eval()
        fn_out = model(ts_ss_augmented.view(b, h, w))
        model.train()

        output: Tensor = torch.cat(
            [anch_out, nn_out, fn_out], dim=0
        )
        losses = criterion(output)

        for loss in losses.keys():
            if f"meter_{loss}" not in avg_meters.keys():
                # init meter if not exists
                avg_meters[f"meter_{loss}"] = AverageMeter(loss, ":.4e")
                progress.update(avg_meters[f"meter_{loss}"])
            avg_meters[f"meter_{loss}"].update(losses[loss].item())

        avg_meters["meter_Margin"].update(criterion.margin)

        losses["loss"].backward()
        optimizer.step()

        if i % 100 == 0:
            progress.display(i)

    return_dict = {key: avg_meters[f"meter_{key}"].avg for key in losses.keys()}
    return_dict["margin"] = avg_meters["meter_Margin"].avg
    return return_dict


def self_sup_classification_train(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    logger,
    update_cluster_head_only=False,
):
    """
    Train w/ classification-Loss
    """
    avg_meters = {}
    progress = ProgressMeter(
        len(train_loader), [], logger, prefix="Epoch: [{}]".format(epoch + 1)
    )

    if update_cluster_head_only:
        model.eval()  # No need to update BN
    else:
        model.train()  # Update BN

    for i, batch in enumerate(train_loader):
        # Forward pass
        anchors = torch.from_numpy(batch["anchor"]).to(device, non_blocking=True)
        nneighbors = torch.from_numpy(batch["NNeighbor"]).to(device, non_blocking=True)
        fneighbors = torch.from_numpy(batch["FNeighbor"]).to(device, non_blocking=True)

        if anchors.ndim == 3:
            b, w, h = anchors.shape
        else:
            b, w = anchors.shape
            h = 1

        anchors = anchors.reshape(b, h, w)
        nneighbors = nneighbors.reshape(b, h, w)
        fneighbors = fneighbors.reshape(b, h, w)

        optimizer.zero_grad()
        if (
            update_cluster_head_only
        ):  # Only calculate gradient for backprop of linear layer
            with torch.no_grad():
                anchors_features = model(anchors, forward_pass="backbone")
                nneighbors_features = model(nneighbors, forward_pass="backbone")
                fneighbors_features = model(fneighbors, forward_pass="backbone")

            anchors_output = model(anchors_features, forward_pass="head")
            nneighbors_output = model(nneighbors_features, forward_pass="head")
            model.eval()
            fneighbors_output = model(fneighbors_features, forward_pass="head")
            model.train()
        else:  # Calculate gradient for backprop of complete network
            anchors_output = model(anchors, forward_pass="return_all")
            nneighbors_output = model(nneighbors, forward_pass="return_all")
            model.eval()
            fneighbors_output = model(fneighbors, forward_pass="return_all")
            model.train()

        # Loss for every head
        losses = criterion(anchors_output, nneighbors_output, fneighbors_output)

        # Aggregate losses and check for NaN
        for loss in losses.keys():
            if f"meter_{loss}" not in avg_meters.keys():
                # init meter if not exists
                avg_meters[f"meter_{loss}"] = AverageMeter(loss, ":.4e")
                progress.update(avg_meters[f"meter_{loss}"])
            avg_meters[f"meter_{loss}"].update(losses[loss].item())

        assert losses["total_loss"].requires_grad, "Total loss does not require grad!"

        losses["total_loss"].backward()
        optimizer.step()
        if i % 100 == 0:
            progress.display(i)

    return_dict = {key: avg_meters[f"meter_{key}"].avg for key in losses.keys()}
    return return_dict
