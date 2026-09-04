import torch
from torch import Tensor

from utils.utils import AverageMeter, ProgressMeter
from utils.evaluate_utils import GradientMonitor


@torch.no_grad()
def update_ema_model(model, model_ema, momentum):
    """Momentum (EMA) update of the key encoder, MoCo-style.

    Parameters are EMA-updated; buffers (e.g. BatchNorm running statistics)
    are copied directly from the live model.
    """
    for p_live, p_ema in zip(model.parameters(), model_ema.parameters()):
        p_ema.mul_(momentum).add_(p_live.detach(), alpha=1 - momentum)
    for b_live, b_ema in zip(model.buffers(), model_ema.buffers()):
        b_ema.copy_(b_live)

def pretext_train(
    train_loader, model, criterion, optimizer, epoch, logger, device="cuda", gradient_monitor: GradientMonitor = None
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

        optimizer.zero_grad()

        anch_out = model(ts_org.reshape(b, h, w))
        nn_out = model(ts_w_augmented.reshape(b, h, w))
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
        if (gradient_monitor is not None) and (i == ((len(train_loader.dataset)//b)-1 if train_loader.drop_last else len(train_loader.dataset)//b)):
            gradient_monitor.step()
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
    device="cuda",
    gradient_monitor: GradientMonitor = None,
    model_ema=None,
    ema_momentum=0.999,
    ema_negatives=False,
):
    """
    Train w/ classification-Loss

    If ``model_ema`` is given, it is momentum-updated after every optimizer
    step. ``ema_negatives`` selects the negative branch: ``False`` keeps the
    default behaviour (live model in eval() mode, keeping the loss graph of
    the negatives so the model learns to process them); ``True`` forwards the
    negatives through the EMA encoder under no_grad (consistent negatives for
    the MoCo-style queues, but no gradient learning on the negative branch).
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
        anchors = batch["anchor"].to(device, non_blocking=True)
        nneighbors = batch["NNeighbor"].to(device, non_blocking=True)
        fneighbors = batch["FNeighbor"].to(device, non_blocking=True)

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
            if ema_negatives and model_ema is not None:
                with torch.no_grad():
                    fneighbors_ema_features = model_ema(fneighbors, forward_pass="backbone")
                    fneighbors_output = model_ema(fneighbors_ema_features, forward_pass="head")
            else:
                model.eval()
                fneighbors_output = model(fneighbors_features, forward_pass="head")
                model.train()
        else:  # Calculate gradient for backprop of complete network
            anchors_output = model(anchors, forward_pass="return_all")
            nneighbors_output = model(nneighbors, forward_pass="return_all")
            if ema_negatives and model_ema is not None:
                with torch.no_grad():
                    fneighbors_output = model_ema(fneighbors, forward_pass="return_all")
            else:
                model.eval()
                fneighbors_output = model(fneighbors, forward_pass="return_all")
                model.train()

        # Loss for every head
        # FNeighbor_mask: timesteps of the FNeighbor windows where a synthetic
        # sub-anomaly was injected (target of the auxiliary localization head).
        fneighbor_mask = batch.get("FNeighbor_mask")
        if fneighbor_mask is not None:
            fneighbor_mask = fneighbor_mask.to(device, non_blocking=True).float()
        losses = criterion(anchors_output, nneighbors_output, fneighbors_output, fneighbor_mask=fneighbor_mask)

        # Aggregate losses and check for NaN
        for loss in losses.keys():
            if f"meter_{loss}" not in avg_meters.keys():
                # init meter if not exists
                avg_meters[f"meter_{loss}"] = AverageMeter(loss, ":.4e")
                progress.update(avg_meters[f"meter_{loss}"])
            avg_meters[f"meter_{loss}"].update(losses[loss].item())

        assert losses["total_loss"].requires_grad, "Total loss does not require grad!"

        losses["total_loss"].backward()
        if gradient_monitor is not None:
            gradient_monitor.step()
        optimizer.step()
        if model_ema is not None:
            update_ema_model(model, model_ema, ema_momentum)
        if i % 100 == 0:
            progress.display(i)

    return_dict = {key: avg_meters[f"meter_{key}"].avg for key in losses.keys()}
    return return_dict
