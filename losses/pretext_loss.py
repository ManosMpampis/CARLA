import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.utilities import find_similarity_loss

EPS = 1e-8

class PretextLoss(nn.Module):
    # Based on the implementation of SupContrast
    def __init__(
        self,
        bs,
        temperature,
        loss_name="euclidean",
        device="cuda",
        normalize=True,
        clamp_neg_loss=True,
        crop=True,
        crop_size_min=5,
        crop_size_max=10,
        pos_weight=1,
        neg_weight=1,
        initial_margin=1.0,
        max_margin=5.0,
        min_margin=0.01,
        margin_distance=False,
        adjust_factor=0,
        ema_alpha=0,
        ema_distance=False,
        pos_supression=False,
        re_weight=False,
    ):
        super(PretextLoss, self).__init__()
        self.temperature = temperature
        self.bs = bs
        self.normalize = normalize
        self.clamp_neg_loss = clamp_neg_loss

        self.crop = crop
        self.crop_size_min = crop_size_min
        self.crop_size_max = crop_size_max

        self.margin = torch.tensor(initial_margin, device=device)
        self.initial_margin = torch.tensor(initial_margin, device=device)
        self.min_margin = torch.tensor(min_margin, device=device)
        self.max_margin = torch.tensor(max_margin, device=device)
        self.margin_distance = margin_distance

        self.pos_supression = pos_supression
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

        self.ema_distance = ema_distance
        self.ema_alpha = ema_alpha
        self.adjust_factor = adjust_factor

        self.re_weight = re_weight

        self.prev_ema_loss = None
        self.previous_loss = None

        self.sim_loss = find_similarity_loss(
            loss_name, device=device, use_cuda=True, temperature=temperature
        )

    def forward(self, features):
        """
        input:
            - features: hidden feature representation of shape [b, 3, dim]

        output:
            - loss: loss computed according to pretext triplet loss
        """
        anchor, positive, negative = torch.split(features, self.bs, dim=0)

        anchor = anchor.view(self.bs, -1, anchor.shape[1])
        positive = positive.view(self.bs, -1, positive.shape[1])
        negative = negative.view(self.bs, -1, negative.shape[1])

        # Normalize features for stable distance computation
        if self.normalize:
            anchor = F.normalize(anchor, dim=-1)
            positive = F.normalize(positive, dim=-1)
            negative = F.normalize(negative, dim=-1)

        pos_1 = []
        pos_2 = []
        neg_1 = []

        if self.crop:
            crop_a, crop_b = self.random_crop(anchor)
            pos_1.append(crop_a)
            pos_2.append(crop_b)
            crop_a, crop_b = self.random_crop(positive)
            pos_1.append(crop_a)
            pos_2.append(crop_b)
            crop_a, crop_b = self.random_crop(negative)
            neg_1.append(crop_a)
            neg_1.append(crop_b)
        else:
            pos_1.append(anchor)
            pos_2.append(positive)
            neg_1.append(negative)

        positive_distance = 0
        negative_distance = 0

        # Compute distances for each triplet
        for crop_a, crop_b, crop_n in zip(pos_1, pos_2, neg_1):
            positive_distance += self.sim_loss(crop_a, crop_b)
            negative_distance += self.sim_loss(crop_a, crop_n)

        positive_distance /= len(pos_1)
        negative_distance /= len(pos_1)

        # Update margin based on the current loss
        if self.margin_distance:
            self.update_margin(
                torch.clamp(
                    self.max_margin - torch.mean(negative_distance), min=self.min_margin
                )
            )

        # Calculate suppression of positive distance on the loss
        pos_supression_weight = 1 - (
            (
                (self.margin - (self.neg_weight * torch.mean(negative_distance)))
                / self.margin
            )
            * self.pos_supression
        )
        pos_supression_weight = torch.clamp(
            pos_supression_weight, min=0, max=self.neg_weight
        )

        positive_distance_c = (
            positive_distance * pos_supression_weight * self.pos_weight
        )

        negative_distance_c = negative_distance * self.neg_weight

        if self.clamp_neg_loss:
            negative_distance_c = torch.clamp(negative_distance_c, max=self.margin)
            loss = positive_distance_c - negative_distance_c
        else:
            loss = torch.clamp(
                self.margin + positive_distance_c - negative_distance_c, min=0.0
            )

        clear_loss = (positive_distance - negative_distance).mean()
        mask = loss > 0
        loss = torch.mean(loss)
        positive_d_loss = torch.mean(positive_distance)
        negative_d_loss = torch.mean(negative_distance)

        non_clamped = ~mask
        non_clamped_count = torch.sum(non_clamped)
        loss_pos_nc = torch.sum(positive_distance_c * non_clamped) / torch.clamp(non_clamped_count, min=1)
        loss_neg_nc = torch.sum(negative_distance_c * non_clamped) / torch.clamp(non_clamped_count, min=1)

        positive_distance_c = torch.mean(positive_distance_c) - loss_pos_nc
        negative_distance_c = torch.mean(negative_distance_c) - loss_neg_nc

        mask_number = torch.sum(mask)
        if mask_number > torch.tensor(0.0, device=loss.device) and self.re_weight:
            weight = self.bs / mask_number
            positive_distance_c *= weight
            negative_distance_c *= weight
            loss = self.margin + positive_distance_c - negative_distance_c

        if self.ema_distance:
            if self.previous_loss is not None:
                previous = self.previous_loss if torch.is_tensor(self.previous_loss) else torch.tensor(self.previous_loss, device=loss.device, dtype=loss.dtype)
                previous = previous.to(loss.device)
                ema_loss = (
                    torch.mean(
                        ((1.0 - self.ema_alpha) * loss) + (self.ema_alpha * previous)
                    )
                )
            else:
                ema_loss = loss
            if self.prev_ema_loss is not None:
                prev_ema = self.prev_ema_loss if torch.is_tensor(self.prev_ema_loss) else torch.tensor(self.prev_ema_loss, device=ema_loss.device, dtype=ema_loss.dtype)
                prev_ema = prev_ema.to(ema_loss.device)
                improvement = (
                    (prev_ema - ema_loss) / torch.clamp(prev_ema, min=EPS)
                )
            else:
                improvement = 0
            if torch.is_tensor(improvement):
                improvement = improvement.detach()
            self.update_margin(
                torch.clamp(
                    self.margin * (1 + improvement),
                    min=self.initial_margin,
                    max=self.max_margin,
                )
            )

            self.previous_loss = loss.detach()
            self.prev_ema_loss = ema_loss.detach()
        return {
            "loss": loss,
            "positive_d_loss": positive_d_loss,
            "negative_d_loss": negative_d_loss,
            "loss_pos_c": positive_distance_c,
            "loss_neg_c": negative_distance_c,
            "loss_pos_nc": loss_pos_nc,
            "loss_neg_nc": loss_neg_nc,
            "clear_loss": clear_loss,
        }

    def update_margin(self, new_margin):
        if new_margin is None:
            return
        self.margin = new_margin.to(self.margin.device) if isinstance(new_margin, torch.Tensor) else torch.tensor(new_margin).to(self.margin.device)
