import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import find_similarity_loss, entropy

EPS = 1e-8


class GCLoss(torch.nn.Module):
    """Global Contrastive Learning loss.
    Based on this implementation
    https://github.com/emadeldeen24/TS-TCC/blob/main/models/loss.py
    """

    def __init__(self, device, temperature, class_num=2):
        """
        Args:
            device (torch.device): Device.
            temperature (float): Temperature parameter.
            class_num (int, optional): Number of classes. Defaults to 2.
        """
        super(GCLoss, self).__init__()
        self.temperature = temperature
        self.device = device

        self.class_num = class_num
        self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_nt_xent_loss(self, z_pos_1, z_pos_2, z_neg):
        """Compute NT-Xent loss to the batch of vectors/

        Args:
            z_pos_1 (torch.Tensor): Batch of positive vectors.
            z_pos_2 (torch.Tensor): Batch of second positive vectors.
            z_neg (list[torch.Tensor], optional): Batch of negative vectors.. Defaults to None.

        Returns:
            torch.Tensor: Loss value.
        """
        representations = torch.cat([z_pos_1, z_pos_2], dim=0)
        batch_size = z_pos_1.size(0)
        # print(representations.shape)
        # print(batch_size)
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        sim_matrix_neg = []
        sim_matrix_neg.append(
            self._cosine_similarity(z_pos_1.unsqueeze(0), z_neg.unsqueeze(0))
        )
        sim_matrix_neg.append(
            self._cosine_similarity(z_pos_2.unsqueeze(0), z_neg.unsqueeze(0))
        )

        non_neg_values = torch.cat(sim_matrix_neg).view(2 * batch_size, -1)

        # Compute similarity matrix between positive views
        similarity_matrix = self._cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0)
        )
        # print(similarity_matrix.shape)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1
        )

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1
        )
        negatives = torch.cat([non_neg_values, negatives], dim=1).view(
            2 * batch_size, -1
        )

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(logits.shape[0]).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (logits.shape[0])

    def forward(self, anchors, nneighbors, fneighbors):
        """Forward pass.

        Args:
            z_pos_1 (torch.Tensor): Batch of positive vectors.
            z_pos_2 (torch.Tensor): Batch of second positive vectors.
            z_neg (list[torch.Tensor], optional): Batch of negative vectors.. Defaults to None.
            cluster(bool, optional): Apply clustering.

        Returns:
            torch.Tensor: Loss value.
        """
        loss = self._get_nt_xent_loss(anchors, nneighbors, fneighbors)
        return loss

class ClassificationLossE2E(nn.Module):
    def __init__(
        self,
        entropy_weight=2.0,
        inconsistency_weight=1.0,
        consistency_weight=1.0,
        entropy_norm=False,
        entropy_to_all_instances=False,
        disimilar_negatives=False,
    ):
        super(ClassificationLossE2E, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.bce = nn.BCELoss()
        self.bce_with_logits = nn.BCEWithLogitsLoss()
        self.entropy_weight = entropy_weight
        self.inconsistency_weight = inconsistency_weight
        self.consistency_weight = consistency_weight
        self.entropy_norm = entropy_norm
        self.entropy_to_all_instances = entropy_to_all_instances
        self.disimilar_negatives = disimilar_negatives
        self.positive_entropy_weight = 1.0

    def forward(self, anchors, nneighbors, fneighbors):
        """
        input:
            - anchors: logits for anchor ts w/ shape [b, num_classes]
            - k nearest neighbors: logits for neighbor ts w/ shape [b, num_classes]
            - k furthest neighbors: logits for neighbor ts w/ shape [b, num_classes]

        output:
            - Loss
        """
        b, n = anchors["cluster"].size()
        b = torch.tensor(b)
        n = torch.tensor(n)
        anchors_prob = self.softmax(anchors["cluster"])
        positives_prob = self.softmax(nneighbors["cluster"])
        negatives_prob = self.softmax(fneighbors["cluster"])

        # Similarity in output space
        similarity = torch.bmm(
            anchors_prob.view(b, 1, n), positives_prob.view(b, n, 1)
        ).squeeze()
        ones = torch.ones_like(similarity)
        consistency_loss = self.bce(similarity, ones)

        # DiSimilarity in output space
        negsimilarities = []
        negsimilarities.append(
            torch.bmm(
                anchors_prob.view(b, 1, n), negatives_prob.view(b, n, 1)
            ).squeeze()
        )

        zeros = torch.zeros_like(ones)

        # DiSimilarity with the near-neighbors
        if self.entropy_to_all_instances:
            negsimilarities.append(
                torch.bmm(
                    positives_prob.view(b, 1, n), negatives_prob.view(b, n, 1)
                ).squeeze()
            )

        inconsistency_loss = 0
        for negsimilarity in negsimilarities:
            inconsistency_loss += self.bce(negsimilarity, zeros)
        inconsistency_loss /= len(negsimilarities)

        diff_neg_sims = torch.tensor(0)
        if self.disimilar_negatives:
            # Calculate similarity between ALL pairs of negatives in the batch
            # We don't want to penalize a sample's similarity with itself (the diagonal)
            # Create a mask to select only off-diagonal elements
            # Extract the similarities between DIFFERENT anomaly samples
            # Shape: [b, b]
            neg_cross_sim = torch.mm(negatives_prob, negatives_prob.t())
            b_size = neg_cross_sim.size(0)
            mask = ~torch.eye(b_size, device=neg_cross_sim.device).bool()
            diff_neg_sims = neg_cross_sim[mask].mean()

        # Entropy loss
        entropy_loss = torch.tensor(0)
        negative_entropy = torch.tensor(0)
        negative_per_sample_entropy = torch.tensor(0)
        if self.entropy_to_all_instances:
            anchors_prob = torch.cat([anchors_prob, positives_prob])
            self.positive_entropy_weight = 0.5  # Reduce the weight of positive entropy since we are also applying it to the neighbors
            negative_entropy = entropy(
                torch.mean(negatives_prob, 0), input_as_probabilities=True
            )
            # negative_per_sample_entropy = entropy(
            #     negatives_prob, input_as_probabilities=True
            # )

        pos_classification = torch.argmax(anchors_prob, dim=0)
        negs_classification = torch.argmax(negatives_prob, dim=0)

        anchors_prob = torch.mean(anchors_prob, 0)
        negatives_prob = torch.mean(negatives_prob, 0)
        
        positive_entropy = entropy(
            anchors_prob, input_as_probabilities=True
        ) * self.positive_entropy_weight
        # Entropy shifts from -2.4 to 0. It is already output to -entropy.
        # Entropy closly to 2.4 mean that the inputs are equally seperated.
        # Entropy closly to 0 means that inputs are classified to one class only.
        # Entropy is subtracted from loss. We want to minim positive_entropy and maximize negative entropy.
        entropy_loss = positive_entropy - negative_entropy
        if self.entropy_norm:
            entropy_loss /= torch.log(torch.tensor(n))  # Normalize to 1

        # Total loss
        marginal_total_loss = (
            (self.consistency_weight * consistency_loss)
            + (self.entropy_weight * entropy_loss)
            + (self.entropy_weight * negative_per_sample_entropy)
            + (self.entropy_weight * diff_neg_sims)
            + (self.inconsistency_weight * inconsistency_loss)
        )

        anchor_margin_per_class = { 
            f"{i+1}": anchors_prob[i] for i in range(n)
        }
        negs_margin_per_class = { 
            f"{i+1}": negatives_prob[i] for i in range(n)
        }
        negs_classified_per_class = {
            f"{i+1}": negs_classification[i] for i in range(n)
        }
        anchor_classified_per_class = {
            f"{i+1}": pos_classification[i] for i in range(n)
        }

        pos_normal_class_probs = torch.cat([anchors["output"], nneighbors["output"]])
        neg_normal_class_probs = fneighbors["output"]

        pos_targets = torch.zeros_like(pos_normal_class_probs)
        pos_targets[:, 0] = 1  # Normal class is class 0

        neg_targets = torch.ones_like(neg_normal_class_probs)
        neg_targets[:, 0] = 0  # Anomalous class is class 1
        pos_bce_loss = self.bce_with_logits(pos_normal_class_probs, pos_targets)
        neg_bce_loss = self.bce_with_logits(neg_normal_class_probs, neg_targets)
        classification_loss = (pos_bce_loss + neg_bce_loss) / 2.0

        shift_weight = (anchors_prob.std() - negatives_prob.std())*(n/torch.sqrt(n-1)).detach()
        total_loss = (1 - shift_weight) * marginal_total_loss + (shift_weight) * classification_loss

        out = {
            "total_loss": total_loss,
            "marginal_total_loss": marginal_total_loss,
            "classification_loss": classification_loss,
            "consistency_loss": consistency_loss,
            "inconsistency_loss": inconsistency_loss,
            "entropy_loss": entropy_loss,
            "positive_entropy": positive_entropy,
            "negative_entropy": negative_entropy,
            "negative_per_sample": negative_per_sample_entropy,
            "diff_neg_sims": diff_neg_sims,
            "shift_weight": shift_weight
        }
        for cls in anchor_margin_per_class.keys():
            out[f"marginal_anchors_cls{cls}"] = anchor_margin_per_class[cls]
            out[f"marginal_negs_cls{cls}"] = negs_margin_per_class[cls]
            out[f"classified_negs_cls{cls}"] = negs_classified_per_class[cls]
            out[f"classified_anchors_cls{cls}"] = anchor_classified_per_class[cls]
            
        return out

class ClassificationLoss(nn.Module):
    def __init__(
        self,
        entropy_weight=2.0,
        inconsistency_weight=1.0,
        consistency_weight=1.0,
        entropy_norm=False,
        entropy_to_all_instances=False,
        disimilar_negatives=False,
        classification_loss_flag=True
    ):
        super(ClassificationLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.bce = nn.BCELoss()
        self.bce_with_logits = nn.BCEWithLogitsLoss()
        self.entropy_weight = entropy_weight
        self.inconsistency_weight = inconsistency_weight
        self.consistency_weight = consistency_weight
        self.entropy_norm = entropy_norm
        self.entropy_to_all_instances = entropy_to_all_instances
        self.disimilar_negatives = disimilar_negatives
        self.positive_entropy_weight = 1.0
        self.classification_loss_flag = classification_loss_flag

    def forward(self, anchors, nneighbors, fneighbors):
        """
        input:
            - anchors: logits for anchor ts w/ shape [b, num_classes]
            - k nearest neighbors: logits for neighbor ts w/ shape [b, num_classes]
            - k furthest neighbors: logits for neighbor ts w/ shape [b, num_classes]

        output:
            - Loss
        """
        b, n = anchors["output"].size()
        b = torch.tensor(b)
        n = torch.tensor(n)
        anchors_prob = self.softmax(anchors["output"])
        positives_prob = self.softmax(nneighbors["output"])
        negatives_prob = self.softmax(fneighbors["output"])

        # Similarity in output space
        similarity = torch.bmm(
            anchors_prob.view(b, 1, n), positives_prob.view(b, n, 1)
        ).squeeze()
        ones = torch.ones_like(similarity)
        consistency_loss = self.bce(similarity, ones)

        # DiSimilarity in output space
        negsimilarities = []
        negsimilarities.append(
            torch.bmm(
                anchors_prob.view(b, 1, n), negatives_prob.view(b, n, 1)
            ).squeeze()
        )

        zeros = torch.zeros_like(ones)

        # DiSimilarity with the near-neighbors
        if self.entropy_to_all_instances:
            negsimilarities.append(
                torch.bmm(
                    positives_prob.view(b, 1, n), negatives_prob.view(b, n, 1)
                ).squeeze()
            )

        inconsistency_loss = 0
        for negsimilarity in negsimilarities:
            inconsistency_loss += self.bce(negsimilarity, zeros)
        inconsistency_loss /= len(negsimilarities)

        diff_neg_sims = torch.tensor(0)
        if self.disimilar_negatives:
            # Calculate similarity between ALL pairs of negatives in the batch
            # We don't want to penalize a sample's similarity with itself (the diagonal)
            # Create a mask to select only off-diagonal elements
            # Extract the similarities between DIFFERENT anomaly samples
            # Shape: [b, b]
            neg_cross_sim = torch.mm(negatives_prob, negatives_prob.t())
            b_size = neg_cross_sim.size(0)
            mask = ~torch.eye(b_size, device=neg_cross_sim.device).bool()
            diff_neg_sims = neg_cross_sim[mask].mean()

        # Entropy loss
        entropy_loss = torch.tensor(0)
        negative_entropy = torch.tensor(0)
        negative_per_sample_entropy = torch.tensor(0)
        if self.entropy_to_all_instances:
            anchors_prob = torch.cat([anchors_prob, positives_prob])
            self.positive_entropy_weight = 0.5  # Reduce the weight of positive entropy since we are also applying it to the neighbors
            negative_entropy = entropy(
                torch.mean(negatives_prob, 0), input_as_probabilities=True
            )
            # negative_per_sample_entropy = entropy(
            #     negatives_prob, input_as_probabilities=True
            # )

        pos_classification = torch.argmax(anchors_prob, dim=0)
        negs_classification = torch.argmax(negatives_prob, dim=0)
        

        normal_class_idx = torch.argmax(anchors_prob.mean(dim=0), dim=-1).item()
        pos_targets = torch.zeros_like(anchors_prob)
        pos_targets[:, normal_class_idx] = 1  # Normal class is class 0

        neg_targets = torch.ones_like(negatives_prob)
        neg_targets[:, normal_class_idx] = 0  # Anomalous class is class 1

        # pos_normal_class_probs = anchors_prob[:, normal_class_idx]
        # neg_normal_class_probs = negatives_prob[:, normal_class_idx]

        pos_bce_loss = self.bce_with_logits(torch.cat([anchors["output"], nneighbors["output"]]), pos_targets)
        neg_bce_loss = self.bce_with_logits(fneighbors["output"], neg_targets)
        classification_loss = (pos_bce_loss + neg_bce_loss) / 2.0

        anchors_prob = torch.mean(anchors_prob, 0)
        negatives_prob = torch.mean(negatives_prob, 0)
        
        positive_entropy = entropy(
            anchors_prob, input_as_probabilities=True
        ) * self.positive_entropy_weight
        # Entropy shifts from -2.4 to 0. It is already output to -entropy.
        # Entropy closly to 2.4 mean that the inputs are equally seperated.
        # Entropy closly to 0 means that inputs are classified to one class only.
        # Entropy is subtracted from loss. We want to minim positive_entropy and maximize negative entropy.
        entropy_loss = positive_entropy - negative_entropy
        if self.entropy_norm:
            entropy_loss /= torch.log(torch.tensor(n))  # Normalize to 1

        # Total loss
        marginal_total_loss = (
            (self.consistency_weight * consistency_loss)
            + (self.entropy_weight * entropy_loss)
            + (self.entropy_weight * negative_per_sample_entropy)
            + (self.entropy_weight * diff_neg_sims)
            + (self.inconsistency_weight * inconsistency_loss)
        )

        anchor_margin_per_class = { 
            f"{i+1}": anchors_prob[i] for i in range(n)
        }
        negs_margin_per_class = { 
            f"{i+1}": negatives_prob[i] for i in range(n)
        }
        negs_classified_per_class = {
            f"{i+1}": negs_classification[i] for i in range(n)
        }
        anchor_classified_per_class = {
            f"{i+1}": pos_classification[i] for i in range(n)
        }


        shift_weight = (anchors_prob.std() - negatives_prob.std())*(n/torch.sqrt(n-1)).detach() * self.classification_loss_flag
        total_loss = (1 - shift_weight) * marginal_total_loss + (shift_weight * classification_loss)

        out = {
            "total_loss": total_loss,
            "marginal_total_loss": marginal_total_loss,
            "classification_loss": classification_loss,
            "consistency_loss": consistency_loss,
            "inconsistency_loss": inconsistency_loss,
            "entropy_loss": entropy_loss,
            "positive_entropy": positive_entropy,
            "negative_entropy": negative_entropy,
            "negative_per_sample": negative_per_sample_entropy,
            "diff_neg_sims": diff_neg_sims,
            "shift_weight": shift_weight
        }
        for cls in anchor_margin_per_class.keys():
            out[f"marginal_anchors_cls{cls}"] = anchor_margin_per_class[cls]
            out[f"marginal_negs_cls{cls}"] = negs_margin_per_class[cls]
            out[f"classified_negs_cls{cls}"] = negs_classified_per_class[cls]
            out[f"classified_anchors_cls{cls}"] = anchor_classified_per_class[cls]
            
        return out


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
        min_improvement=0.0001,
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

        self.margin = initial_margin
        self.initial_margin = initial_margin
        self.min_margin = min_margin
        self.max_margin = max_margin
        self.margin_distance = margin_distance

        self.pos_supression = pos_supression
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

        self.min_improvement = min_improvement
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

        loss_pos_nc = torch.mean(positive_distance_c[~mask])
        loss_neg_nc = torch.mean(negative_distance_c[~mask])

        positive_distance_c = torch.mean(positive_distance_c[mask])
        negative_distance_c = torch.mean(negative_distance_c[mask])

        mask_number = torch.sum(mask)
        if mask_number > 0 and self.re_weight:
            weight = self.bs / mask_number
            positive_distance_c *= weight
            negative_distance_c *= weight
            loss = self.margin + positive_distance_c - negative_distance_c

        ema_loss = (
            torch.mean(
                ((1.0 - self.ema_alpha) * loss) + (self.ema_alpha * self.previous_loss)
            )
            if self.previous_loss is not None
            else loss
        )
        improvement = (
            (self.prev_ema_loss - ema_loss) / max(self.prev_ema_loss, EPS)
            if self.prev_ema_loss is not None
            else ema_loss
        )
        self.update_margin(
            torch.clamp(
                self.margin * (1 + improvement),
                min=self.initial_margin,
                max=self.max_margin,
            ).item()
        )

        self.previous_loss = loss.item()
        self.prev_ema_loss = ema_loss
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
        self.margin = new_margin
