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
        classification_loss_flag=True
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
        b, n = anchors["cluster"].size()
        b = torch.tensor(b)
        n = torch.tensor(n)
        anchors_prob = self.softmax(anchors["cluster"])
        positives_prob = self.softmax(nneighbors["cluster"])
        negatives_prob = self.softmax(fneighbors["cluster"])

        # Per-sample predicted classes (used for per-class count logging)
        anchor_preds = torch.argmax(anchors_prob, dim=1)
        neg_preds = torch.argmax(negatives_prob, dim=1)

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
            diff_neg_sims = (neg_cross_sim.sum() - neg_cross_sim.diagonal().sum()) / (b_size * (b_size - 1))

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
        anchor_class_counts = torch.nn.functional.one_hot(anchor_preds, int(n)).sum(dim=0)
        negs_class_counts = torch.nn.functional.one_hot(neg_preds, int(n)).sum(dim=0)
        negs_classified_per_class = {
            f"{i+1}": negs_class_counts[i] for i in range(n)
        }
        anchor_classified_per_class = {
            f"{i+1}": anchor_class_counts[i] for i in range(n)
        }

        pos_normal_class_probs = torch.cat([anchors["output"], nneighbors["output"]])
        neg_normal_class_probs = fneighbors["output"]

        pos_targets = torch.zeros_like(pos_normal_class_probs)

        if anchors["output"].size(1) != 1:
            # Multi-class classification; regular class is fixed to class 0
            pos_targets[:, 0] = 1
            # Negatives: only the regular-class logit is constrained (to be off);
            # entropy/dissimilarity terms shape the remaining classes.
            neg_normal_logits = neg_normal_class_probs[:, 0]
            neg_bce_loss = self.bce_with_logits(
                neg_normal_logits, torch.zeros_like(neg_normal_logits)
            )
        else:
            neg_targets = torch.ones_like(neg_normal_class_probs)
            neg_bce_loss = self.bce_with_logits(neg_normal_class_probs, neg_targets)
        pos_bce_loss = self.bce_with_logits(pos_normal_class_probs, pos_targets)
        classification_loss = (pos_bce_loss + neg_bce_loss) / 2.0

        shift_weight = (
            (anchors_prob.std() - negatives_prob.std())
            * (n / torch.sqrt(n - 1))
            * self.classification_loss_flag
        ).detach()  # Pure scheduler: no gradient through the mixing coefficient
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

        # Per-sample predicted classes (used for per-class count logging)
        anchor_preds = torch.argmax(anchors_prob, dim=1)
        neg_preds = torch.argmax(negatives_prob, dim=1)

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
            diff_neg_sims = (neg_cross_sim.sum() - neg_cross_sim.diagonal().sum()) / (b_size * (b_size - 1))

        # Entropy loss
        entropy_loss = torch.tensor(0)
        negative_entropy = torch.tensor(0)
        negative_per_sample_entropy = torch.tensor(0)

        pos_logits = anchors["output"]
        if self.entropy_to_all_instances:
            anchors_prob = torch.cat([anchors_prob, positives_prob])
            pos_logits = torch.cat([pos_logits, nneighbors["output"]])
            self.positive_entropy_weight = 0.5  # Reduce the weight of positive entropy since we are also applying it to the neighbors
            negative_entropy = entropy(
                torch.mean(negatives_prob, 0), input_as_probabilities=True
            )
            # negative_per_sample_entropy = entropy(
            #     negatives_prob, input_as_probabilities=True
            # )

        normal_class_idx = torch.argmax(anchors_prob.mean(dim=0), dim=-1)
        normal_class_idx = normal_class_idx.view(1, 1)

        pos_targets = torch.zeros_like(pos_logits)
        pos_targets = pos_targets.scatter(1, normal_class_idx.expand(pos_logits.size(0), 1), 1.0)  # Regular class: most probable on average

        # Negatives: only the regular-class logit is constrained (to be off);
        # entropy/dissimilarity terms shape the remaining classes.
        neg_normal_logits = fneighbors["output"].gather(1, normal_class_idx.expand(fneighbors["output"].size(0), 1)).squeeze(1)

        pos_bce_loss = self.bce_with_logits(pos_logits, pos_targets)

        # neg_targets = torch.ones_like(negatives_prob)
        # neg_targets[:, normal_class_idx] = 0  # Regular class: most probable on average
        # neg_bce_loss = self.bce_with_logits(fneighbors["output"], neg_targets)
        neg_bce_loss = self.bce_with_logits(
            neg_normal_logits, torch.zeros_like(neg_normal_logits)
        )
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
        anchor_class_counts = torch.nn.functional.one_hot(anchor_preds, int(n)).sum(dim=0)
        negs_class_counts = torch.nn.functional.one_hot(neg_preds, int(n)).sum(dim=0)
        negs_classified_per_class = {
            f"{i+1}": negs_class_counts[i] for i in range(n)
        }
        anchor_classified_per_class = {
            f"{i+1}": anchor_class_counts[i] for i in range(n)
        }

        # shift_weight = (
        #     0.5 #positive = 0 και neg = 1
        #     * (-positive_entropy + negative_entropy)/torch.log(torch.tensor(n)) 
        #     * self.classification_loss_flag
        # ).detach()  # Pure scheduler: no gradient through the mixing coefficient
        # total_loss = (1 - torch.clamp(shift_weight, min=-1)) * marginal_total_loss + (torch.clamp(shift_weight, max=1) * classification_loss)

        shift_weight = (
            (anchors_prob.std() - negatives_prob.std())
            * (n / torch.sqrt(n - 1))
            * self.classification_loss_flag
        ).detach()  # Pure scheduler: no gradient through the mixing coefficient
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


class ClassificationLossMoCo(ClassificationLoss):
    """ClassificationLoss + MoCo-v2-style queues.

    Adds two FIFO queues of detached softmax probabilities to the base
    classification objective:

    - ``queue_neg``: past negative (synthetic anomaly) probabilities, used as
      extra negatives for the inconsistency term. ``queue_topk`` controls the
      usage: ``> 0`` keeps only the top-k hardest (most anchor-similar) queued
      negatives per anchor, ``0`` uses the full queue, ``< 0`` disables the
      queue inconsistency term.
    - ``queue_anchor``: past anchor probabilities, used to estimate the normal
      class from a large recent window instead of the per-batch argmax
      (``queue_anchor: True``). This stabilizes the classification target.

    Queue contents are always detached: gradients never flow into the queues.
    The queues are transient (``persistent=False``); they refill within a few
    steps after a resume. With ``queue_topk < 0`` and ``queue_anchor: False``
    the objective is numerically identical to ``ClassificationLoss``.
    """

    def __init__(
        self,
        *args,
        queue_size=8192,
        queue_topk=32,
        queue_warmup=0,
        queue_anchor=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.queue_size = queue_size
        self.queue_topk = queue_topk
        self.queue_warmup = queue_warmup
        self.queue_anchor = queue_anchor
        self._queues_ready = False
        self._ptr = 0
        self._filled = 0

    def _init_queues(self, num_classes, device, dtype):
        self.register_buffer(
            "queue_neg",
            torch.zeros(self.queue_size, num_classes, device=device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "queue_pos",
            torch.zeros(self.queue_size, num_classes, device=device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "queue_ptr", torch.zeros(1, dtype=torch.long, device=device), persistent=False
        )
        self.register_buffer(
            "queue_filled", torch.zeros(1, dtype=torch.long, device=device), persistent=False
        )
        self._queues_ready = True

    @torch.no_grad()
    def _enqueue(self, anchor_probs, neg_probs):
        if not self._queues_ready:
            self._init_queues(anchor_probs.size(1), anchor_probs.device, anchor_probs.dtype)
        b = anchor_probs.size(0)
        if b >= self.queue_size:  # keep only the most recent queue_size samples
            anchor_probs = anchor_probs[-self.queue_size:]
            neg_probs = neg_probs[-self.queue_size:]
            b = self.queue_size
        ptr = self._ptr
        idx = (ptr + torch.arange(b, device=anchor_probs.device)) % self.queue_size
        self.queue_pos[idx] = anchor_probs
        self.queue_neg[idx] = neg_probs
        self._ptr = (ptr + b) % self.queue_size
        self._filled = min(self._filled + b, self.queue_size)
        # Keep the (unused-for-logic) buffers in sync without a blocking host copy.
        self.queue_ptr.fill_(float(self._ptr))
        self.queue_filled.fill_(float(self._filled))

    def _queue_filled_enough(self):
        return self._queues_ready and self._filled >= max(
            self.queue_warmup, 1
        )

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

        # Detached per-sample copies for the queues (never carry gradients)
        anchors_prob_q = anchors_prob.detach()
        negatives_prob_q = negatives_prob.detach()

        # Per-sample predicted classes (used for per-class count logging)
        anchor_preds = torch.argmax(anchors_prob, dim=1)
        neg_preds = torch.argmax(negatives_prob, dim=1)

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

        # MoCo-style queue inconsistency: anchors vs past queued negatives.
        # topk > 0: hardest k queued negatives per anchor; 0: full queue; <0: off.
        queue_inconsistency = torch.zeros((), device=anchors_prob.device)
        filled = self._filled if self._queues_ready else 0
        if self.queue_topk >= 0 and self._queue_filled_enough():
            # Clone: the FIFO enqueue at the end of forward modifies the queue
            # buffers in-place, which would otherwise invalidate this graph.
            qneg = self.queue_neg[:filled].clone()
            queue_sims = torch.mm(anchors_prob, qneg.t())  # [b, filled]
            if self.queue_topk > 0:
                k = min(self.queue_topk, filled)
                queue_sims = queue_sims.topk(k, dim=1).values
            queue_inconsistency = self.bce(queue_sims, torch.zeros_like(queue_sims))
            inconsistency_loss = (inconsistency_loss + queue_inconsistency) / 2

        diff_neg_sims = torch.tensor(0)
        if self.disimilar_negatives:
            # Calculate similarity between ALL pairs of negatives in the batch
            # We don't want to penalize a sample's similarity with itself (the diagonal)
            # Create a mask to select only off-diagonal elements
            # Extract the similarities between DIFFERENT anomaly samples
            # Shape: [b, b]
            neg_cross_sim = torch.mm(negatives_prob, negatives_prob.t())
            b_size = neg_cross_sim.size(0)
            diff_neg_sims = (neg_cross_sim.sum() - neg_cross_sim.diagonal().sum()) / (b_size * (b_size - 1))

        # Entropy loss
        entropy_loss = torch.tensor(0)
        negative_entropy = torch.tensor(0)
        negative_per_sample_entropy = torch.tensor(0)

        pos_logits = anchors["output"]
        if self.entropy_to_all_instances:
            anchors_prob = torch.cat([anchors_prob, positives_prob])
            pos_logits = torch.cat([pos_logits, nneighbors["output"]])
            self.positive_entropy_weight = 0.5  # Reduce the weight of positive entropy since we are also applying it to the neighbors
            negative_entropy = entropy(
                torch.mean(negatives_prob, 0), input_as_probabilities=True
            )

        normal_class_idx = self.queue_pos[:filled].mean(dim=0).argmax() if (self.queue_anchor and self._queue_filled_enough()) else torch.argmax(anchors_prob.mean(dim=0), dim=-1)
        normal_class_idx_scalar = normal_class_idx.detach()
        normal_class_idx = normal_class_idx.view(1, 1)

        pos_targets = torch.zeros_like(pos_logits)
        pos_targets = pos_targets.scatter(1, normal_class_idx.expand(pos_logits.size(0), 1), 1.0)  # Regular class: most probable on average

        # Negatives: only the regular-class logit is constrained (to be off);
        # entropy/dissimilarity terms shape the remaining classes.
        neg_normal_logits = fneighbors["output"].gather(1, normal_class_idx.expand(fneighbors["output"].size(0), 1)).squeeze(1)

        pos_bce_loss = self.bce_with_logits(pos_logits, pos_targets)

        neg_bce_loss = self.bce_with_logits(
            neg_normal_logits, torch.zeros_like(neg_normal_logits)
        )
        classification_loss = (pos_bce_loss + neg_bce_loss) / 2.0

        anchors_prob = torch.mean(anchors_prob, 0)
        negatives_prob = torch.mean(negatives_prob, 0)

        positive_entropy = entropy(
            anchors_prob, input_as_probabilities=True
        ) * self.positive_entropy_weight
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
        anchor_class_counts = torch.nn.functional.one_hot(anchor_preds, int(n)).sum(dim=0)
        negs_class_counts = torch.nn.functional.one_hot(neg_preds, int(n)).sum(dim=0)
        negs_classified_per_class = {
            f"{i+1}": negs_class_counts[i] for i in range(n)
        }
        anchor_classified_per_class = {
            f"{i+1}": anchor_class_counts[i] for i in range(n)
        }

        shift_weight = (
            (anchors_prob.std() - negatives_prob.std())
            * (n / torch.sqrt(n - 1))
            * self.classification_loss_flag
        ).detach()  # Pure scheduler: no gradient through the mixing coefficient
        total_loss = (1 - shift_weight) * marginal_total_loss + (shift_weight * classification_loss)

        # FIFO update happens after the loss use: current batch is queued for future steps
        self._enqueue(anchors_prob_q, negatives_prob_q)

        out = {
            "total_loss": total_loss,
            "marginal_total_loss": marginal_total_loss,
            "classification_loss": classification_loss,
            "consistency_loss": consistency_loss,
            "inconsistency_loss": inconsistency_loss,
            "queue_inconsistency": queue_inconsistency,
            "queue_filled": torch.tensor(float(filled)),
            "normal_class_idx": normal_class_idx_scalar.float(),
            "entropy_loss": entropy_loss,
            "positive_entropy": positive_entropy,
            "negative_entropy": negative_entropy,
            "negative_per_sample": negative_per_sample_entropy,
            "diff_neg_sims": diff_neg_sims,
            "shift_weight": shift_weight,
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
