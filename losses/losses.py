import torch
import torch.nn as nn

from losses.utilities import entropy

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
        total_loss = marginal_total_loss + (torch.clamp(shift_weight, max=1) *4* classification_loss)

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

class ClassificationLossPart(nn.Module):
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
            super(ClassificationLossPart, self).__init__()
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
    
    def forward(self, anchors, fneighbors):
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
        negatives_prob = self.softmax(fneighbors["output"])
        
        # Per-sample predicted classes (used for per-class count logging)
        anchor_preds = torch.argmax(anchors_prob, dim=1)
        neg_preds = torch.argmax(negatives_prob, dim=1)
    
        pos_logits = anchors["output"]
        
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
        
        out = {
            "total_loss": classification_loss
        }
        
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
