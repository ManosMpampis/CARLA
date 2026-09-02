import torch
import torch.nn as nn

from losses.utilities import entropy

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