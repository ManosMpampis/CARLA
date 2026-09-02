import torch


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