import math

import torch
import torch.nn.functional as F


class FewShotNCALoss(torch.nn.Module):
    def __init__(
        self,
        classes,
        temperature,
        batch_size,
        frac_negative_samples=1,
        frac_positive_samples=1,
    ):
        super(FewShotNCALoss, self).__init__()
        self.temperature = torch.tensor(float(temperature), requires_grad=True).cuda()
        self.cls = classes
        self.batch_size = batch_size
        self.frac_negative_samples = frac_negative_samples
        self.frac_positive_samples = frac_positive_samples

    def forward(self, pred, target):
        n, d = pred.shape
        # identity matrix needed for masking matrix
        self.eye = torch.eye(target.shape[0]).cuda()

        # compute distance
        p_norm = torch.pow(torch.cdist(pred, pred), 2)
        # lower bound distances to avoid NaN errors
        p_norm[p_norm < 1e-10] = 1e-10
        dist = torch.exp(-1 * p_norm / self.temperature).cuda()

        # create matrix identifying all positive pairs
        bool_matrix = target[:, None] == target[:, None].T
        # substracting identity matrix removes positive pair with itself
        positives_matrix = (
            torch.tensor(bool_matrix, dtype=torch.int16).cuda() - self.eye
        ).cuda()
        # negative matrix is the opposite using ~ as not operator
        negatives_matrix = torch.tensor(~bool_matrix, dtype=torch.int16).cuda()

        denominators = torch.sum(dist * negatives_matrix, axis=0)

        numerators = torch.sum(dist * positives_matrix, axis=0)

        # avoiding nan errors
        denominators[denominators < 1e-10] = 1e-10
        frac = numerators / (numerators + denominators)

        loss = -1 * torch.sum(torch.log(frac[frac >= 1e-10])) / n

        return loss


class SupervisedContrastiveLoss(torch.nn.Module):
    def __init__(
        self, classes, temperature, batch_size, n_samples=0, frac_hard_samples=0
    ):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cls = classes
        self.batch_size = batch_size
        self.n_samples = n_samples

    def forward(self, pred, target):
        n, d = pred.shape
        # identity matrix needed
        self.eye = torch.eye(target.shape[0]).cuda()

        x1 = pred.unsqueeze(1).expand(n, n, d)
        x2 = pred.unsqueeze(0).expand(n, n, d)

        # compute pairwise distances between all points
        p_norm = F.cosine_similarity(x1, x2, dim=2)

        # lower bound distances to avoid NaN errors
        p_norm[p_norm < 1e-10] = 1e-10
        dist = torch.exp(p_norm / self.temperature).cuda()

        # create matrix identifying all positive pairs
        bool_matrix = target[:, None] == target[:, None].T
        # substracting identity matrix removes positive pair with itself
        positives_matrix = (
            torch.tensor(bool_matrix, dtype=torch.int32).cuda() - self.eye
        ).cuda()
        # negative matrix is the opposite
        negatives_matrix = torch.tensor(~bool_matrix, dtype=torch.int32).cuda()

        denominators = torch.sum(dist * negatives_matrix, axis=0)

        # compute numerators and denominators for NCA
        numerators = (dist * positives_matrix).cuda()

        # avoiding nan errors
        denominators[denominators < 1e-10] = 1e-10
        frac = numerators / denominators

        loss = (
            -1
            * torch.sum(torch.sum(torch.log(frac[frac >= 1e-10]), axis=0))
            / self.batch_size
        )

        return loss


class LGMLoss(torch.nn.Module):
    def __init__(self, num_classes, feat_dim=64, alpha=0.1, lambda_=0.01):
        super(LGMLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.lambda_ = lambda_
        self.means = torch.nn.Parameter(torch.randn(num_classes, feat_dim))
        torch.nn.init.xavier_uniform_(self.means, gain=math.sqrt(2.0))

    def forward(self, feat, labels):
        batch_size = feat.size()[0]

        XY = torch.matmul(feat, torch.transpose(self.means, 0, 1))
        XX = torch.sum(feat**2, dim=1, keepdim=True)
        YY = torch.sum(torch.transpose(self.means, 0, 1) ** 2, dim=0, keepdim=True)
        neg_sqr_dist = -0.5 * (XX - 2.0 * XY + YY)

        labels_reshaped = labels.view(labels.size()[0], -1)

        ALPHA = (
            torch.zeros(batch_size, self.num_classes)
            .cuda()
            .scatter_(1, labels_reshaped, self.alpha)
        )
        K = ALPHA + torch.ones([batch_size, self.num_classes]).cuda()

        logits_with_margin = torch.mul(neg_sqr_dist, K)

        return F.cross_entropy(logits_with_margin, labels)
