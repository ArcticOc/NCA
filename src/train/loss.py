import torch
import torch.nn.functional as F

INF = 1e3


def pmax(D, M, T):
    mask = torch.where(M, 0.0, -INF)
    return torch.logsumexp(D / T + mask, dim=-1) * T


def pmin(D, M, T):
    mask = torch.where(M, 0.0, -INF)
    return -torch.logsumexp(-D / T + mask, dim=-1) * T


class FewShotNCALoss(torch.nn.Module):
    def __init__(
        self,
        classes,
        temperature,
        batch_size,
    ):
        super().__init__()
        self.temperature = torch.tensor(float(temperature), requires_grad=True).cuda()
        self.cls = classes
        self.batch_size = batch_size

    def forward(self, pred, target):
        n, d = pred.shape
        # identity matrix needed for masking matrix
        # self.eye = torch.eye(target.shape[0]).cuda()
        self.eye = torch.eye(target.size(0), dtype=torch.bool).cuda()

        # compute distance
        p_norm = torch.pow(torch.cdist(pred, pred), 2)
        # lower bound distances to avoid NaN errors
        # p_norm[p_norm < 1e-10] = 1e-10
        dist = -0.5 * p_norm / self.temperature
        # dist = torch.exp(-1 * p_norm / self.temperature).cuda()
        # create matrix identifying all positive pairs
        bool_matrix = target[:, None] == target[:, None].T
        # substracting identity matrix removes positive pair with itself
        # positives_matrix = (bool_matrix.clone().detach().type(torch.int16).cuda() - self.eye)
        positives_matrix = (bool_matrix & ~self.eye).cuda()
        # negative matrix is the opposite using ~ as not operator
        # negatives_matrix = (~bool_matrix).clone().detach().type(torch.int16).cuda()
        negatives_matrix = (~bool_matrix).cuda()
        # sampling random elements for the negatives

        # denominators = torch.sum(dist * negatives_matrix, axis=0)
        denominators = pmax(dist, negatives_matrix, 1)
        # numerators = torch.sum(dist_m * positives_matrix, axis=0)
        # numerators = torch.sum(dist * positives_matrix, axis=0)
        numerators = pmin(dist, positives_matrix, 2)
        # avoiding nan errors
        # denominators[denominators < 1e-10] = 1e-10
        # temp = numerators * denominators
        # frac = 1 / (1 + numerators * denominators)
        # frac = numerators / (numerators + denominators)

        # loss = -1 * torch.sum(torch.log(frac[frac >= 1e-10])) / n
        loss = F.softplus(denominators - numerators).mean()

        return loss


class LGMLoss(torch.nn.Module):
    pass


class SupervisedContrastiveLoss:
    pass
