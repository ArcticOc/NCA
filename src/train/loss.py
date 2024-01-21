import torch
import torch.nn.functional as F

INF = 1e10


def mean(D, M, T):
    mask = torch.where(M, 1.0, 0.0)
    masked_sum = (D * mask).sum(dim=-1)
    count = mask.sum(dim=-1)

    zero_count_mask = count == 0

    safe_count = torch.clamp(count, min=1)
    average = masked_sum / safe_count
    average = torch.where(zero_count_mask, -INF, average)
    return average


def pmax(D, M, T):
    mask = torch.where(M, 0.0, -INF)
    return torch.logsumexp(D / T + mask, dim=-1) * T


def pmin(D, M, T):
    mask = torch.where(M, 0.0, -INF)
    return -torch.logsumexp(-D / T + mask, dim=-1) * T


# Compute prototypes
def compute_prototypes(pred, target):
    prototypes = {}
    for class_index in torch.unique(target):
        class_mask = target == class_index
        class_features = pred[class_mask]
        prototypes[class_index.item()] = torch.mean(class_features, axis=0)

    return prototypes


def cosine_similarity(samples, prototypes, labels):
    for key in prototypes:
        prototypes[key] = F.normalize(prototypes[key].unsqueeze(0), p=2, dim=1).squeeze(
            0
        )

    samples_normalized = F.normalize(samples, p=2, dim=1)

    relevant_prototypes = torch.stack([prototypes[label.item()] for label in labels])

    cosine_sim = torch.sum(samples_normalized * relevant_prototypes, dim=1)

    return cosine_sim


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
        self.eye = torch.eye(target.shape[0]).cuda()

        # compute distance
        p_norm = torch.pow(torch.cdist(pred, pred), 2)
        # lower bound distances to avoid NaN errors
        p_norm[p_norm < 1e-10] = 1e-10
        dist = torch.exp(-1 * p_norm / self.temperature).cuda()
        # create matrix identifying all positive pairs
        bool_matrix = (target[:, None] == target[:, None].T).cuda()
        # substracting identity matrix removes positive pair with itself
        positives_matrix = (
            bool_matrix.clone().detach().type(torch.int16).cuda() - self.eye
        )
        # negative matrix is the opposite using ~ as not operator
        negatives_matrix = (~bool_matrix).clone().detach().type(torch.int16).cuda()
        # sampling random elements for the negatives
        denominators = torch.sum(dist * negatives_matrix, axis=0)
        # numerators = torch.sum(dist_m * positives_matrix, axis=0)
        numerators = torch.sum(dist * positives_matrix, axis=0)
        # avoiding nan errors
        denominators[denominators < 1e-10] = 1e-10
        # frac = 1 / (1 + numerators * denominators)
        frac = numerators / (numerators + denominators)

        loss = -1 * torch.sum(torch.log(frac[frac >= 1e-10])) / n

        return loss


"""     def forward(self, pred, target):
        n, d = pred.shape
        # identity matrix needed for masking matrix
        self.eye = torch.eye(target.size(0), dtype=torch.bool).cuda()
        # prototypes = compute_prototypes(pred, target)
        # cos_sim = cosine_similarity(pred, prototypes, target)

        # compute distance
        p_norm = torch.pow(torch.cdist(pred, pred), 2)
        # lower bound distances to avoid NaN errors
        dist = -0.5 * p_norm / self.temperature

        # create matrix identifying all positive pairs
        bool_matrix = (target[:, None] == target[:, None].T).cuda()
        # substracting identity matrix removes positive pair with itself

        positives_matrix = (bool_matrix & ~self.eye).cuda()
        # negative matrix is the opposite using ~ as not operator

        negatives_matrix = (~bool_matrix).cuda()
        # sampling random elements for the negatives

        # denominators = pmax(dist, negatives_matrix, 1)
        denominators = mean(dist, negatives_matrix, 1)
        # numerators = pmin(dist, positives_matrix, 2)
        numerators = mean(dist, positives_matrix, 2)

        _temp = denominators - numerators
        loss = F.softplus(denominators - numerators).mean()

        return loss """
