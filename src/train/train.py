import time

import torch

from src.utils.evaluation import AverageMeter, warp_tqdm


def train(
    train_loader,
    model,
    xent_f,
    nca_f,
    optimizer,
    epoch,
    scheduler,
    tb_writer_train,
    args,
):
    """
    Function that performs 1 training iteration.
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    nca_losses, xent_losses = AverageMeter(), AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    tqdm_train_loader = warp_tqdm(train_loader, args)
    for i, (input, target) in enumerate(tqdm_train_loader):
        if args.scheduler == "cosine":
            scheduler.step(epoch * len(train_loader) + i)

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        # we have to shuffle the batch for multi-gpu training purposeswhen training with protonets
        shuffle = torch.randperm(input.shape[0])
        input = input[shuffle]

        # idx keeps track on how to shuffle back to original order
        idx = torch.argsort(shuffle)

        # get features
        features, output = model(input, use_fc=False)

        # shuffle back so protonet loss still works
        features = features[idx]

        nca_loss = nca_f(features, target)
        total_loss = nca_loss

        if not args.disable_tqdm:
            tqdm_train_loader.set_description(
                "NCA loss (train): {:.3f}".format(nca_losses.avg)
            )

        nca_losses.update(nca_loss.item(), input.size(0))
        tb_writer_train.add_scalar(
            "Loss/NCA", nca_loss.item(), epoch * len(train_loader) + i
        )

        # compute gradient and do gradient step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


# FIXME: This function is not used anywhere
def query_support_nca_loss(output, target_support, target_query, args):
    """
    Function to compute the NCA loss between a support and query set.
    This is the same as the NCA loss, except that
    not all pairs of points are considered, but only pairs between points from
    the query set and points from the support set.
    Instead, the standard NCA loss would compute pairs between all points

    :param output (torch.Tensor): shape=(n_support, n_query) pairwise
                                   distances between support and query set
    :param target_support (torch.Tensor): shape=(n_support) target for
                                   each support embedding in batch
    :param target_query (torch.Tensor): shape=(n_query) target for
                                   each query embedding in batch
    :param args: command line arguments

    :return loss: returns computed loss
    """

    # compute distances
    dist = torch.exp(output).cuda()

    # compute boolean matrix to find positive and negative pairs
    bool_matrix = target_support[:, None] == target_query[:, None].T
    # construct positive pairs mask
    positives_matrix = torch.tensor(bool_matrix, dtype=torch.int16).cuda()
    # negative pairs mask is the opposite
    negatives_matrix = torch.tensor(~bool_matrix, dtype=torch.int16).cuda()
    # compute terms for denominator and numerator
    denominators = torch.sum(dist * negatives_matrix, axis=0)
    numerators = torch.sum(dist * positives_matrix, axis=0).cuda()

    # avoiding nan errors
    denominators[denominators < 1e-10] = 1e-10

    frac = numerators / (numerators + denominators)
    loss = (
        -1 * torch.sum(torch.log(frac[frac >= 1e-10]), axis=0) / target_query.shape[0]
    )

    return loss
