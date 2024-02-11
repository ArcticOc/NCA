import time

import torch

from src.utils.evaluation import AverageMeter, warp_tqdm


def train(
    train_loader,
    model,
    nca_f,
    optimizer,
    epoch,
    scheduler,
    tb_writer_train,
    args,
):
    # Function that performs 1 training iteration.

    batch_time = AverageMeter()
    data_time = AverageMeter()
    nca_losses = AverageMeter()

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
        """ for param in model.parameters():
            if param.requires_grad:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM) """

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
