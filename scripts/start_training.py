import datetime
import random
from pprint import PrettyPrinter

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

import src.models as models
from src.configs import configuration
from src.configs.get_configs import get_dataloader, get_optimizer, get_scheduler
from src.configs.load_yaml import load_dataset_yaml
from src.train.loss import FewShotNCALoss
from src.train.train import train
from src.utils.evaluation import (
    extract_and_evaluate,
    validate_loss,
)
from src.utils.logs import save_checkpoint
from src.utils.meters import BestAccuracySlots, warp_tqdm


def main():
    args = configuration.parser_args()

    num_classes, args.data, args.split_dir = load_dataset_yaml(args.dataset)

    # num_classes determined by load_dataset_yaml, except if set manually
    # done in cases when doing cross domain experiments
    if not args.num_classes:
        args.num_classes = num_classes

    # assert not repo.is_dirty(
    #    untracked_files=False), 'Please commit your changes before running any experiment (comment this code if you want to run experiments without commiting).'

    now = datetime.datetime.now()
    datetime_string = now.strftime("%y-%m-%d_%H-%M-%S")

    # print options as dictionary and save to output
    PrettyPrinter(indent=4).pprint(vars(args))

    # if seed is set, run will be deterministic for reproducability
    if args.seed is not None:
        print("\n>> Using fixed seed #" + str(args.seed))
        # Not fully deterministic, but without cudnn.benchmark is slower
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    else:
        cudnn.benchmark = True

    expm_id = datetime_string + args.expm_id
    tb_writer_train = SummaryWriter("../runs/" + expm_id + "/train/")
    tb_writer_val = SummaryWriter("../runs/" + expm_id + "/val/")

    # init meter to store best values
    best_accuracy_meter = BestAccuracySlots()

    # create model
    model = models.__dict__[args.arch](
        feature_dim=args.projection_feat_dim,
        num_classes=args.num_classes,
        projection=args.projection,
        use_fc=args.xent_weight > 0 or args.pretrained_model,
    )
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running DDP on rank {rank}.")

    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    model.to(device_id)
    model = DDP(model, device_ids=[device_id])
    # model = torch.nn.DataParallel(model).cuda()

    # define xent loss function (criterion) and optimizer
    xent = nn.CrossEntropyLoss().cuda()

    print("\n>> Number of CUDA devices: " + str(torch.cuda.device_count()))

    loss = FewShotNCALoss(
        args.num_classes,
        batch_size=args.batch_size,
        temperature=args.temperature,
    ).cuda()
    # loss_norm used for computing validation NCA loss
    loss_norm = FewShotNCALoss(
        args.num_classes,
        batch_size=args.batch_size,
        temperature=args.temperature,
    ).cuda()

    # train loader is different when training protonets, due to batch creation

    train_loader = get_dataloader(
        "train", args, not args.disable_train_augment, shuffle=True
    )

    # init train loader used for centering
    train_loader_for_avg = get_dataloader(
        "train", args, aug=False, shuffle=False, out_name=False
    )

    # init standard validation and test loader
    val_loader = get_dataloader("val", args, aug=False, shuffle=False, out_name=False)
    test_loader = get_dataloader("test", args, aug=False, shuffle=False, out_name=False)

    # init optimizer and scheduler
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(len(train_loader), optimizer, args)

    tqdm_loop = warp_tqdm(list(range(args.start_epoch, args.epochs)), args)
    for epoch in tqdm_loop:
        # train for one epoch
        train(
            train_loader,
            model,
            xent,
            loss,
            optimizer,
            epoch,
            scheduler,
            tb_writer_train,
            args,
        )
        scheduler.step(epoch)

        # evaluate on meta validation set
        if (epoch + 1) % args.val_interval == 0:
            # compute loss on the validation set
            nca_loss_val, xent_loss_val = validate_loss(
                val_loader, model, loss_norm, xent, args
            )
            tb_writer_val.add_scalar(
                "Loss/NCA_Val", nca_loss_val, epoch * len(train_loader)
            )
            if args.xent_weight > 0:
                tb_writer_val.add_scalar(
                    "Loss/X-entropy", xent_loss_val, epoch * len(train_loader)
                )
            # full evaluation on the val set
            shot1_info, shot5_info = extract_and_evaluate(
                model,
                train_loader_for_avg,
                val_loader,
                "val",
                args,
                writer=tb_writer_val,
                t=epoch * len(train_loader),
            )

            # update best accuracies
            is_best1, is_best5 = best_accuracy_meter.update(shot1_info, shot5_info)

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "scheduler": scheduler.state_dict(),
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "best_accuracies": best_accuracy_meter,
                    "optimizer": optimizer.state_dict(),
                },
                is_best1,
                is_best5,
                folder=args.save_path,
                filename=expm_id,
            )

    # print the best epoch accuracies for the val set
    tb_writer_val.add_scalar("val/1-shot/Best-CL2N", best_accuracy_meter.cl2n_1shot, 0)
    tb_writer_val.add_scalar("val/5-shot/Best-CL2N", best_accuracy_meter.cl2n_5shot, 0)

    # at the end of training, evaluate on the VAL set with the best performing epoch of the model just trained
    # on more epochs than during training
    """
    extract_and_evaluate(
        model,
        train_loader_for_avg,
        val_loader,
        "val",
        args,
        model_name=expm_id,
        writer=tb_writer_val,
        t=0,
        print_stdout=True,
        expm_id=expm_id,
        num_iter=args.test_iter,  # set number of iter to same as test iter (higher than val iter)
    )

    # at the end of training, evaluate on the TEST set with the best performing epoch of the model just trained
    extract_and_evaluate(
        model,
        train_loader_for_avg,
        test_loader,
        "test",
        args,
        model_name=expm_id,
        writer=tb_writer_val,
        t=0,
        print_stdout=True,
        expm_id=expm_id,
    )
    """
    tb_writer_train.close()
    tb_writer_val.close()


if __name__ == "__main__":
    main()
