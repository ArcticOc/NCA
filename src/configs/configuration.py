import configargparse

# Constants for default values
DEFAULT_DATASET = "miniimagenet"
DEFAULT_WORKERS = 60
DEFAULT_EPOCHS = 120
DEFAULT_NUM_CLASSES = 0
DEFAULT_START_EPOCH = 0
DEFAULT_BATCH_SIZE = 512
DEFAULT_DISABLE_TRAIN_AUGMENT = False
DEFAULT_DISABLE_RANDOM_RESIZE = False
DEFAULT_ENLARGE = False
DEFAULT_ARCH = "resnet12"
DEFAULT_SCHEDULER = "multi_step"
DEFAULT_LR = 0.1
DEFAULT_LR_STEPSIZE = 30
DEFAULT_LR_MILESTONES = "0.7"
DEFAULT_LR_GAMMA = 0.1
DEFAULT_OPTIMIZER = "SGD"
DEFAULT_WEIGHT_DECAY = 5e-4
DEFAULT_TEMPERATURE = 1
DEFAULT_NESTEROV = False
DEFAULT_OPTIMIZE_TEMPERATURE = False
DEFAULT_PROJECTION = False
DEFAULT_PROJECTION_FEAT_DIM = 512
DEFAULT_MULTI_LAYER_EVAL = False
DEFAULT_TEST_ITER = 10000
DEFAULT_VAL_ITER = 3000
DEFAULT_TEST_WAY = 5
DEFAULT_TEST_QUERY = 15
DEFAULT_VAL_INTERVAL = 10
DEFAULT_NUM_NN = 1
DEFAULT_EXPM_ID = "test"
DEFAULT_SAVE_ID = ""
DEFAULT_SAVE_PATH = "../results"
DEFAULT_SEED = 0
DEFAULT_DISABLE_TQDM = False
DEFAULT_SOFT_ASSIGNMENT = False
DEFAULT_CONTRASTIVELOSS = False
DEFAULT_REPLACEMENT_SAMPLING = False
DEFAULT_RESUME_MODEL = ""
DEFAULT_EPISODE_NO_REPLACEMENT_SAMPLING = True
DEFAULT_EVALUATE_MODEL = ""
DEFAULT_EVALUATE_ALL_SHOTS = 0
DEFAULT_XENT_WEIGHT = 0
DEFAULT_NEGATIVES_FRAC_RANDOM = 1
DEFAULT_POSITIVES_FRAC_RANDOM = 1
DEFAULT_PRETRAINED_MODEL = None
DEFAULT_PROTO_TRAIN = False
DEFAULT_PROTO_TRAIN_ITER = 100
DEFAULT_PROTO_TRAIN_WAY = 30
DEFAULT_PROTO_TRAIN_SHOT = 1
DEFAULT_PROTO_TRAIN_QUERY = 15
DEFAULT_PROTO_DISABLE_AGGREGATES = False
DEFAULT_PROTO_ENABLE_ALL_PAIRS = False
DEFAULT_MEDIAN_PROTOTYPE = False
DEFAULT_EPISODE_OPTIMIZE = False


def parser_args():
    parser = configargparse.ArgParser()

    # Dataset and training configuration
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        choices=("miniimagenet", "tieredimagenet", "CIFARFS"),
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=DEFAULT_WORKERS,
        type=int,
        metavar="N",
        help="number of data loading workers",
    )
    parser.add_argument(
        "--epochs",
        default=DEFAULT_EPOCHS,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--num-classes",
        default=DEFAULT_NUM_CLASSES,
        type=int,
        metavar="N",
        help="number of classes in the dataset",
    )
    parser.add_argument(
        "--start-epoch",
        default=DEFAULT_START_EPOCH,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=DEFAULT_BATCH_SIZE,
        type=int,
        metavar="N",
        help="mini-batch size",
    )
    parser.add_argument(
        "--disable-train-augment",
        action="store_true",
        default=DEFAULT_DISABLE_TRAIN_AUGMENT,
        help="disable training augmentation",
    )
    parser.add_argument(
        "--disable-random-resize",
        action="store_true",
        default=DEFAULT_DISABLE_RANDOM_RESIZE,
        help="disable random resizing",
    )
    parser.add_argument(
        "--enlarge",
        action="store_true",
        default=DEFAULT_ENLARGE,
        help="enlarge the image size then center crop",
    )

    # Model configuration
    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default=DEFAULT_ARCH,
        help="network architecture",
    )
    parser.add_argument(
        "--scheduler",
        default=DEFAULT_SCHEDULER,
        choices=("step", "multi_step"),
        help="scheduler, the detail is shown in train.py",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=DEFAULT_LR,
        type=float,
        metavar="LR",
        help="initial learning rate",
    )
    parser.add_argument(
        "--lr-stepsize",
        default=DEFAULT_LR_STEPSIZE,
        type=int,
        help='learning rate decay step size (for "step" scheduler)',
    )
    parser.add_argument(
        "--lr-milestones",
        default=DEFAULT_LR_MILESTONES,
        help='Fractions of total epochs at which decay the LR (for "multi_step" scheduler). Separate values using commas ,',
    )
    parser.add_argument(
        "--lr-gamma",
        default=DEFAULT_LR_GAMMA,
        type=float,
        help="gamma for learning rate decay",
    )
    parser.add_argument(
        "--optimizer", default=DEFAULT_OPTIMIZER, choices=("SGD", "Adam")
    )
    parser.add_argument(
        "--weight-decay",
        "--wd",
        default=DEFAULT_WEIGHT_DECAY,
        type=float,
        metavar="W",
        help="weight decay (L2 penalty)",
    )
    parser.add_argument("--temperature", default=DEFAULT_TEMPERATURE, type=float)
    parser.add_argument(
        "--nesterov",
        action="store_true",
        default=DEFAULT_NESTEROV,
        help="use nesterov for SGD, disable it in default",
    )
    parser.add_argument(
        "--optimize-temperature",
        action="store_true",
        default=DEFAULT_OPTIMIZE_TEMPERATURE,
        help="optimize temperature parameter of NCA using SGD during training",
    )

    # Projection configuration
    parser.add_argument(
        "--projection",
        action="store_true",
        default=DEFAULT_PROJECTION,
        help="Use projection network",
    )
    parser.add_argument(
        "--projection-feat-dim",
        type=int,
        default=DEFAULT_PROJECTION_FEAT_DIM,
        help="Feature dimensionality of output of projection network",
    )
    parser.add_argument(
        "--multi-layer-eval",
        action="store_true",
        default=DEFAULT_MULTI_LAYER_EVAL,
        help="Use earlier layers as embedding during evalution",
    )

    # Testing and validation configuration
    parser.add_argument(
        "--test-iter",
        type=int,
        default=DEFAULT_TEST_ITER,
        help="number of iterations on test set",
    )
    parser.add_argument(
        "--val-iter",
        type=int,
        default=DEFAULT_VAL_ITER,
        help="number of iterations on val set",
    )
    parser.add_argument(
        "--test-way",
        type=int,
        default=DEFAULT_TEST_WAY,
        help="number of ways during val/test",
    )
    parser.add_argument(
        "--test-query",
        type=int,
        default=DEFAULT_TEST_QUERY,
        help="number of queries during val/test",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=DEFAULT_VAL_INTERVAL,
        help="evaluate every X epochs",
    )
    parser.add_argument(
        "--num-NN",
        type=int,
        default=DEFAULT_NUM_NN,
        help="number of nearest neighbors, set this number >1 when doing kNN",
    )

    # Experiment and logging configuration
    parser.add_argument(
        "--expm-id",
        default=DEFAULT_EXPM_ID,
        type=str,
        help="experiment name for logging results",
    )
    parser.add_argument(
        "--save-id",
        default=DEFAULT_SAVE_ID,
        type=str,
        help="argument for a save file to save results for --evaluate-all-shots",
    )
    parser.add_argument(
        "--save-path",
        default=DEFAULT_SAVE_PATH,
        type=str,
        help="path to folder stored the log and checkpoint",
    )
    parser.add_argument(
        "--seed",
        default=DEFAULT_SEED,
        type=int,
        help="seed for initializing training. ",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        default=DEFAULT_DISABLE_TQDM,
        help="disable tqdm.",
    )
    parser.add_argument(
        "--soft-assignment",
        action="store_true",
        default=DEFAULT_SOFT_ASSIGNMENT,
        help="use soft assignment for multiple shot classification.",
    )
    parser.add_argument(
        "--contrastiveloss",
        action="store_true",
        default=DEFAULT_CONTRASTIVELOSS,
        help="Use the supervised contrastive loss instead of NCA",
    )
    parser.add_argument(
        "--replacement-sampling",
        action="store_true",
        default=DEFAULT_REPLACEMENT_SAMPLING,
        help="for non-episodic batch generation, sample with replacement (number of batches needs to be set by proto-train-iter argument)",
    )
    parser.add_argument(
        "--resume-model",
        default=DEFAULT_RESUME_MODEL,
        type=str,
        metavar="PATH",
        help="path to latest checkpoint of the model you want to resume training",
    )
    parser.add_argument(
        "--episode-no-replacement-sampling",
        action="store_false",
        default=DEFAULT_EPISODE_NO_REPLACEMENT_SAMPLING,
        help="for episode batch generation, sample batches with replacement by default, if triggered episodes are sampled without replacement",
    )
    parser.add_argument(
        "--evaluate-model",
        default=DEFAULT_EVALUATE_MODEL,
        type=str,
        metavar="PATH",
        help="path to the model to evaluate on test set",
    )
    parser.add_argument(
        "--evaluate-all-shots",
        type=int,
        default=DEFAULT_EVALUATE_ALL_SHOTS,
        help="do evaluation over multiple values of shot",
    )
    parser.add_argument(
        "--xent-weight",
        default=DEFAULT_XENT_WEIGHT,
        type=float,
        help="part of the loss that should consist of cross entropy training",
    )
    parser.add_argument(
        "--negatives-frac-random",
        default=DEFAULT_NEGATIVES_FRAC_RANDOM,
        type=float,
        help="fraction of random negatives to sample for the NCA loss",
    )
    parser.add_argument(
        "--positives-frac-random",
        default=DEFAULT_POSITIVES_FRAC_RANDOM,
        type=float,
        help="fraction of random positives to sample for the NCA loss",
    )
    parser.add_argument(
        "--pretrained-model",
        default=DEFAULT_PRETRAINED_MODEL,
        type=str,
        help="path to the pretrained model",
    )

    # Prototype network configuration
    parser.add_argument(
        "--proto-train",
        action="store_true",
        default=DEFAULT_PROTO_TRAIN,
        help="do prototypical training",
    )
    parser.add_argument(
        "--proto-train-iter",
        type=int,
        default=DEFAULT_PROTO_TRAIN_ITER,
        help="number of iterations for proto train",
    )
    parser.add_argument(
        "--proto-train-way",
        type=int,
        default=DEFAULT_PROTO_TRAIN_WAY,
        help="number of ways for protonet training",
    )
    parser.add_argument(
        "--proto-train-shot",
        type=int,
        default=DEFAULT_PROTO_TRAIN_SHOT,
        help="number of shots for protonet training",
    )
    parser.add_argument(
        "--proto-train-query",
        type=int,
        default=DEFAULT_PROTO_TRAIN_QUERY,
        help="number of queries for protonet training",
    )
    parser.add_argument(
        "--proto-disable-aggregates",
        action="store_true",
        default=DEFAULT_PROTO_DISABLE_AGGREGATES,
        help="disables the construction of aggregates in prototypical networks",
    )
    parser.add_argument(
        "--proto-enable-all-pairs",
        action="store_true",
        default=DEFAULT_PROTO_ENABLE_ALL_PAIRS,
        help="disregards the split between the query and support set and computes all pairs of distances",
    )
    parser.add_argument(
        "--median-prototype",
        action="store_true",
        default=DEFAULT_MEDIAN_PROTOTYPE,
        help="use median instead of mean for computing prototypes",
    )
    parser.add_argument(
        "--episode-optimize",
        action="store_true",
        default=DEFAULT_EPISODE_OPTIMIZE,
        help="optimize model on support set during evaluation",
    )

    return parser.parse_args()
