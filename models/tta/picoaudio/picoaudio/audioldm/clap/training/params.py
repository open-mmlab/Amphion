import argparse
import os

CACHE_DIR = os.getenv(
    "AUDIOLDM_CACHE_DIR",
    "~/.cache")



def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to h5 filewith training data",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to h5 file with validation data",
    )
    parser.add_argument(
        "--freeze-text",
        default=False,
        action="store_true",
        help="if you need to freeze the text encoder, make this True",
    )
    parser.add_argument(
        "--freeze-text-after",
        type=int,
        default=-1,
        help="if you need to freeze the text encoder after (include) epoch x, set this param to x. Set -1 to disable it",
    )
    parser.add_argument(
        "--train-ipc",
        type=str,
        default=None,
        help="Path to npy file of the number of instance per class in training data",
    )
    parser.add_argument(
        "--val-ipc",
        type=str,
        default=None,
        help="Path to npy file of the number of instance per class in validation data",
    )
    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Required for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--val-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Useful for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["webdataset", "csv", "auto", "toy"],
        default="auto",
        help="Which type of dataset to process.",
    )
    parser.add_argument(
        "--csv-separator",
        type=str,
        default="\t",
        help="For csv-like datasets, which separator to use.",
    )
    parser.add_argument(
        "--csv-img-key",
        type=str,
        default="filepath",
        help="For csv-like datasets, the name of the key for the image paths.",
    )
    parser.add_argument(
        "--csv-caption-key",
        type=str,
        default="title",
        help="For csv-like datasets, the name of the key for the captions.",
    )
    parser.add_argument(
        "--imagenet-val",
        type=str,
        default=None,
        help="Path to imagenet val set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet-v2",
        type=str,
        default=None,
        help="Path to imagenet v2 for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--datasetnames",
        nargs="+",
        default=None,
        help="If loading webdataset, spedify the dataset names to load. Can be some of these: Clotho, audioset, audiocaps, BBCSoundEffects",
    )
    parser.add_argument(
        "--full-train-dataset",
        nargs="+",
        default=None,
        help="Which dataset will be trained with all the subsets. (train+test)",
    )
    parser.add_argument(
        "--exclude-eval-dataset",
        nargs="+",
        default=None,
        help="Which dataset will be excluded with evaluation",
    )
    parser.add_argument(
        "--datasetinfos",
        nargs="+",
        default=None,
        help="If loading webdataset, spedify the dataset types to load. Can be some of these: train, test, valid, unbalanced_train, balanced_train, eval",
    )
    parser.add_argument(
        "--dataset-proportion",
        type=float,
        default=1.0,
        help="How much proportion of dataset we want to train.",
    )
    parser.add_argument(
        "--remotedata",
        default=False,
        action="store_true",
        help="if the dataset is remote, set this flag",
    )
    parser.add_argument(
        "--class-label-path",
        type=str,
        default=None,
        help="The path of the class label pickle or csv.",
    )
    parser.add_argument(
        "--datasetpath",
        type=str,
        default="/mnt/audio_clip/webdataset_tar",
        help="The path to the dataset",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--log-local",
        action="store_true",
        default=False,
        help="log files on local master, otherwise global master only.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=32, help="Number of epochs to train for."
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--momentum", type=float, default=None, help="SGD epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")

    parser.add_argument(
        "--split-opt",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--lr-pretrained", type=float, default=None, help="Learning rate for text."
    )
    parser.add_argument(
        "--beta1-pretrained", type=float, default=None, help="Adam beta 1 for text."
    )
    parser.add_argument(
        "--beta2-pretrained", type=float, default=None, help="Adam beta 2 for text."
    )
    parser.add_argument(
        "--eps-pretrained", type=float, default=None, help="Adam epsilon for text."
    )
    parser.add_argument(
        "--wd-pretrained", type=float, default=0.2, help="Weight decay for text."
    )
    parser.add_argument(
        "--momentum-pretrained", type=float, default=0.9, help="Momentum for text."
    )
    parser.add_argument(
        "--lr-new", type=float, default=None, help="Learning rate for audio."
    )
    parser.add_argument(
        "--beta1-new", type=float, default=None, help="Adam beta 1 for audio."
    )
    parser.add_argument(
        "--beta2-new", type=float, default=None, help="Adam beta 2 for audio."
    )
    parser.add_argument(
        "--eps-new", type=float, default=None, help="Adam epsilon for audio."
    )
    parser.add_argument(
        "--wd-new", type=float, default=0.2, help="Weight decay for audio."
    )
    parser.add_argument(
        "--momentum-new", type=float, default=0.9, help="Momentum for audio."
    )
    parser.add_argument(
        "--warmup", type=int, default=10000, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.",
    )
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--save-top-performance",
        type=int,
        default=0,
        help="Save the top x performance weights if the value >0",
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=False,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--zeroshot-frequency", type=int, default=2, help="How often to run zero shot."
    )
    parser.add_argument(
        "--val-frequency",
        type=int,
        default=1,
        help="How often to run evaluation with val data.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precision.",
    )
    parser.add_argument(
        "--amodel",
        type=str,
        default="RN50",
        help="Name of the audio backbone to use.",
    )
    parser.add_argument(
        "--tmodel",
        type=str,
        default="transformer",
        help="Name of the text backbone to use. Can be [transformer, bert, roberta, bart]",
    )
    parser.add_argument(
        "--pretrained-audio",
        default="",
        type=str,
        help="Use a pretrained audio model weights for the audio encoder of CLAP",
    )
    parser.add_argument(
        "--pretrained-text",
        default="",
        type=str,
        help="Use a pretrained text model weights for the text encoder of CLAP",
    )
    parser.add_argument(
        "--pretrained",
        default="",
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--pretrained-image",
        default=False,
        action="store_true",
        help="Load imagenet pretrained weights for image tower backbone if available.",
    )
    parser.add_argument(
        "--lock-image",
        default=False,
        action="store_true",
        help="Lock full image tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-image-unlocked-groups",
        type=int,
        default=0,
        help="Leave last n image tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-image-freeze-bn-stats",
        default=False,
        action="store_true",
        help="Freeze BatchNorm running stats in image tower for any locked layers.",
    )
    parser.add_argument(
        "--local-loss",
        default=False,
        action="store_true",
        help="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)",
    )
    parser.add_argument(
        "--gather-with-grad",
        default=False,
        action="store_true",
        help="enable full distributed gradient for feature gather",
    )
    parser.add_argument(
        "--force-quick-gelu",
        default=False,
        action="store_true",
        help="Force use of QuickGELU activation for non-OpenAI transformer models.",
    )
    parser.add_argument(
        "--torchscript",
        default=False,
        action="store_true",
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
    )
    parser.add_argument(
        "--trace",
        default=False,
        action="store_true",
        help="torch.jit.trace the model for inference / eval only",
    )
    # arguments for distributed training
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--report-to",
        default="",
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']",
    )
    parser.add_argument(
        "--wandb-notes", default="", type=str, help="Notes if logging with wandb"
    )
    parser.add_argument(
        "--C", type=float, default=3.16, help="inverse regularizer for logistic reg."
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged.",
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, we copy the entire base on the log diretory, and execute from there.",
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--ddp-static-graph",
        default=False,
        action="store_true",
        help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    parser.add_argument("--seed", type=int, default=4242, help="Default random seed.")

    parser.add_argument(
        "--top-k-checkpoint-select-dataset",
        type=str,
        default="all",
        help="The dataset of selecting top-k checkpoint.",
    )

    # @R10, @R@5, @R1, mAP@10
    parser.add_argument(
        "--top-k-checkpoint-select-metric",
        type=str,
        default="_R@10",
        help="The metric for selecting top-k checkpoint.",
    )
    parser.add_argument(
        "--openai-model-cache-dir",
        type=str,
        default=f"{CACHE_DIR}/clip",
        help="Directory to download OpenAI models.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        help="can be AdamW or SGD",
    )
    parser.add_argument(
        "--parallel-eval",
        default=False,
        action="store_true",
        help="Eval in parallel (multi-GPU, multi-node).",
    )

    parser.add_argument(
        "--no-eval",
        default=False,
        action="store_true",
        help="Training without evaluation.",
    )

    parser.add_argument(
        "--lp-mlp",
        default=False,
        action="store_true",
        help="Linear Probe using MLP layer or not.",
    )

    parser.add_argument(
        "--lp-freeze",
        default=False,
        action="store_true",
        help="Linear Probe using Freeze CLAP or not",
    )

    parser.add_argument(
        "--lp-act",
        default="None",
        type=str,
        help="Options are ['relu','elu','prelu','softmax','sigmoid']",
    )

    parser.add_argument(
        "--lp-loss", type=str, default="bce", help="Loss func of Linear Probe."
    )

    parser.add_argument(
        "--lp-metrics",
        type=str,
        default="map,mauc,acc",
        help="Metrics of Linear Probe.",
    )

    parser.add_argument(
        "--lp-lr", type=float, default=1e-4, help="learning rate of linear probe"
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=0,
        help="the kappa in the weighted contrastive loss, default is to turn off the weighted contrastive loss",
    )

    parser.add_argument(
        "--data-filling",
        type=str,
        default="pad",
        help="type of data filling when the audio length is shorter than the max length."
        "Can be one of the following: repeat, repeatpad, pad",
    )
    parser.add_argument(
        "--data-truncating",
        type=str,
        default="rand_trunc",
        help="type of data truncation when the audio length is longer than the max length."
        "Can be one of the following: rand_trunc, fusion",
    )

    parser.add_argument(
        "--clap-mlploss",
        default=False,
        action="store_true",
        help="Using MLP loss for CLAP model or not",
    )

    parser.add_argument(
        "--wandb-id",
        type=str,
        default=None,
        help="the id of wandb experiment to restore.",
    )

    parser.add_argument(
        "--sleep", type=float, default=0, help="sleep n seconds before start training"
    )

    # variable length processing
    parser.add_argument(
        "--enable-fusion",
        default=False,
        action="store_true",
        help="Enable feature funsion for variable-length data",
    )

    parser.add_argument(
        "--fusion-type",
        type=str,
        default="None",
        help="Type is among ['channel_map', 'daf_1d','aff_1d','iaff_1d','daf_2d','aff_2d','iaff_2d']",
    )

    parser.add_argument(
        "--mixup",
        default=False,
        action="store_true",
        help="Enable mixup in finetuning training.",
    )
    parser.add_argument(
        "--text-augment-selection",
        type=str,
        default=None,
        help="For selecting levels of augmented text. Type is among ['all', 'augment_only', 'none']",
    )

    args = parser.parse_args()

    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.amodel)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    return args
