# Jets Recipe

In this recipe, we will show how to train [Jets](https://arxiv.org/abs/2203.16852) using Amphion's infrastructure. Jets is an end-to-end text-to-speech (E2E-TTS) model which jointly trains FastSpeech2 and HiFi-GAN.

There are four stages in total:

1. Data preparation
2. Features extraction
3. Training
4. Inference

> **NOTE:** You need to run every command of this recipe in the `Amphion` root path:
>
> ```bash
> cd Amphion
> ```

## 1. Data Preparation

### Dataset Download

You can use LJSpeech to train TTS model. How to download dataset is detailed [here](../../datasets/README.md).

### Configuration

After downloading the dataset, you can set the dataset paths in  `exp_config.json`. Note that you can change the `dataset` list to use your preferred datasets.

```json
    "dataset": [
        "LJSpeech",
    ],
    "dataset_path": {
        // TODO: Fill in your dataset path
        "LJSpeech": "[LJSpeech dataset path]",
    },
```

## 2. Features Extraction

### Configuration

Specify the `processed_dir` and the `log_dir` and for saving the processed data and the checkpoints in `exp_config.json`:

```json
    // TODO: Fill in the output log path
    "log_dir": "ckpts/tts",
    "preprocess": {
        // TODO: Fill in the output data path
        "processed_dir": "data",
        ...
    },
```

### Run

Run the `run.sh` as the preproces stage (set  `--stage 1`):

```bash
sh egs/tts/Jets/run.sh --stage 1
```

## 3. Training

### Configuration

We provide the default hyparameters in the `exp_config.json`. They can work on single NVIDIA-24g GPU. You can adjust them based on your GPU machines.

```
"train": {
        "batch_size": 16,
    }
```

### Run

Run the `run.sh` as the training stage (set  `--stage 2`). Specify a experimental name to run the following command. The tensorboard logs and checkpoints will be saved in `ckpts/tts/[YourExptName]`.

```bash
sh egs/tts/Jets/run.sh --stage 2 --name [YourExptName]
```

> **NOTE:** The `CUDA_VISIBLE_DEVICES` is set as `"0"` in default. We recommend you to only use one GPU for training.

## 4. Inference

### Configuration

For inference, you need to specify the following configurations when running `run.sh`:

| Parameters              | Description                                                        | Example                                                                                                   |
| ----------------------- | ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| `--infer_expt_dir`    | The experimental directory which contains `checkpoint`           | `ckpts/tts/[YourExptName]`                                                                              |
| `--infer_output_dir`  | The output directory to save inferred audios.                      | `ckpts/tts/[YourExptName]/result`                                                                       |
| `--infer_mode`        | The inference mode, e.g., "`batch`".                             | `batch`" to generate a batch of speech at a time.                                                       |
| `--infer_dataset`     | The dataset used for inference.                                    | For LJSpeech dataset, the inference dataset would be `LJSpeech`.                                        |
| `--infer_testing_set` | The subset of the inference dataset used for inference, e.g., test | For LJSpeech dataset, the testing set would be  "`test`" split from LJSpeech at the feature extraction |

### Run

For example, if you want to generate speech of all testing set split from LJSpeech, just run:

```bash
sh egs/tts/Jets/run.sh --stage 3 \
    --infer_expt_dir ckpts/tts/[YourExptName] \
    --infer_output_dir ckpts/tts/[YourExptName]/result \
    --infer_mode "batch" \
    --infer_dataset "LJSpeech" \
    --infer_testing_set "test"
```

### ISSUES and Solutions

```
NotImplementedError: Using RTX 3090 or 4000 series doesn't support faster communication broadband via P2P or IB. Please set `NCCL_P2P_DISABLE="1"` and `NCCL_IB_DISABLE="1" or use `accelerate launch` which will do this automatically.
2024-02-24 10:57:49 | INFO | torch.distributed.distributed_c10d | Rank 0: Completed store-based barrier for key:store_based_barrier_key:1 with 2 nodes.
```

The error message is related to an incompatibility issue with the NVIDIA RTX 3090 or 4000 series GPUs when trying to use peer-to-peer (P2P) communication or InfiniBand (IB) for faster communication. This incompatibility arises within the PyTorch accelerate library, which facilitates distributed training and inference.

To fix this issue, before running your script, you can set the environment variables in your terminal:

```
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
```

### Noted

Extensive logging messages related to `torch._subclasses.fake_tensor` and `torch._dynamo.output_graph` may be observed during inference. Despite attempts to ignore these logs, no effective solution has been found. However, it does not impact the inference process.

```bibtex
@article{lim2022jets,
  title={JETS: Jointly training FastSpeech2 and HiFi-GAN for end to end text to speech},
  author={Lim, Dan and Jung, Sunghee and Kim, Eesung},
  journal={arXiv preprint arXiv:2203.16852},
  year={2022}
}
```
