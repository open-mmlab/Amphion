# VITS Recipe

[![hf](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Spaces-yellow)](https://huggingface.co/spaces/amphion/Text-to-Speech)
[![openxlab](https://cdn-static.openxlab.org.cn/app-center/openxlab_app.svg)](https://openxlab.org.cn/apps/detail/Amphion/Text-to-Speech)

In this recipe, we will show how to train [VITS](https://arxiv.org/abs/2106.06103) using Amphion's infrastructure. VITS is an end-to-end TTS architecture that utilizes conditional variational autoencoder with adversarial learning.

There are four stages in total:

1. Data preparation
2. Features extraction
3. Training
4. Inference

> **NOTE:** You need to run every command of this recipe in the `Amphion` root path:
> ```bash
> cd Amphion
> ```

## 1. Data Preparation

### Dataset Download
You can use the commonly used TTS dataset to train TTS model, e.g., LJSpeech, VCTK, LibriTTS, etc. We strongly recommend you use LJSpeech to train TTS model for the first time. How to download dataset is detailed [here](../../datasets/README.md).

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
    // TODO: Fill in the output log path. The default value is "Amphion/ckpts/tts"
    "log_dir": "ckpts/tts",
    "preprocess": {
        // TODO: Fill in the output data path. The default value is "Amphion/data"
        "processed_dir": "data",
        ...
    },
```

### Run

Run the `run.sh` as the preproces stage (set  `--stage 1`):

```bash
sh egs/tts/VITS/run.sh --stage 1
```

> **NOTE:** The `CUDA_VISIBLE_DEVICES` is set as `"0"` in default. You can change it when running `run.sh` by specifying such as `--gpu "1"`.

## 3. Training

### Configuration

We provide the default hyparameters in the `exp_config.json`. They can work on single NVIDIA-24g GPU. You can adjust them based on your GPU machines.

```
"train": {
        "batch_size": 16,
    }
```

### Train From Scratch

Run the `run.sh` as the training stage (set  `--stage 2`). Specify a experimental name to run the following command. The tensorboard logs and checkpoints will be saved in `Amphion/ckpts/tts/[YourExptName]`.

```bash
sh egs/tts/VITS/run.sh --stage 2 --name [YourExptName]
```

### Train From Existing Source

We support training from existing source for various purposes. You can resume training the model from a checkpoint or fine-tune a model from another checkpoint.

Setting `--resume true`, the training will resume from the **latest checkpoint** from the current `[YourExptName]` by default. For example, if you want to resume training from the latest checkpoint in `Amphion/ckpts/tts/[YourExptName]/checkpoint`, run:

```bash
sh egs/tts/VITS/run.sh --stage 2 --name [YourExptName] \
    --resume true
```

You can also choose a **specific checkpoint** for retraining by `--resume_from_ckpt_path` argument. For example, if you want to resume training from the checkpoint `Amphion/ckpts/tts/[YourExptName]/checkpoint/[SpecificCheckpoint]`, run:

```bash
sh egs/tts/VITS/run.sh --stage 2 --name [YourExptName] \
    --resume true
    --resume_from_ckpt_path "Amphion/ckpts/tts/[YourExptName]/checkpoint/[SpecificCheckpoint]" \
```

If you want to **fine-tune from another checkpoint**, just use `--resume_type` and set it to `"finetune"`. For example, If you want to fine-tune the model from the checkpoint `Amphion/ckpts/tts/[AnotherExperiment]/checkpoint/[SpecificCheckpoint]`, run:


```bash
sh egs/tts/VITS/run.sh --stage 2 --name [YourExptName] \
    --resume true
    --resume_from_ckpt_path "Amphion/ckpts/tts/[YourExptName]/checkpoint/[SpecificCheckpoint]" \
    --resume_type "finetune"
```

> **NOTE:** The `--resume_type` is set as `"resume"` in default. It's not necessary to specify it when resuming training.
> 
> The difference between `"resume"` and `"finetune"` is that the `"finetune"` will **only** load the pretrained model weights from the checkpoint, while the `"resume"` will load all the training states (including optimizer, scheduler, etc.) from the checkpoint.

Here are some example scenarios to better understand how to use these arguments:
| Scenario | `--resume` | `--resume_from_ckpt_path` | `--resume_type` |
| ------ | -------- | ----------------------- | ------------- |
| You want to train from scratch | no | no | no |
| The machine breaks down during training and you want to resume training from the latest checkpoint | `true` | no | no |
| You find the latest model is overfitting and you want to re-train from the checkpoint before | `true` | `SpecificCheckpoint Path` | no |
| You want to fine-tune a model from another checkpoint | `true` | `SpecificCheckpoint Path` | `"finetune"` |


> **NOTE:** The `CUDA_VISIBLE_DEVICES` is set as `"0"` in default. You can change it when running `run.sh` by specifying such as `--gpu "0,1,2,3"`.


## 4. Inference

### Configuration

For inference, you need to specify the following configurations when running `run.sh`:


| Parameters            | Description                                                                            | Example                                                                                                                                                                         |
| --------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--infer_expt_dir`    | The experimental directory which contains `checkpoint`                                 | `Amphion/ckpts/tts/[YourExptName]`                                                                                                                                              |
| `--infer_output_dir`  | The output directory to save inferred audios.                                          | `Amphion/ckpts/tts/[YourExptName]/result`                                                                                                                                       |
| `--infer_mode`        | The inference mode, e.g., "`single`", "`batch`".                                       | "`single`" to generate a clip of speech, "`batch`" to generate a batch of speech at a time.                                                                                     |
| `--infer_dataset`     | The dataset used for inference.                                                        | For LJSpeech dataset, the inference dataset would be `LJSpeech`.                                                                                                                |
| `--infer_testing_set` | The subset of the inference dataset used for inference, e.g., train, test, golden_test | For LJSpeech dataset, the testing set would be  "`test`" split from LJSpeech at the feature extraction, or "`golden_test`" cherry-picked from test set as template testing set. |
| `--infer_text`        | The text to be synthesized.                                                            | "`This is a clip of generated speech with the given text from a TTS model.`"                                                                                                    |

### Run
For example, if you want to generate speech of all testing set split from LJSpeech, just run:

```bash
sh egs/tts/VITS/run.sh --stage 3 --gpu "0" \
    --infer_expt_dir Amphion/ckpts/tts/[YourExptName] \
    --infer_output_dir Amphion/ckpts/tts/[YourExptName]/result \
    --infer_mode "batch" \
    --infer_dataset "LJSpeech" \
    --infer_testing_set "test"
```

Or, if you want to generate a single clip of speech from a given text, just run:

```bash
sh egs/tts/VITS/run.sh --stage 3 --gpu "0" \
    --infer_expt_dir Amphion/ckpts/tts/[YourExptName] \
    --infer_output_dir Amphion/ckpts/tts/[YourExptName]/result \
    --infer_mode "single" \
    --infer_text "This is a clip of generated speech with the given text from a TTS model."
```

We released a pre-trained Amphion VITS model trained on LJSpeech. So you can download the pre-trained model [here](https://huggingface.co/amphion/vits-ljspeech) and generate speech following the above inference instruction.


```bibtex
@inproceedings{kim2021conditional,
  title={Conditional variational autoencoder with adversarial learning for end-to-end text-to-speech},
  author={Kim, Jaehyeon and Kong, Jungil and Son, Juhee},
  booktitle={International Conference on Machine Learning},
  pages={5530--5540},
  year={2021},
}
```