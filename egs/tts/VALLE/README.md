# VALL-E Recipe

In this recipe, we will show how to train [VALL-E](https://arxiv.org/abs/2301.02111) using Amphion's infrastructure. VALL-E is a zero-shot TTS architecture that uses a neural codec language model with discrete codes.

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
You can use the commonly used TTS dataset to train the VALL-E model, e.g., LibriTTS, etc. We strongly recommend you use LibriTTS to train the VALL-E model for the first time. How to download the dataset is detailed [here](../../datasets/README.md).

### Configuration

After downloading the dataset, you can set the dataset paths in  `exp_config.json`. Note that you can change the `dataset` list to use your preferred datasets.

```json
    "dataset": [
        "libritts",
    ],
    "dataset_path": {
        // TODO: Fill in your dataset path
        "libritts": "[LibriTTS dataset path]",
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

Run the `run.sh` as the preprocess stage (set  `--stage 1`):

```bash
sh egs/tts/VALLE/run.sh --stage 1
```

> **NOTE:** The `CUDA_VISIBLE_DEVICES` is set as `"0"` in default. You can change it when running `run.sh` by specifying such as `--gpu "1"`.


## 3. Training

### Configuration

We provide the default hyperparameters in the `exp_config.json`. They can work on a single NVIDIA-24g GPU. You can adjust them based on your GPU machines.

```json
"train": {
        "batch_size": 4,
    }
```

### Train From Scratch

Run the `run.sh` as the training stage (set  `--stage 2`). Specify an experimental name to run the following command. The tensorboard logs and checkpoints will be saved in `Amphion/ckpts/tts/[YourExptName]`.

Specifically, VALL-E needs to train an autoregressive (AR) model and then a non-autoregressive (NAR) model. So, you can set `--model_train_stage 1` to train AR model, and set `--model_train_stage 2` to train NAR model, where `--ar_model_ckpt_dir` should be set as the checkpoint path to the trained AR model.


Train an AR model, just run:

```bash
sh egs/tts/VALLE/run.sh --stage 2 --model_train_stage 1 --name [YourExptName]
```

Train a NAR model, just run:
```bash
sh egs/tts/VALLE/run.sh --stage 2 --model_train_stage 2 --ar_model_ckpt_dir [ARModelPath] --name [YourExptName]
```
<!-- > **NOTE:** To train a NAR model, `--checkpoint_path` should be set as the checkpoint path to the trained AR model. -->


### Train From Existing Source

We support training from existing sources for various purposes. You can resume training the model from a checkpoint or fine-tune a model from another checkpoint.

By setting `--resume true`, the training will resume from the **latest checkpoint** from the current `[YourExptName]` by default. For example, if you want to resume training from the latest checkpoint in `Amphion/ckpts/tts/[YourExptName]/checkpoint`, 

Train an AR model, just run:

```bash
sh egs/tts/VALLE/run.sh --stage 2 --model_train_stage 1 --name [YourExptName] \
    --resume true
```

Train a NAR model, just run:
```bash
sh egs/tts/VALLE/run.sh --stage 2 --model_train_stage 2 --ar_model_ckpt_dir [ARModelPath] --name [YourExptName] \
    --resume true
```



You can also choose a **specific checkpoint** for retraining by `--resume_from_ckpt_path` argument. For example, if you want to resume training from the checkpoint `Amphion/ckpts/tts/[YourExptName]/checkpoint/[SpecificCheckpoint]`,

Train an AR model, just run:

```bash
sh egs/tts/VALLE/run.sh --stage 2 --model_train_stage 1 --name [YourExptName] \
    --resume true \
    --resume_from_ckpt_path "Amphion/ckpts/tts/[YourExptName]/checkpoint/[SpecificARCheckpoint]"
```

Train a NAR model, just run:
```bash
sh egs/tts/VALLE/run.sh --stage 2 --model_train_stage 2 --ar_model_ckpt_dir [ARModelPath] --name [YourExptName] \
    --resume true \
    --resume_from_ckpt_path "Amphion/ckpts/tts/[YourExptName]/checkpoint/[SpecificNARCheckpoint]"
```


If you want to **fine-tune from another checkpoint**, just use `--resume_type` and set it to `"finetune"`. For example, If you want to fine-tune the model from the checkpoint `Amphion/ckpts/tts/[AnotherExperiment]/checkpoint/[SpecificCheckpoint]`, 


Train an AR model, just run:

```bash
sh egs/tts/VALLE/run.sh --stage 2 --model_train_stage 1 --name [YourExptName] \
    --resume true \
    --resume_from_ckpt_path "Amphion/ckpts/tts/[YourExptName]/checkpoint/[SpecificARCheckpoint]" \
    --resume_type "finetune"
```

Train a NAR model, just run:
```bash
sh egs/tts/VALLE/run.sh --stage 2 --model_train_stage 2 --ar_model_ckpt_dir [ARModelPath] --name [YourExptName] \
    --resume true \
    --resume_from_ckpt_path "Amphion/ckpts/tts/[YourExptName]/checkpoint/[SpecificNARCheckpoint]" \
    --resume_type "finetune"
```

> **NOTE:** The `--resume_type` is set as `"resume"` in default. It's not necessary to specify it when resuming training.
> 
> The difference between `"resume"` and `"finetune"` is that the `"finetune"` will **only** load the pretrained model weights from the checkpoint, while the `"resume"` will load all the training states (including optimizer, scheduler, etc.) from the checkpoint.




> **NOTE:** The `CUDA_VISIBLE_DEVICES` is set as `"0"` in default. You can change it when running `run.sh` by specifying such as `--gpu "0,1,2,3"`.


## 4. Inference

### Configuration

For inference, you need to specify the following configurations when running `run.sh`:



| Parameters            | Description                                                                            | Example                                                                                                                                                                         |
| --------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--infer_expt_dir`    | The experimental directory of NAR model which contains `checkpoint`                                 | `Amphion/ckpts/tts/[YourExptName]`                                                                                                                                              |
| `--infer_output_dir`  | The output directory to save inferred audios.                                          | `Amphion/ckpts/tts/[YourExptName]/result`                                                                                                                                       |
| `--infer_mode`        | The inference mode, e.g., "`single`", "`batch`".                                       | "`single`" to generate a clip of speech, "`batch`" to generate a batch of speech at a time.                                                                                     |
| `--infer_text`        | The text to be synthesized.                                                            | "`This is a clip of generated speech with the given text from a TTS model.`"                                                                                                    |
| `--infer_text_prompt`     | The text prompt for inference.                                                        | The text prompt should be aligned with the audio prompt.                                                                                                                |
| `--infer_audio_prompt` | The audio prompt for inference. | The audio prompt should be aligned with text prompt.|
| `--test_list_file` | The test list file used for batch inference. | The format of test list file is `text\|text_prompt\|audio_prompt`.|


### Run
For example, if you want to generate a single clip of speech, just run:

```bash
sh egs/tts/VALLE/run.sh --stage 3 --gpu "0" \
    --infer_expt_dir Amphion/ckpts/tts/[YourExptName] \
    --infer_output_dir Amphion/ckpts/tts/[YourExptName]/result \
    --infer_mode "single" \
    --infer_text "This is a clip of generated speech with the given text from a TTS model." \
    --infer_text_prompt "But even the unsuccessful dramatist has his moments." \
    --infer_audio_prompt egs/tts/VALLE/prompt_examples/7176_92135_000004_000000.wav
```

We have released pre-trained VALL-E models, so you can download the pre-trained model and then generate speech following the above inference instruction. Specifically, 
1. The pre-trained VALL-E trained on [LibriTTS](https://github.com/open-mmlab/Amphion/tree/main/egs/datasets#libritts) can be downloaded [here](https://huggingface.co/amphion/valle-libritts).
2. The pre-trained VALL-E trained on the part of [Libri-light](https://ai.meta.com/tools/libri-light/) (about 6k hours) can be downloaded [here](https://huggingface.co/amphion/valle_librilight_6k).

```bibtex
@article{wang2023neural,
  title={Neural codec language models are zero-shot text to speech synthesizers},
  author={Wang, Chengyi and Chen, Sanyuan and Wu, Yu and Zhang, Ziqiang and Zhou, Long and Liu, Shujie and Chen, Zhuo and Liu, Yanqing and Wang, Huaming and Li, Jinyu and others},
  journal={arXiv preprint arXiv:2301.02111},
  year={2023}
}
```