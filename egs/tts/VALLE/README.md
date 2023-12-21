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
You can use the commonly used TTS dataset to train VALL-E model, e.g., LibriTTS, etc. We strongly recommend you use LibriTTS to train VALL-E model for the first time. How to download dataset is detailed [here](../../datasets/README.md).

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

Run the `run.sh` as the preproces stage (set  `--stage 1`):

```bash
sh egs/tts/VALLE/run.sh --stage 1
```

> **NOTE:** The `CUDA_VISIBLE_DEVICES` is set as `"0"` in default. You can change it when running `run.sh` by specifying such as `--gpu "1"`.


## 3. Training

### Configuration

We provide the default hyparameters in the `exp_config.json`. They can work on single NVIDIA-24g GPU. You can adjust them based on your GPU machines.

```
"train": {
        "batch_size": 4,
    }
```

### Run

Run the `run.sh` as the training stage (set  `--stage 2`). Specify a experimental name to run the following command. The tensorboard logs and checkpoints will be saved in `Amphion/ckpts/tts/[YourExptName]`.

Specifically, VALL-E need to train a autoregressive (AR) model and then a non-autoregressive (NAR) model. So, you can set `--model_train_stage 1` to train AR model, and set `--model_train_stage 2` to train NAR model, where `--ar_model_ckpt_dir` should be set as the ckeckpoint path to the trained AR model.


Train a AR moel, just run:

```bash
sh egs/tts/VALLE/run.sh --stage 2 --model_train_stage 1 --name [YourExptName]
```

Train a NAR model, just run:
```bash
sh egs/tts/VALLE/run.sh --stage 2 --model_train_stage 2 --ar_model_ckpt_dir [ARModelPath] --name [YourExptName]
```
<!-- > **NOTE:** To train a NAR model, `--checkpoint_path` should be set as the ckeckpoint path to the trained AR model. -->

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


We released a pre-trained Amphion VALL-E model. So you can download the pre-trained model [here](https://huggingface.co/amphion/valle-libritts) and generate speech following the above inference instruction.

```bibtex
@article{wang2023neural,
  title={Neural codec language models are zero-shot text to speech synthesizers},
  author={Wang, Chengyi and Chen, Sanyuan and Wu, Yu and Zhang, Ziqiang and Zhou, Long and Liu, Shujie and Chen, Zhuo and Liu, Yanqing and Wang, Huaming and Li, Jinyu and others},
  journal={arXiv preprint arXiv:2301.02111},
  year={2023}
}
```