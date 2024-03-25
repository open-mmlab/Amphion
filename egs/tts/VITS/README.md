# VITS Recipe

[![hf](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Spaces-yellow)](https://huggingface.co/spaces/amphion/Text-to-Speech)
[![openxlab](https://cdn-static.openxlab.org.cn/app-center/openxlab_app.svg)](https://openxlab.org.cn/apps/detail/Amphion/Text-to-Speech)

In this recipe, we will show how to train VITS using Amphion's infrastructure. [VITS](https://arxiv.org/abs/2106.06103) is an end-to-end TTS architecture that utilizes a conditional variational autoencoder with adversarial learning.

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
You can use the commonly used TTS dataset to train the TTS model, e.g., LJSpeech, VCTK, Hi-Fi TTS, LibriTTS, etc. We strongly recommend using LJSpeech to train the single-speaker TTS model for the first time. While training the multi-speaker TTS model for the first time, we recommend using Hi-Fi TTS. The process of downloading the dataset has been detailed [here](../../datasets/README.md).

### Configuration

After downloading the dataset, you can set the dataset paths in  `exp_config.json`. Note that you can change the `dataset` list to use your preferred datasets.

```json
    "dataset": [
        "LJSpeech",
        //"hifitts"
    ],
    "dataset_path": {
        // TODO: Fill in your dataset path
        "LJSpeech": "[LJSpeech dataset path]",
        //"hifitts": "[Hi-Fi TTS dataset path]
    },
```

## 2. Features Extraction

### Configuration

In `exp_config.json`, specify the `log_dir` for saving the checkpoints and logs, and specify the `processed_dir` for saving processed data. For preprocessing the multi-speaker TTS dataset, set `extract_audio` and `use_spkid` to `true`:

```json
    // TODO: Fill in the output log path. The default value is "Amphion/ckpts/tts"
    "log_dir": "ckpts/tts",
    "preprocess": {
        //"extract_audio": true,
        "use_phone": true,
        // linguistic features
        "extract_phone": true,
        "phone_extractor": "espeak", // "espeak, pypinyin, pypinyin_initials_finals, lexicon (only for language=en-us right now)"
        // TODO: Fill in the output data path. The default value is "Amphion/data"
        "processed_dir": "data",
        "sample_rate": 22050, //target sampling rate
        "valid_file": "valid.json", //validation set
        //"use_spkid": true, //use speaker ID to train multi-speaker TTS model
    },
```

### Run

Run the `run.sh` as the preprocess stage (set  `--stage 1`):

```bash
sh egs/tts/VITS/run.sh --stage 1
```

> **NOTE:** The `CUDA_VISIBLE_DEVICES` is set as `"0"` in default. You can change it when running `run.sh` by specifying such as `--gpu "1"`.

## 3. Training

### Configuration

We provide the default hyperparameters in the `exp_config.json`. They can work on a single NVIDIA-24g GPU. You can adjust them based on your GPU machines.
For training the multi-speaker TTS model, specify the `n_speakers` value to be greater (used for new speaker fine-tuning) than or equal to the number of speakers in your dataset(s) and set `multi_speaker_training` to `true`.

```json
  "model": {
    //"n_speakers": 10 //Number of speakers in the dataset(s) used. The default value is 0 if not specified.
  },
  "train": {
    "batch_size": 16,
    //"multi_speaker_training": true, 
  }
```

### Train From Scratch

Run the `run.sh` as the training stage (set  `--stage 2`). Specify an experimental name to run the following command. The tensorboard logs and checkpoints will be saved in `Amphion/ckpts/tts/[YourExptName]`.

```bash
sh egs/tts/VITS/run.sh --stage 2 --name [YourExptName]
```

### Train From Existing Source

We support training from existing sources for various purposes. You can resume training the model from a checkpoint or fine-tune a model from another checkpoint.

By setting `--resume true`, the training will resume from the **latest checkpoint** from the current `[YourExptName]` by default. For example, if you want to resume training from the latest checkpoint in `Amphion/ckpts/tts/[YourExptName]/checkpoint`, run:

```bash
sh egs/tts/VITS/run.sh --stage 2 --name [YourExptName] \
    --resume true
```

You can also choose a **specific checkpoint** for retraining by `--resume_from_ckpt_path` argument. For example, if you want to resume training from the checkpoint `Amphion/ckpts/tts/[YourExptName]/checkpoint/[SpecificCheckpoint]`, run:

```bash
sh egs/tts/VITS/run.sh --stage 2 --name [YourExptName] \
    --resume true \
    --resume_from_ckpt_path "Amphion/ckpts/tts/[YourExptName]/checkpoint/[SpecificCheckpoint]"
```

If you want to **fine-tune from another checkpoint**, just use `--resume_type` and set it to `"finetune"`. For example, If you want to fine-tune the model from the checkpoint `Amphion/ckpts/tts/[AnotherExperiment]/checkpoint/[SpecificCheckpoint]`, run:


```bash
sh egs/tts/VITS/run.sh --stage 2 --name [YourExptName] \
    --resume true \
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

### Pre-trained Model Download

We released a pre-trained Amphion VITS model trained on LJSpeech. So you can download the pre-trained model [here](https://huggingface.co/amphion/vits-ljspeech) and generate speech according to the following inference instruction.


### Configuration

For inference, you need to specify the following configurations when running `run.sh`:


| Parameters            | Description                                                                            | Example                                                                                                                                                                         |
| --------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--infer_expt_dir`    | The experimental directory which contains `checkpoint`                                 | `Amphion/ckpts/tts/[YourExptName]`                                                                                                                                              |
| `--infer_output_dir`  | The output directory to save inferred audios.                                          | `Amphion/ckpts/tts/[YourExptName]/result`                                                                                                                                       |
| `--infer_mode`        | The inference mode, e.g., "`single`", "`batch`".                                       | "`single`" to generate a clip of speech, "`batch`" to generate a batch of speech at a time.                                                                                     |
| `--infer_dataset`     | The dataset used for inference.                                                        | For LJSpeech dataset, the inference dataset would be `LJSpeech`.<br> For Hi-Fi TTS dataset, the inference dataset would be `hifitts`.                                                                                                              |
| `--infer_testing_set` | The subset of the inference dataset used for inference, e.g., train, test, golden_test | For LJSpeech dataset, the testing set would be  "`test`" split from LJSpeech at the feature extraction, or "`golden_test`" cherry-picked from the test set as template testing set.<br>For Hi-Fi TTS dataset, the testing set would be "`test`" split from Hi-Fi TTS during the feature extraction process. |
| `--infer_text`        | The text to be synthesized.                                                            | "`This is a clip of generated speech with the given text from a TTS model.`"                                                                                                    |
| `--infer_speaker_name`        | The target speaker's voice is to be  synthesized.<br> (***Note: only applicable to multi-speaker TTS model***)                                                   | For Hi-Fi TTS dataset, the list of available speakers includes: "`hifitts_11614`", "`hifitts_11697`", "`hifitts_12787`", "`hifitts_6097`", "`hifitts_6670`", "`hifitts_6671`", "`hifitts_8051`", "`hifitts_9017`", "`hifitts_9136`", "`hifitts_92`". <br> You may find the list of available speakers from `spk2id.json` file generated in  ```log_dir/[YourExptName]``` that you have specified in `exp_config.json`.                                                                         |

### Run
#### Single text inference: 
For the single-speaker TTS model, if you want to generate a single clip of speech from a given text, just run:

```bash
sh egs/tts/VITS/run.sh --stage 3 --gpu "0" \
    --infer_expt_dir Amphion/ckpts/tts/[YourExptName] \
    --infer_output_dir Amphion/ckpts/tts/[YourExptName]/result \
    --infer_mode "single" \
    --infer_text "This is a clip of generated speech with the given text from a TTS model."
```

For the multi-speaker TTS model, in addition to the above-mentioned arguments, you need to add ```infer_speaker_name``` argument, and run: 
```bash
sh egs/tts/VITS/run.sh --stage 3 --gpu "0" \
    --infer_expt_dir Amphion/ckpts/tts/[YourExptName] \
    --infer_output_dir Amphion/ckpts/tts/[YourExptName]/result \
    --infer_mode "single" \
    --infer_text "This is a clip of generated speech with the given text from a TTS model." \
    --infer_speaker_name "hifitts_92"
```

#### Batch inference: 
For the single-speaker TTS model, if you want to generate speech of all testing sets split from LJSpeech, just run:

```bash
sh egs/tts/VITS/run.sh --stage 3 --gpu "0" \
    --infer_expt_dir Amphion/ckpts/tts/[YourExptName] \
    --infer_output_dir Amphion/ckpts/tts/[YourExptName]/result \
    --infer_mode "batch" \
    --infer_dataset "LJSpeech" \
    --infer_testing_set "test"
```
For the multi-speaker TTS model, if you want to generate speech of all testing sets split from Hi-Fi TTS, the same procedure follows from above, with ```LJSpeech``` replaced by ```hifitts```.
```bash
sh egs/tts/VITS/run.sh --stage 3 --gpu "0" \
    --infer_expt_dir Amphion/ckpts/tts/[YourExptName] \
    --infer_output_dir Amphion/ckpts/tts/[YourExptName]/result \
    --infer_mode "batch" \
    --infer_dataset "hifitts" \
    --infer_testing_set "test" 
```


We released a pre-trained Amphion VITS model trained on LJSpeech. So, you can download the pre-trained model [here](https://huggingface.co/amphion/vits-ljspeech) and generate speech following the above inference instructions. Meanwhile, the pre-trained multi-speaker VITS model trained on Hi-Fi TTS will be released soon. Stay tuned.


```bibtex
@inproceedings{kim2021conditional,
  title={Conditional variational autoencoder with adversarial learning for end-to-end text-to-speech},
  author={Kim, Jaehyeon and Kong, Jungil and Son, Juhee},
  booktitle={International Conference on Machine Learning},
  pages={5530--5540},
  year={2021},
}
```
