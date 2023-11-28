
# FastSpeech2 Recipe

In this recipe, we will show how to train [FastSpeech2](https://openreview.net/forum?id=piLPYqxtWuA) using Amphion's infrastructure. FastSpeech2 is a non-autoregressive TTS architecture that utilizes feed-forward Transformer blocks.

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
sh egs/tts/FastSpeech2/run.sh --stage 1
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
sh egs/tts/FastSpeech2/run.sh --stage 2 --name [YourExptName]
```

> **NOTE:** The `CUDA_VISIBLE_DEVICES` is set as `"0"` in default. You can change it when running `run.sh` by specifying such as `--gpu "0,1,2,3"`.


## 4. Inference

### Configuration

For inference, you need to specify the following configurations when running `run.sh`:


| Parameters                                          | Description                                                                                                                                                       | Example                                                                                                                                                                                                |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--infer_expt_dir`                              | The experimental directory which contains `checkpoint`                                                                                                            | `ckpts/tts/[YourExptName]`                                                                                                                                              |
| `--infer_output_dir`                                | The output directory to save inferred audios.                                                                                                                     | `ckpts/tts/[YourExptName]/result`                                                                                                                                       |
| `--infer_mode`                            | The inference mode, e.g., "`single`", "`batch`".  | "`single`" to generate a clip of speech, "`batch`" to generate a batch of speech at a time.                                     |
| `--infer_dataset`                            | The dataset used for inference.  |  For LJSpeech dataset, the inference dataset would be `LJSpeech`.                                                                                                                                    |
| `--infer_testing_set`                             | The subset of the inference dataset used for inference, e.g., train, test, golden_test | For LJSpeech dataset, the testing set would be  "`test`" split from LJSpeech at the feature extraction, or "`golden_test`" cherry-picked from test set as template testing set.                                                                                                                                    |
| `--infer_text`                            | The text to be synthesized. | "`This is a clip of generated speech with the given text from a TTS model.`"                                                                                                                                    |

### Run
For example, if you want to generate speech of all testing set split from LJSpeech, just run:

```bash
sh egs/tts/FastSpeech2/run.sh --stage 3 \
    --infer_expt_dir ckpts/tts/[YourExptName] \
    --infer_output_dir ckpts/tts/[YourExptName]/result \
    --infer_mode "batch" \
    --infer_dataset "LJSpeech" \
    --infer_testing_set "test"
```

Or, if you want to generate a single clip of speech from a given text, just run:

```bash
sh egs/tts/FastSpeech2/run.sh --stage 3 \
    --infer_expt_dir ckpts/tts/[YourExptName] \
    --infer_output_dir ckpts/tts/[YourExptName]/result \
    --infer_mode "single" \
    --infer_text "This is a clip of generated speech with the given text from a TTS model."
```

We will release a pre-trained FastSpeech2 model trained on LJSpeech. So you can download the pre-trained model and generate speech following the above inference instruction.


```bibtex
@inproceedings{ren2020fastspeech,
  title={FastSpeech 2: Fast and High-Quality End-to-End Text to Speech},
  author={Ren, Yi and Hu, Chenxu and Tan, Xu and Qin, Tao and Zhao, Sheng and Zhao, Zhou and Liu, Tie-Yan},
  booktitle={International Conference on Learning Representations},
  year={2020}
}
```
