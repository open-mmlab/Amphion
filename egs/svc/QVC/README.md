# Quick (Singing) Voice Conversion

This is an implementation of a simple Webui which provides a simple and quick text-free one-shot voice conversion for the uninitiated. Thereotically, the user only takes two short audios (source and target) and a few minutes to receive the VC result. 
It aims to use the base model (checkpoint) trained from the VCTK, M4Singer datasets (or other supported datasets) as a foundation, and then fine-tune the base model using the input source audio for voice conversion and output. Now it supports MultipleContentSVC and VITS. 

Like other SVC tasks, There are four stages in total:

1. Data preparation
2. Features extraction
3. Training
4. Inference/conversion

> **NOTE:** You need to run every command of this recipe in the `Amphion` root path:
> ```bash
> cd Amphion
> ```

## 1. Data Preparation

### Dataset Download

By default, we utilize the five datasets for training: M4Singer, Opencpop, OpenSinger, SVCC, and VCTK. How to download them is detailed [here](../../datasets/README.md).

### Configuration

Specify the dataset paths in  `exp_config_[model_type].json`. Note that you can change the `dataset` list to use your preferred datasets.

```json
    "dataset": [
        "m4singer",
        "opencpop",
        "opensinger",
        "svcc",
        "vctk"
    ],
    "dataset_path": {
        // TODO: Fill in your dataset path
        "m4singer": "[M4Singer dataset path]",
        "opencpop": "[Opencpop dataset path]",
        "opensinger": "[OpenSinger dataset path]",
        "svcc": "[SVCC dataset path]",
        "vctk": "[VCTK dataset path]"
    },
```

## 2. Features Extraction

### Content-based Pretrained Models Download

By default, we utilize ContentVec and Whisper to extract content features. How to download them is detailed [here](../../../pretrained/README.md).

### Configuration

Specify the dataset path and the output path for saving the processed data and the training model in `exp_config_[model_type].json`:

```json
    // TODO: Fill in the output log path. The default value is "Amphion/ckpts/svc"
    "log_dir": "ckpts/svc",
    "preprocess": {
        // TODO: Fill in the output data path. The default value is "Amphion/data"
        "processed_dir": "data",
        ...
    },
```

### Run

Run the `run.sh` as the preproces stage (set  `--stage 1`). Config_type 1 means MultipleContentSVC (DiffWaveNet); 2 means VITS. 

```bash
sh egs/svc/QVC/run.sh --stage 1 --config_type (1 or 2)
```

> **NOTE:** The `CUDA_VISIBLE_DEVICES` is set as `"0"` in default. You can change it when running `run.sh` by specifying such as `--gpu "1"`.

## 3. Training

### Configuration

We provide the default hyparameters in the `exp_config_[model_type].json`. They can work on single NVIDIA-24g GPU. You can adjust them based on you GPU machines.

```json
"train": {
        "batch_size": 32,
        ...
        "adamw": {
            "lr": 2.0e-4
        },
        ...
    }
```

### Run

Run the `run.sh` as the training stage (set  `--stage 2`). Specify a experimental name to run the following command. The tensorboard logs and checkpoints will be saved in `Amphion/ckpts/svc/[YourExptName]`.

```bash
sh egs/svc/QVC/run.sh --stage 2 --name [YourExptName] --config_type (1 or 2)
```

## 4. Inference/Conversion

### Run

`inf_config_[model_type].json` is a file similar to `exp_config_[model_type].json` storing training parameters, but you need to be careful when modifying configs here (especially the temp storing path here). 
For inference/conversion, you need to specify the following configurations when running `run.sh`:

| Parameters                                          | Description                                                                                                                                                       | Example                                                                                                                                                                                                  |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--infer_output_dir`                                | The output directory to save inferred audios.                                                                                                                     | `[Your path to save logs and checkpoints]/[YourExptName]/result`                
| `--resume_from_ckpt_path`                                | The checkpoint path to load model parameters.                                                                                                                     | `[Your path to save logs and checkpoints]` |
| `--infer_source_file`                                `   | The `infer_source_file` should be a file with *.wav, *.mp3 or *.flac. |
| `--infer_target_speaker`                            | The target speaker you want to convert into. You can refer to `[Your path to save logs and checkpoints]/[YourExptName]/singers.json` to choose a trained speaker. | For opencpop dataset, the speaker name would be `opencpop_female1`.                                                                                                                                      |
| `--infer_key_shift`                                 | How many semitones you want to transpose.                                                                                                                         | `"autoshfit"` (by default), `3`, `-3`, etc.                                                                                                                                                              |

Note that type of checkpoint model weights must match the model config type, now run:

```bash
sh egs/svc/QVC/run.sh --stage 3 --gpu "0" --config_type (1 or 2) \
    --resume_from_ckpt_path [Your checkpoint Path] \
	--infer_output_dir Amphion/ckpts/svc/[YourExptName]/result \
    --infer_source_file [Your Audio Path] \
	--target_source_dir [Your Audios Folder] \
	--infer_key_shift "autoshift"
```

Before opening the Webui, you need to install:
```
pip install gradio==3.42.0
```

Then you can initilize the Webui by running: 

```bash
sh egs/svc/QVC/run.sh --stage 4 --gpu "0" --config_type (1 or 2) \
    --resume_from_ckpt_path [Your checkpoint Path] \
	--infer_output_dir Amphion/ckpts/svc/[YourExptName]/result \
    --infer_source_file [Your Audio Path] \
	--target_source_dir [Your Audios Folder] \
	--infer_key_shift "autoshift"

```