# VITS for Voice Conversion

This is an implementation of VITS as acoustic model for end-to-end voice conversion. Adapted from [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc), SoftVC content encoder is used to extract content features from the source audio. These feature vectors are directly fed into VITS without the need for conversion to a text-based intermediate representation.

There are four stages in total:

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

By default, we utilize the LibriTTS datasets for training. How to download them is detailed [here](../../datasets/README.md).

### Configuration

Specify the dataset paths in  `exp_config.json`. Note that you can change the `dataset` list to use your preferred datasets.

```json
    "dataset": [
        "libritts"
    ],
    "dataset_path": {
        // TODO: Fill in your dataset path
        "libritts": "[LibriTTS dataset path]"
    },
```

## 2. Features Extraction

### Content-based Pretrained Models Download

By default, we utilize Hubert to extract content features. How to download them is detailed [here](../../../pretrained/README.md).

### Configuration

Specify the dataset path and the output path for saving the processed data and the training model in `exp_config.json`:

```json
    // TODO: Fill in the output log path. The default value is "Amphion/ckpts/svc"
    "log_dir": "ckpts/vc",
    "preprocess": {
        // TODO: Fill in the output data path. The default value is "Amphion/data"
        "processed_dir": "data",
        ...
    },
```

### Run

Run the `run.sh` as the preproces stage (set  `--stage 1`).

```bash
sh egs/vc/VitsVC/run.sh --stage 1
```

> **NOTE:** The `CUDA_VISIBLE_DEVICES` is set as `"0"` in default. You can change it when running `run.sh` by specifying such as `--gpu "1"`.

## 3. Training

### Configuration

We provide the default hyparameters in the `exp_config.json`. They can work on single NVIDIA-24g GPU. You can adjust them based on you GPU machines.

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
sh egs/vc/VitsVC/run.sh --stage 2 --name [YourExptName]
```

> **NOTE:** The `CUDA_VISIBLE_DEVICES` is set as `"0"` in default. You can change it when running `run.sh` by specifying such as `--gpu "0,1,2,3"`.

## 4. Inference/Conversion

### Run

For inference/conversion, you need to specify the following configurations when running `run.sh`:

| Parameters                                          | Description                                                   | Example                                                                                                                                                                                                  |
| --------------------------------------------------- |---------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--infer_expt_dir`                                  | The experimental directory which contains `checkpoint`        | `[Your path to save logs and checkpoints]/[YourExptName]`                                                                                                                                                |
| `--infer_output_dir`                                | The output directory to save inferred audios.                 | `[Your path to save logs and checkpoints]/[YourExptName]/result`                                                                                                                                         |
| `--infer_source_file` or `--infer_source_audio_dir` | The inference source (can be a json file or a dir).           | The `infer_source_file` could be `[Your path to save processed data]/[YourDataset]/test.json`, and the `infer_source_audio_dir` is a folder which includes several audio files (*.wav, *.mp3 or *.flac). |
| `--infer_target_speaker`                            | The audio file of the target speaker you want to convert into.| `[Your path to the target audio file]`                                                                                                                                                                   |

For example, if you want to make the speaker in `reference.wav` to speake the utterances in the `[Your Audios Folder]`, just run:

```bash
sh egs/vc/VitsVC/run.sh --stage 3 --gpu "0" \
	--infer_expt_dir Amphion/ckpts/vc/[YourExptName] \
	--infer_output_dir Amphion/ckpts/vc/[YourExptName]/result \
	--infer_source_audio_dir [Your Audios Folder] \
	--infer_target_speaker "reference.wav" 
```