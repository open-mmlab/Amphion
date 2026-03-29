# FreeVC

This is an implementation of [FreeVC: Towards High-Quality Text-Free One-Shot Voice Conversion](https://arxiv.org/abs/2210.15418). Adapted from end-to-end framework of [VITS](https://arxiv.org/abs/2106.06103) for high-quality waveform reconstruction, and propose strategies for clean content information extraction without text annotation. It disentangle content information by imposing an information bottleneck to [WavLM](https://arxiv.org/abs/2110.13900) features, utilize the **spectrogram-resize** based data augmentation to improve the purity of extracted content information.

There are four stages in total:

1. Data preparation
2. Features extraction
3. Training
4. Inference/conversion

> **NOTE:** You need to run every command of this recipe in the `Amphion` root path:
>
> ```bash
> cd Amphion
> ```

## 1. Data Preparation

### Dataset Download

For other experiments, we utilize the five datasets for training: M4Singer, Opencpop, OpenSinger, SVCC, and VCTK. How to download them is detailed [here](../../datasets/README.md).

In this experiment, we only utilize two datasets: VTCK and LibriTTS

### Configuration

Specify the dataset path in  `exp_config.json`.

```json
    "preprocess": {
        "vctk_dir": "[VCTK dataset path]",
        // ...
    }
```

## 2. Features Extraction

### Pretrained Models Download

You should download pretrained HiFi-GAN (VCTK_V1) from [its repo](https://github.com/jik876/hifi-gan) according to the original paper.

The code will automatically download pretrained [WavLM-Large](https://huggingface.co/microsoft/wavlm-large) model from Huggingface. You can also download it in advance:

```bash
huggingface-cli download microsoft/wavlm-large
```

The pretrained speaker encoder is available at: <https://github.com/liusongxiang/ppg-vc/tree/main/speaker_encoder/ckpt>

The weight should be put in `models/vc/FreeVC/speaker_encoder/ckpt/` since it is excluded from the git history.

### Configuration

Specify the data path and the checkpoint path for saving the processed data in `exp_config.json`:

```json
    "preprocess": {
        // ...
        "vctk_16k_dir": "[preprocessed VCTK 16k directory]",
        "vctk_22k_dir": "[preprocessed VCTK 22k directory]",
        "spk_dir": "[preprocess_spk directory]",
        "ssl_dir": "[preprocess_ssl directory]",
        "sr_dir": "[preprocess_sr directory]", 
        "hifigan_ckpt_path": "[hifigan checkpoint file path]"
        // ...
    },
```

Note that the preprocessed data will take about 600GB disk space.

### Run

Run the `run.sh` as the preproces stage (set  `--stage 1`).

```bash
sh egs/vc/FreeVC/run.sh --stage 1 -c egs/vc/FreeVC/exp_config.json
```

> **NOTE:** The `CUDA_VISIBLE_DEVICES` is set as `"0"` in default. You can change it when running `run.sh` by specifying such as `--gpu "1"`.

## 3. Training

### Configuration

We provide the default hyparameters in the `config/freevc.json`. They can work on single NVIDIA-24g GPU. You can adjust them based on you GPU machines.

```json
"model": {
    "use_spk": true
    // ...
},
"train": {
    "use_sr": true,
    // ...
    "batch_size": 32,
    // ...
    "learning_rate": 2.0e-4
    // ...
}
```

### Run

Run the `run.sh` as the training stage (set  `--stage 2`). Specify a experimental name to run the following command. The tensorboard logs and checkpoints will be saved in `Amphion/ckpts/vc/FreeVC/[YourExptName]`.

```bash
sh egs/vc/FreeVC/run.sh --stage 2 -c egs/vc/FreeVC/exp_config.json --name [YourExptName]
```

> **NOTE:** The `CUDA_VISIBLE_DEVICES` is set as `"0"` in default. You can change it when running `run.sh` by specifying such as `--gpu "0,1,2,3"`.

## 4. Inference/Conversion

### Run

For inference/conversion, you need to first create a file `convert.txt` indicating the source audio, target audio and the name of the output audio in following format:

```
# format
[name of the output]|[path/to/the/source/audio]|[path/to/the/target/audio]

# an example(each reconstruction written in a line)
title1|data/vctk-16k/p225/p225_001.wav|data/vctk-16k/p226/p226_002.wav
```


Then you should run `run.sh`,  you need to specify the following configurations:

| Parameters        | Description                                                  | Example                                                      |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `--config`        | The base configuration                                       | `[Your path to the base configuration]`                      |
| `--ckpt`          | The experimental directory which contains `checkpoint`       | `[Your path to save logs and checkpoints]/[YourExptName]`    |
| `--convert`       | The convert.txt path which contains the audios to be reconstructed | `[Your path to save convert.txt]`                            |
| `--outdir`        | The output directory to save inferred audios.                | `[Your path to save logs and checkpoints]/[YourExptName]/result` |

For example:

```bash
sh egs/vc/FreeVC/run.sh --stage 3 \
	--config egs/vc/FreeVC/exp_config.json \
	--ckpt ckpts/vc/FreeVC/[YourExptName]/G_100000.ckpt \
	--convert ckpts/vc/FreeVC/[YourExptName] \
	--outdir ckpts/vc/FreeVC/[YourExptName]/result \
```
