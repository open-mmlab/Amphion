# Pretrained Models Dependency

The models dependency of Amphion are as follows (sort alphabetically):

- [Pretrained Models Dependency](#pretrained-models-dependency)
  - [Amphion Singing BigVGAN](#amphion-singing-bigvgan)
  - [Amphion Speech HiFi-GAN](#amphion-speech-hifi-gan)
  - [ContentVec](#contentvec)
  - [WeNet](#wenet)
  - [Whisper](#whisper)
  - [RawNet3](#rawnet3)


The instructions about how to download them is displayed as follows.

## Amphion Singing BigVGAN

We fine-tune the official BigVGAN pretrained model with over 120 hours singing voice data. The fine-tuned checkpoint can be downloaded [here](https://cuhko365-my.sharepoint.com/:f:/g/personal/222042021_link_cuhk_edu_cn/EtiHh5JZ0_xGlYbyLLSoqBgBe9kI5q3ROY-SvBqefae-IA?e=dk4Pqa). You need to download the `400000.pt` and `args.json` files into `Amphion/pretrained/bigvgan`:

```
Amphion
 ┣ pretrained
 ┃ ┣ bivgan
 ┃ ┃ ┣ 400000.pt
 ┃ ┃ ┣ args.json
```

## Amphion Speech HiFi-GAN

We trained our HiFi-GAN pretrained model with 685 hours speech data. Which can be downloaded [here](https://cuhko365-my.sharepoint.com/:f:/g/personal/xueliumeng_cuhk_edu_cn/Ei24hGJO_PVBopjhKje1uzEBqfhV9h89HoLrOoy9K8tzGg?e=ka7MCO). You need to download the whole folder of `hifigan_speech` into `Amphion/pretrained/hifigan`.

```
Amphion
 ┣ pretrained
 ┃ ┣ hifigan
 ┃ ┃ ┣ hifigan_speech
 ┃ ┃ ┃ ┣ log
 ┃ ┃ ┃ ┣ result
 ┃ ┃ ┃ ┣ checkpoint
 ┃ ┃ ┃ ┣ args.json
```

## ContentVec

You can download the pretrained ContentVec model [here](https://github.com/auspicious3000/contentvec). Note that we use the `ContentVec_legacy-500 classes` checkpoint. Assume that you download the `checkpoint_best_legacy_500.pt` into the `Amphion/pretrained/contentvec`.

```
Amphion
 ┣ pretrained
 ┃ ┣ contentvec
 ┃ ┃ ┣ checkpoint_best_legacy_500.pt
```

## WeNet

You can download the pretrained WeNet model [here](https://github.com/wenet-e2e/wenet/blob/main/docs/pretrained_models.md). Take the `wenetspeech` pretrained checkpoint as an example, assume you download the `wenetspeech_u2pp_conformer_exp.tar` into the `Amphion/pretrained/wenet`. Unzip it and modify its configuration file as follows:

```sh
cd Amphion/pretrained/wenet

### Unzip the expt dir
tar -xvf wenetspeech_u2pp_conformer_exp.tar.gz

### Specify the updated path in train.yaml
cd 20220506_u2pp_conformer_exp
vim train.yaml
# TODO: Change the value of "cmvn_file" (Line 2) to the absolute path of the `global_cmvn` file. (Eg: [YourPath]/Amphion/pretrained/wenet/20220506_u2pp_conformer_exp/global_cmvn)
```

The final file struture tree is like:

```
Amphion
 ┣ pretrained
 ┃ ┣ wenet
 ┃ ┃ ┣ 20220506_u2pp_conformer_exp
 ┃ ┃ ┃ ┣ final.pt
 ┃ ┃ ┃ ┣ global_cmvn
 ┃ ┃ ┃ ┣ train.yaml
 ┃ ┃ ┃ ┣ units.txt
```

## Whisper

The official pretrained whisper checkpoints can be available [here](https://github.com/openai/whisper/blob/e58f28804528831904c3b6f2c0e473f346223433/whisper/__init__.py#L17). In Amphion, we use the `medium` whisper model by default. You can download it as follows:

```bash
cd Amphion/pretrained
mkdir whisper
cd whisper

wget https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt
```

The final file structure tree is like:

```
Amphion
 ┣ pretrained
 ┃ ┣ whisper
 ┃ ┃ ┣ medium.pt
```

## RawNet3

The official pretrained RawNet3 checkpoints can be available [here](https://huggingface.co/jungjee/RawNet3). You need to download the `model.pt` file and put it in the folder.

The final file structure tree is like:

```
Amphion
 ┣ pretrained
 ┃ ┣ rawnet3
 ┃ ┃ ┣ model.pt
```

