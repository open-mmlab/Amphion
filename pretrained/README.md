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

We fine-tune the official BigVGAN pretrained model with over 120 hours singing voice data. The fine-tuned checkpoint can be downloaded [here](https://huggingface.co/amphion/BigVGAN_singing_bigdata). You need to download the `400000.pt` and `args.json` files into `Amphion/pretrained/bigvgan`:

```
Amphion
 ┣ pretrained
 ┃ ┣ bivgan
 ┃ ┃ ┣ 400000.pt
 ┃ ┃ ┣ args.json
```

## Amphion Speech HiFi-GAN

We trained our HiFi-GAN pretrained model with 685 hours speech data. Which can be downloaded [here](https://huggingface.co/amphion/hifigan_speech_bigdata). You need to download the whole folder of `hifigan_speech` into `Amphion/pretrained/hifigan`.

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

## Amphion DiffWave

We trained our DiffWave pretrained model with 125 hours speech data and around 80 hours of singing voice data. Which can be downloaded [here](https://huggingface.co/amphion/diffwave). You need to download the whole folder of `diffwave` into `Amphion/pretrained/diffwave`.

```
Amphion
 ┣ pretrained
 ┃ ┣ diffwave
 ┃ ┃ ┣ diffwave_speech
 ┃ ┃ ┃ ┣ samples
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


# (Optional) Model Dependencies for Evaluation
When utilizing Amphion's Evaluation Pipelines, terminals without access to `huggingface.co` may encounter error messages such as "OSError: Can't load tokenizer for ...". To work around this, the dependant models for evaluation can be pre-prepared and stored here, at `Amphion/pretrained`, and follow [this README](../egs/metrics/README.md#troubleshooting) to configure your environment to load local models.

The dependant models of Amphion's evaluation pipeline are as follows (sort alphabetically):

- [Evaluation Pipeline Models Dependency](#optional-model-dependencies-for-evaluation)
  - [bert-base-uncased](#bert-base-uncased)
  - [facebook/bart-base](#facebookbart-base)
  - [roberta-base](#roberta-base)
  - [wavlm](#wavlm)

The instructions about how to download them is displayed as follows.

## bert-base-uncased

To load `bert-base-uncased` locally, follow [this link](https://huggingface.co/bert-base-uncased) to download all files for `bert-base-uncased` model, and store them under `Amphion/pretrained/bert-base-uncased`, conforming to the following file structure tree:

```
Amphion
 ┣ pretrained
 ┃ ┣ bert-base-uncased
 ┃ ┃ ┣ config.json
 ┃ ┃ ┣ coreml 
 ┃ ┃ ┃ ┣ fill-mask
 ┃ ┃ ┃   ┣ float32_model.mlpackage
 ┃ ┃ ┃      ┣ Data
 ┃ ┃ ┃         ┣ com.apple.CoreML
 ┃ ┃ ┃            ┣ model.mlmodel 
 ┃ ┃ ┣ flax_model.msgpack
 ┃ ┃ ┣ LICENSE
 ┃ ┃ ┣ model.onnx
 ┃ ┃ ┣ model.safetensors
 ┃ ┃ ┣ pytorch_model.bin
 ┃ ┃ ┣ README.md
 ┃ ┃ ┣ rust_model.ot
 ┃ ┃ ┣ tf_model.h5
 ┃ ┃ ┣ tokenizer_config.json
 ┃ ┃ ┣ tokenizer.json
 ┃ ┃ ┣ vocab.txt
```

## facebook/bart-base

To load `facebook/bart-base` locally, follow [this link](https://huggingface.co/facebook/bart-base) to download all files for `facebook/bart-base` model, and store them under `Amphion/pretrained/facebook/bart-base`, conforming to the following file structure tree:

```
Amphion
 ┣ pretrained
 ┃ ┣ facebook
 ┃ ┃ ┣ bart-base
 ┃ ┃ ┃ ┣ config.json
 ┃ ┃ ┃ ┣ flax_model.msgpack
 ┃ ┃ ┃ ┣ gitattributes.txt
 ┃ ┃ ┃ ┣ merges.txt
 ┃ ┃ ┃ ┣ model.safetensors
 ┃ ┃ ┃ ┣ pytorch_model.bin
 ┃ ┃ ┃ ┣ README.txt
 ┃ ┃ ┃ ┣ rust_model.ot
 ┃ ┃ ┃ ┣ tf_model.h5
 ┃ ┃ ┃ ┣ tokenizer.json
 ┃ ┃ ┃ ┣ vocab.json
```

## roberta-base

To load `roberta-base` locally, follow [this link](https://huggingface.co/roberta-base) to download all files for `roberta-base` model, and store them under `Amphion/pretrained/roberta-base`, conforming to the following file structure tree:

```
Amphion
 ┣ pretrained
 ┃ ┣ roberta-base
 ┃ ┃ ┣ config.json
 ┃ ┃ ┣ dict.txt
 ┃ ┃ ┣ flax_model.msgpack
 ┃ ┃ ┣ gitattributes.txt
 ┃ ┃ ┣ merges.txt
 ┃ ┃ ┣ model.safetensors
 ┃ ┃ ┣ pytorch_model.bin
 ┃ ┃ ┣ README.txt
 ┃ ┃ ┣ rust_model.ot
 ┃ ┃ ┣ tf_model.h5
 ┃ ┃ ┣ tokenizer.json
 ┃ ┃ ┣ vocab.json
```

## wavlm

The official pretrained wavlm checkpoints can be available [here](https://huggingface.co/microsoft/wavlm-base-plus-sv). The file structure tree is as follows:

```
Amphion
 ┣ wavlm
 ┃ ┣ config.json
 ┃ ┣ preprocessor_config.json
 ┃ ┣ pytorch_model.bin
```