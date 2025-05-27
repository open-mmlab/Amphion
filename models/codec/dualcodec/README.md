

# DualCodec: A Low-Frame-Rate, Semantically-Enhanced Neural Audio Codec for Speech Generation

[![arXiv](https://img.shields.io/badge/arXiv-2505.13000-brightgreen.svg?style=flat-square)](http://arxiv.org/abs/2505.13000)
[![githubio](https://img.shields.io/badge/GitHub.io-Demo_Page-blue?logo=Github&style=flat-square)](https://dualcodec.github.io/)
[![PyPI](https://img.shields.io/pypi/v/dualcodec?color=blue&label=PyPI&logo=PyPI&style=flat-square)](https://pypi.org/project/dualcodec/)
[![GitHub](https://img.shields.io/badge/Github-Dev_Release-pink?logo=Github&style=flat-square)](https://github.com/jiaqili3/dualcodec)
[![Amphion](https://img.shields.io/badge/Amphion-Stable_Release-blue?style=flat-square)](https://github.com/open-mmlab/Amphion/blob/main/models/codec/dualcodec/README.md)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VvUhsDffLdY5TdNuaqlLnYzIoXhvI8MK#scrollTo=Lsos3BK4J-4E)

## About
DualCodec is a low-frame-rate (12.5Hz or 25Hz), semantically-enhanced (with SSL feature) Neural Audio Codec designed to extract discrete tokens for efficient speech generation.

You can check out its [demo page](https://dualcodec.github.io/).
The overview of DualCodec system is shown in the following figure:

<!-- show dualcodec.png -->
![DualCodec](dualcodec.png)


## Installation
```bash
pip install dualcodec
```

## News
- 2025-05-19: DualCodec is accepted to Interspeech 2025!
- 2025-03-30: Added automatic downloading from huggingface. Uploaded some TTS models (DualCodec-VALLE, DualCodec-Voicebox).
- 2025-01-22: I added training and finetuning instructions for DualCodec, as well as a gradio interface. Version is v0.3.0.
- 2025-01-16: Finished writing DualCodec inference codes, the version is v0.1.0. Latest versions are synced to pypi.

## Available models
<!-- - 12hz_v1: DualCodec model trained with 12Hz sampling rate. 
- 25hz_v1: DualCodec model trained with 25Hz sampling rate. -->

| Model_ID   | Frame Rate | RVQ Quantizers | Semantic Codebook Size (RVQ-1 Size) | Acoustic Codebook Size (RVQ-rest Size) | Training Data       |
|-----------|------------|----------------------|-------------------------------------|----------------------------------------|---------------------|
| 12hz_v1   | 12.5Hz     | Any from 1-8 (maximum 8)        | 16384                               | 4096                                   | 100K hours Emilia  |
| 25hz_v1   | 25Hz       | Any from 1-12 (maximum 12)       | 16384                               | 1024                                   | 100K hours Emilia  |


## How to inference DualCodec
### 1. Programmic usage (automatically downloads checkpoints from Huggingface): 
```python
import dualcodec

model_id = "12hz_v1" # select from available Model_IDs, "12hz_v1" or "25hz_v1"

dualcodec_model = dualcodec.get_model(model_id)
dualcodec_inference = dualcodec.Inference(dualcodec_model=dualcodec_model, device="cuda")

# do inference for your wav
import torchaudio
audio, sr = torchaudio.load("YOUR_WAV.wav")
# resample to 24kHz
audio = torchaudio.functional.resample(audio, sr, 24000)
audio = audio.reshape(1,1,-1)
audio = audio.to("cuda")
# extract codes, for example, using 8 quantizers here:
semantic_codes, acoustic_codes = dualcodec_inference.encode(audio, n_quantizers=8)
# semantic_codes shape: torch.Size([B, 1, T])
# acoustic_codes shape: torch.Size([B, n_quantizers-1, T])

# produce output audio
out_audio = dualcodec_inference.decode(semantic_codes, acoustic_codes)

# save output audio
torchaudio.save("out.wav", out_audio.cpu().squeeze(0), 24000)
```


### 2. Alternative usage with local checkpoints
First, download checkpoints to local: 
```
# export HF_ENDPOINT=https://hf-mirror.com      # uncomment this to use huggingface mirror if you're in China
huggingface-cli download facebook/w2v-bert-2.0 --local-dir w2v-bert-2.0
huggingface-cli download amphion/dualcodec dualcodec_12hz_16384_4096.safetensors dualcodec_25hz_16384_1024.safetensors w2vbert2_mean_var_stats_emilia.pt --local-dir dualcodec_ckpts
```
The second command downloads the two DualCodec model (12hz_v1 and 25hz_v1) checkpoints and a w2v-bert-2 mean and variance statistics to the local directory `dualcodec_ckpts`.

Then you can use the following code to inference DualCodec with local checkpoints.
```python
import dualcodec

w2v_path = "./w2v-bert-2.0" # your downloaded path
dualcodec_model_path = "./dualcodec_ckpts" # your downloaded path
model_id = "12hz_v1" # select from available Model_IDs, "12hz_v1" or "25hz_v1"

dualcodec_model = dualcodec.get_model(model_id, dualcodec_model_path)
dualcodec_inference = dualcodec.Inference(dualcodec_model=dualcodec_model, dualcodec_path=dualcodec_model_path, w2v_path=w2v_path, device="cuda")

# do inference for your wav
import torchaudio
audio, sr = torchaudio.load("YOUR_WAV.wav")
# resample to 24kHz
audio = torchaudio.functional.resample(audio, sr, 24000)
audio = audio.reshape(1,1,-1)
audio = audio.to("cuda")
# extract codes, for example, using 8 quantizers here:
semantic_codes, acoustic_codes = dualcodec_inference.encode(audio, n_quantizers=8)
# semantic_codes shape: torch.Size([1, 1, T])
# acoustic_codes shape: torch.Size([1, n_quantizers-1, T])

# produce output audio. If `acoustic_codes=None` is passed, will decode only semantic codes (RVQ-1)
out_audio = dualcodec_inference.decode(semantic_codes, acoustic_codes)

# save output audio
torchaudio.save("out.wav", out_audio.cpu().squeeze(0), 24000)
```

See "example.ipynb" for a running example.

### 3. Google Colab
The notebook provides a demo of reconstructing audios using different number of RVQ layers:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VvUhsDffLdY5TdNuaqlLnYzIoXhvI8MK#scrollTo=Lsos3BK4J-4E)

### 4. Gradio interface
If you want to use the Gradio interface, you can run the following command:
```bash
python -m dualcodec.app
```
This will launch an app that allows you to upload a wav file and get the output wav file.

## DualCodec-based TTS models
Models available: 
- DualCodec-VALLE: A super fast 12.5Hz VALL-E TTS model based on DualCodec.
- DualCodec-Voicebox: A flow matching decoder for DualCodec 12.5Hz's semantic codes. (this can be used as the second stage of tts). The component alone is not a TTS.

To continue, first install other necessary components for training:
```bash
pip install "dualcodec[tts]"
```
Alternatively, if you want to install from source,
```bash
pip install -e .[tts]
```

### DualCodec-VALLE
DualCodec-VALLE is a TTS model based on DualCodec. It is trained with 12Hz sampling rate and 8 quantizers. The model is trained on 100K hours of Emilia data.
#### CLI Inference
```bash
python -m dualcodec.infer.valle.cli_valle_infer --ref_audio <path_to_ref_audio> --ref_text "TEXT OF YOUR REF AUDIO" --gen_text "This is the generated text" --output_dir test --output_file test.wav
```
You can also leave all options empty and it will use the default values.

#### Gradio interface
```bash
python -m dualcodec.infer.valle.gradio_valle_demo
```

### DualCodec-Voicebox
#### CLI Inference
```bash
python -m dualcodec.infer.voicebox.cli_voicebox_infer --ref_audio <path_to_ref_audio> --output_dir test --output_file test.wav
```
You can also leave all options empty and it will use the default values.



### FAQ
If you meet problems with environment in this stage, try the following:
```
pip install -U wandb protobuf transformers
```


## Training DualCodec from scratch
1. Install other necessary components for training:
```bash
pip install "dualcodec[tts]"
```
2. Clone this repository and `cd` to the project root folder (the folder that contains this readme):
```bash
git clone https://github.com/open-mmlab/Amphion.git
cd Amphion/models/codec/dualcodec/
```

3. To run example training on example Emilia German data:
```bash
accelerate launch train.py --config-name=dualcodec_train \
model=dualcodec_12hz_16384_4096_8vq \
trainer.batch_size=3 \
data.segment_speech.segment_length=24000
```
This trains from scratch a v1_12hz model with a training batch size of 3. (typically you need larger batch sizes like 10)

To train a v1_25Hz model:
```bash
accelerate launch train.py --config-name=dualcodec_train \
model=dualcodec_25hz_16384_1024_12vq \
trainer.batch_size=3 \
data.segment_speech.segment_length=24000

```



## Finetuning DualCodec
1. Install other necessary components for training:
```bash
pip install "dualcodec[train]"
```
2. Clone this repository and `cd` to the project root folder (the folder that contains this readme).

3. Get discriminator checkpoints:
```bash
huggingface-cli download amphion/dualcodec --local-dir dualcodec_ckpts
```

4. To run example finetuning on Emilia German data (streaming, no need to download files. Need network access to Huggingface):
```bash
accelerate launch train.py --config-name=dualcodec_ft_12hzv1 \
trainer.batch_size=3 \
data.segment_speech.segment_length=24000
```
This finetunes a 12hz_v1 model with a training batch size of 3. (typically you need larger batch sizes like 10)

To finetune a 25Hz_V1 model:
```bash
accelerate launch train.py --config-name=dualcodec_ft_25hzv1 \
trainer.batch_size=3 \
data.segment_speech.segment_length=24000
```


## Citation
If you find this work useful for your research, please cite our paper:

```bibtex
@inproceedings{dualcodec,
  title     = {DualCodec: A Low-Frame-Rate, Semantically-Enhanced Neural Audio Codec for Speech Generation},
  author    = {Li, Jiaqi and Lin, Xiaolong and Li, Zhekai and Huang, Shixi and Wang, Yuancheng and Wang, Chaoren and Zhan, Zhenpeng and Wu, Zhizheng},
  booktitle = {Proceedings of Interspeech 2025},
  year      = {2025}
}
```

If you use the pre-trained models or training recipe of Amphion, please also cite:

```bibtex
@article{amphion2,
  title        = {Overview of the Amphion Toolkit (v0.2)},
  author       = {Jiaqi Li and Xueyao Zhang and Yuancheng Wang and Haorui He and Chaoren Wang and Li Wang and Huan Liao and Junyi Ao and Zeyu Xie and Yiqiao Huang and Junan Zhang and Zhizheng Wu},
  year         = {2025},
  journal      = {arXiv preprint arXiv:2501.15442},
}

@inproceedings{amphion,
    author={Xueyao Zhang and Liumeng Xue and Yicheng Gu and Yuancheng Wang and Jiaqi Li and Haorui He and Chaoren Wang and Ting Song and Xi Chen and Zihao Fang and Haopeng Chen and Junan Zhang and Tze Ying Tang and Lexiao Zou and Mingxuan Wang and Jun Han and Kai Chen and Haizhou Li and Zhizheng Wu},
    title={Amphion: An Open-Source Audio, Music and Speech Generation Toolkit},
    booktitle={{IEEE} Spoken Language Technology Workshop, {SLT} 2024},
    year={2024}
}
```
