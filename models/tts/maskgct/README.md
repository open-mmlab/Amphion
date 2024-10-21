## MaskGCT: Zero-Shot Text-to-Speech with Masked Generative Codec Transformer

[![arXiv](https://img.shields.io/badge/arXiv-Paper-COLOR.svg)](https://arxiv.org/abs/2409.00750)
[![hf](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-model-yellow)](https://huggingface.co/amphion/maskgct)
[![hf](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-demo-pink)](https://huggingface.co/spaces/amphion/maskgct)
[![readme](https://img.shields.io/badge/README-Key%20Features-blue)](../../../models/tts/maskgct/README.md)

## Overview

MaskGCT (**Mask**ed **G**enerative **C**odec **T**ransformer) is *a fully non-autoregressive TTS model that eliminates the need for explicit alignment information between text and speech supervision, as well as phone-level duration prediction*. MaskGCT is a two-stage model: in the first stage, the model uses text to predict semantic tokens extracted from a speech self-supervised learning (SSL) model, and in the second stage, the model predicts acoustic tokens conditioned on these semantic tokens. MaskGCT follows the *mask-and-predict* learning paradigm. During training, MaskGCT learns to predict masked semantic or acoustic tokens based on given conditions and prompts. During inference, the model generates tokens of a specified length in a parallel manner. Experiments with 100K hours of in-the-wild speech demonstrate that MaskGCT outperforms the current state-of-the-art zero-shot TTS systems in terms of quality, similarity, and intelligibility. Audio samples are available at [demo page](https://maskgct.github.io/).

<br>
<div align="center">
<img src="../../../imgs/maskgct/maskgct.png" width="100%">
</div>
<br>

## News

- **2024/10/19**: We release **MaskGCT**, a fully non-autoregressive TTS model that eliminates the need for explicit alignment information between text and speech supervision. MaskGCT is trained on [Emilia](https://huggingface.co/datasets/amphion/Emilia-Dataset) dataset and achieves SOTA zero-shot TTS perfermance.

## Quickstart

**Clone and install**

```bash
git clone https://github.com/open-mmlab/Amphion.git
# create env
bash ./models/tts/maskgct/env.sh
```

**Model download**

We provide the following pretrained checkpoints:


| Model Name          | Description   |    
|-------------------|-------------|
| [Acoustic Codec](https://huggingface.co/amphion/MaskGCT/tree/main/acoustic_codec)      | Converting speech to semantic tokens. |
| [Semantic Codec](https://huggingface.co/amphion/MaskGCT/tree/main/semantic_codec)      | Converting speech to acoustic tokens and reconstructing waveform from acoustic tokens. |
| [MaskGCT-T2S](https://huggingface.co/amphion/MaskGCT/tree/main/t2s_model)         | Predicting semantic tokens with text and prompt semantic tokens.             |
| [MaskGCT-S2A](https://huggingface.co/amphion/MaskGCT/tree/main/s2a_model)         | Predicts acoustic tokens conditioned on semantic tokens.              |

You can download all pretrained checkpoints from [HuggingFace](https://huggingface.co/amphion/MaskGCT/tree/main) or use huggingface api.

```python
from huggingface_hub import hf_hub_download

# download semantic codec ckpt
semantic_code_ckpt = hf_hub_download("amphion/MaskGCT" filename="semantic_codec/model.safetensors")

# download acoustic codec ckpt
codec_encoder_ckpt = hf_hub_download("amphion/MaskGCT", filename="acoustic_codec/model.safetensors")
codec_decoder_ckpt = hf_hub_download("amphion/MaskGCT", filename="acoustic_codec/model_1.safetensors")

# download t2s model ckpt
t2s_model_ckpt = hf_hub_download("amphion/MaskGCT", filename="t2s_model/model.safetensors")

# download s2a model ckpt
s2a_1layer_ckpt = hf_hub_download("amphion/MaskGCT", filename="s2a_model/s2a_model_1layer/model.safetensors")
s2a_full_ckpt = hf_hub_download("amphion/MaskGCT", filename="s2a_model/s2a_model_full/model.safetensors")
```

**Basic Usage**

You can use the following code to generate speech from text and a prompt speech (the code is also provided in [inference.py](../../../models/tts/maskgct/maskgct_inference.py)).

```python
from models.tts.maskgct.maskgct_utils import *
from huggingface_hub import hf_hub_download
import safetensors
import soundfile as sf

if __name__ == "__main__":

    # build model
    device = torch.device("cuda:0")
    cfg_path = "./models/tts/maskgct/config/maskgct.json"
    cfg = load_config(cfg_path)
    # 1. build semantic model (w2v-bert-2.0)
    semantic_model, semantic_mean, semantic_std = build_semantic_model(device)
    # 2. build semantic codec
    semantic_codec = build_semantic_codec(cfg.model.semantic_codec, device)
    # 3. build acoustic codec
    codec_encoder, codec_decoder = build_acoustic_codec(cfg.model.acoustic_codec, device)
    # 4. build t2s model
    t2s_model = build_t2s_model(cfg.model.t2s_model, device)
    # 5. build s2a model
    s2a_model_1layer = build_s2a_model(cfg.model.s2a_model.s2a_1layer, device)
    s2a_model_full =  build_s2a_model(cfg.model.s2a_model.s2a_full, device)

    # download checkpoint
    ...

    # load semantic codec
    safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
    # load acoustic codec
    safetensors.torch.load_model(codec_encoder, codec_encoder_ckpt)
    safetensors.torch.load_model(codec_decoder, codec_decoder_ckpt)
    # load t2s model
    safetensors.torch.load_model(t2s_model, t2s_model_ckpt)
    # load s2a model
    safetensors.torch.load_model(s2a_model_1layer, s2a_1layer_ckpt)
    safetensors.torch.load_model(s2a_model_full, s2a_full_ckpt)

    # inference
    prompt_wav_path = "./models/tts/maskgct/wav/prompt.wav"
    save_path = "[YOUR SAVE PATH]"
    prompt_text = " We do not break. We never give in. We never back down."
    target_text = "In this paper, we introduce MaskGCT, a fully non-autoregressive TTS model that eliminates the need for explicit alignment information between text and speech supervision."
    # Specify the target duration (in seconds). If target_len = None, we use a simple rule to predict the target duration.
    target_len = 18

    maskgct_inference_pipeline = MaskGCT_Inference_Pipeline(
        semantic_model,
        semantic_codec,
        codec_encoder,
        codec_decoder,
        t2s_model,
        s2a_model_1layer,
        s2a_model_full,
        semantic_mean,
        semantic_std,
        device,
    )

    recovered_audio = maskgct_inference_pipeline.maskgct_inference(
        prompt_wav_path, prompt_text, target_text, "en", "en", target_len=target_len
    )
    sf.write(save_path, recovered_audio, 24000)        
```

**Jupyter Notebook**

We also provide a [jupyter notebook](../../../models/tts/maskgct/maskgct_demo.ipynb) to show more details of MaskGCT inference.

## Training Dataset

We use the [Emilia](https://huggingface.co/datasets/amphion/Emilia-Dataset) dataset to train our models. Emilia is a multilingual and diverse in-the-wild speech dataset designed for large-scale speech generation. In this work, we use English and Chinese data from Emilia, each with 50K hours of speech (totaling 100K hours).


## Evaluation Results of MaskGCT

| System | SIM-O↑ | WER↓ | FSD↓ | SMOS↑ | CMOS↑ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| | | **LibriSpeech test-clean** |
| Ground Truth | 0.68 | 1.94 |  | 4.05±0.12 | 0.00 |
| VALL-E | 0.50 | 5.90 | - | 3.47 ±0.26 | -0.52±0.22 |
| VoiceBox | 0.64 | 2.03 | 0.762 | 3.80±0.17 | -0.41±0.13 |
| NaturalSpeech 3 | 0.67 | 1.94 | 0.786 | 4.26±0.10 | 0.16±0.14 |
| VoiceCraft | 0.45 | 4.68 | 0.981 | 3.52±0.21 | -0.33 ±0.16 |
| XTTS-v2 | 0.51 | 4.20 | 0.945 | 3.02±0.22 | -0.98 ±0.19 |
| MaskGCT | 0.687(0.723) | 2.634(1.976) | 0.886 | 4.27±0.14 | 0.10±0.16 |
| MaskGCT(gt length) | 0.697 | 2.012 | 0.746 | 4.33±0.11 | 0.13±0.13 |
| | | **SeedTTS test-en** |
| Ground Truth | 0.730 | 2.143 |  | 3.92±0.15 | 0.00 |
| CosyVoice | 0.643 | 4.079 | 0.316 | 3.52±0.17 | -0.41 ±0.18 |
| XTTS-v2 | 0.463 | 3.248 | 0.484 | 3.15±0.22 | -0.86±0.19 |
| VoiceCraft | 0.470 | 7.556 | 0.226 | 3.18±0.20 | -1.08 ±0.15 |
| MaskGCT | 0.717(0.760) | 2.623(1.283) | 0.188 | 4.24 ±0.12 | 0.03 ±0.14 |
| MaskGCT(gt length) | 0.728 | 2.466 | 0.159 | 4.13 ±0.17 | 0.12 ±0.15 |
| | | **SeedTTS test-zh** |
| Ground Truth | 0.750 | 1.254 |  | 3.86 ±0.17 | 0.00 |
| CosyVoice | 0.750 | 4.089 | 0.276 | 3.54 ±0.12 | -0.45 ±0.15 |
| XTTS-v2 | 0.635 | 2.876 | 0.413 | 2.95 ±0.18 | -0.81 ±0.22 |
| MaskGCT | 0.774(0.805) | 2.273(0.843) | 0.106 | 4.09 ±0.12 | 0.05 ±0.17 |
| MaskGCT(gt length) | 0.777 | 2.183 | 0.101 | 4.11 ±0.12 | 0.08±0.18 |

## Citations

If you use MaskGCT in your research, please cite the following paper:

```bibtex
@article{wang2024maskgct,
  title={MaskGCT: Zero-Shot Text-to-Speech with Masked Generative Codec Transformer},
  author={Wang, Yuancheng and Zhan, Haoyue and Liu, Liwei and Zeng, Ruihong and Guo, Haotian and Zheng, Jiachen and Zhang, Qiang and Zhang, Xueyao and Zhang, Shunsi and Wu, Zhizheng},
  journal={arXiv preprint arXiv:2409.00750},
  year={2024}
}

@inproceedings{amphion,
    author={Zhang, Xueyao and Xue, Liumeng and Gu, Yicheng and Wang, Yuancheng and Li, Jiaqi and He, Haorui and Wang, Chaoren and Song, Ting and Chen, Xi and Fang, Zihao and Chen, Haopeng and Zhang, Junan and Tang, Tze Ying and Zou, Lexiao and Wang, Mingxuan and Han, Jun and Chen, Kai and Li, Haizhou and Wu, Zhizheng},
    title={Amphion: An Open-Source Audio, Music and Speech Generation Toolkit},
    booktitle={{IEEE} Spoken Language Technology Workshop, {SLT} 2024},
    year={2024}
}
```
