# NaturalSpeech2 Recipe

[![hf](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Spaces-yellow)](https://huggingface.co/spaces/amphion/NaturalSpeech2)

In this recipe, we will show how to train [NaturalSpeech2](https://arxiv.org/abs/2304.09116) using Amphion's infrastructure. NaturalSpeech2 is a zero-shot TTS architecture that predicts latent representations of a neural audio codec.

There are three stages in total:

1. Data processing
2. Training
3. Inference

> **NOTE:** You need to run every command of this recipe in the `Amphion` root path:
> ```bash
> cd Amphion
> ```

## 1. Data processing

You can use the commonly used TTS dataset to train NaturalSpeech2 model, e.g., LibriTTS, etc. We strongly recommend you use LibriTTS to train NaturalSpeech2 model for the first time. How to download dataset is detailed [here](../../datasets/README.md).

You can follow other Amphion TTS recipes for the data processing.

## 3. Training

```bash
sh egs/tts/NaturalSpeech2/run_train.sh
```

## 4. Inference

```bash
bash egs/tts/NaturalSpeech2/run_inference.sh --text "[The text you want to generate]"
```

We released a pre-trained Amphion NatrualSpeech2 model. So you can download the pre-trained model [here](https://huggingface.co/amphion/naturalspeech2_libritts) and generate speech following the above inference instruction.

We also provided an online [demo](https://huggingface.co/spaces/amphion/NaturalSpeech2), feel free to try it!

```bibtex
@article{shen2023naturalspeech,
  title={Naturalspeech 2: Latent diffusion models are natural and zero-shot speech and singing synthesizers},
  author={Shen, Kai and Ju, Zeqian and Tan, Xu and Liu, Yanqing and Leng, Yichong and He, Lei and Qin, Tao and Zhao, Sheng and Bian, Jiang},
  journal={arXiv preprint arXiv:2304.09116},
  year={2023}
}
```
