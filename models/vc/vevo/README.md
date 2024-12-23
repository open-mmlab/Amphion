# Vevo: Controllable Zero-Shot Voice Imitation with Self-Supervised Disentanglement

[![arXiv](https://img.shields.io/badge/OpenReview-Paper-COLOR.svg)](https://openreview.net/pdf?id=anQDiQZhDP)
[![hf](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-model-yellow)](https://huggingface.co/amphion/Vevo)
[![WebPage](https://img.shields.io/badge/WebPage-Demo-red)](https://versavoice.github.io/)

We present our reproduction of [Vevo](https://openreview.net/pdf?id=anQDiQZhDP), a versatile zero-shot voice imitation framework with controllable timbre and style. We invite you to explore the [audio samples](https://versavoice.github.io/) to experience Vevo's capabilities firsthand.

<br>
<div align="center">
<img src="../../../imgs/vc/vevo.png" width="100%">
</div>
<br>

We have included the following pre-trained Vevo models at Amphion:

- **Vevo-Timbre**: It can conduct *style-preserved* voice conversion.
- **Vevo-Style**: It can conduct style conversion, such as *accent conversion* and *emotion conversion*.
- **Vevo-Voice**: It can conduct *style-converted* voice conversion.
- **Vevo-TTS**: It can conduct *style and timbre controllable* TTS.

Besides, we also release the **content tokenizer** and **content-style tokenizer** proposed by Vevo. Notably, all these pre-trained models are trained on [Emilia](https://huggingface.co/datasets/amphion/Emilia-Dataset), containing 101k hours of speech data among six languages (English, Chinese, German, French, Japanese, and Korean).

## Quickstart

To run this model, you need to follow the steps below:

1. Clone the repository and install the environment.
2. Run the inference script.

### Clone and Environment Setup

#### 1. Clone the repository

```bash
git clone https://github.com/open-mmlab/Amphion.git
cd Amphion
```

#### 2. Install the environment

Before start installing, making sure you are under the `Amphion` directory. If not, use `cd` to enter.

Since we use `phonemizer` to convert text to phoneme, you need to install `espeak-ng` first. More details can be found [here](https://bootphon.github.io/phonemizer/install.html). Choose the correct installation command according to your operating system:

```bash
# For Debian-like distribution (e.g. Ubuntu, Mint, etc.)
sudo apt-get install espeak-ng
# For RedHat-like distribution (e.g. CentOS, Fedora, etc.) 
sudo yum install espeak-ng

# For Windows
# Please visit https://github.com/espeak-ng/espeak-ng/releases to download .msi installer
```

Now, we are going to install the environment. It is recommended to use conda to configure:

```bash
conda create -n vevo python=3.10
conda activate vevo

pip install -r models/vc/vevo/requirements.txt
```

### Inference Script

```bash
# Vevo-Timbre
python -m models.vc.vevo.infer_vevotimbre

# Vevo-Style
python -m models.vc.vevo.infer_vevostyle

# Vevo-Voice
python -m models.vc.vevo.infer_vevovoice

# Vevo-TTS
python -m models.vc.vevo.infer_vevotts
```

Running this will automatically download the pretrained model from HuggingFace and start the inference process. The result audio is by default saved in `models/vc/vevo/wav/output*.wav`, you can change this in the scripts  `models/vc/vevo/infer_vevo*.py`

## Citations

If you use Vevo in your research, please cite the following papers:

```bibtex
@article{vevo,
  title={Vevo: Controllable Zero-Shot Voice Imitation with Self-Supervised Disentanglement},
  journal={OpenReview},
  year={2024}
}

@inproceedings{amphion,
    author={Zhang, Xueyao and Xue, Liumeng and Gu, Yicheng and Wang, Yuancheng and Li, Jiaqi and He, Haorui and Wang, Chaoren and Song, Ting and Chen, Xi and Fang, Zihao and Chen, Haopeng and Zhang, Junan and Tang, Tze Ying and Zou, Lexiao and Wang, Mingxuan and Han, Jun and Chen, Kai and Li, Haizhou and Wu, Zhizheng},
    title={Amphion: An Open-Source Audio, Music and Speech Generation Toolkit},
    booktitle={{IEEE} Spoken Language Technology Workshop, {SLT} 2024},
    year={2024}
}
```