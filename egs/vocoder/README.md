# Amphion Vocoder Recipe

## Quick Start

We provide a [**beginner recipe**](gan/tfr_enhanced_hifigan/README.md) to demonstrate how to train a high quality HiFi-GAN speech vocoder. Specially, it is also an official implementation of our paper "[Multi-Scale Sub-Band Constant-Q Transform Discriminator for High-Fidelity Vocoder](https://arxiv.org/abs/2311.14957)". Some demos can be seen [here](https://vocodexelysium.github.io/MS-SB-CQTD/).

## Supported Models

Neural vocoder generates audible waveforms from acoustic representations, which is one of the key parts for current audio generation systems. Until now, Amphion has supported various widely-used vocoders according to different vocoder types, including:

- **GAN-based vocoders**, which we have provided [**a unified recipe**](gan/README.md) :
  - [MelGAN](https://arxiv.org/abs/1910.06711)
  - [HiFi-GAN](https://arxiv.org/abs/2010.05646)
  - [NSF-HiFiGAN](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts)
  - [BigVGAN](https://arxiv.org/abs/2206.04658)
  - [APNet](https://arxiv.org/abs/2305.07952)
- **Flow-based vocoders** (👨‍💻 developing):
  - [WaveGlow](https://arxiv.org/abs/1811.00002)
- **Diffusion-based vocoders**, which we have provided [**a unified recipe**](diffusion/README.md):
  - [Diffwave](https://arxiv.org/abs/2009.09761)
- **Auto-regressive based vocoders** (👨‍💻 developing):
  - [WaveNet](https://arxiv.org/abs/1609.03499)
  - [WaveRNN](https://arxiv.org/abs/1802.08435v1)