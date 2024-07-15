
# Amphion Text-to-Speech (TTS) Recipe

## Quick Start

We provide a **[beginner recipe](VALLE_V2/)** to demonstrate how to train a cutting edge TTS model. Specifically, it is Amphion's re-implementation for [VALL-E](https://arxiv.org/abs/2301.02111), which is a zero-shot TTS architecture that uses a neural codec language model with discrete codes.

## Supported Model Architectures

Until now, Amphion TTS supports the following models or architectures,
- **[FastSpeech2](FastSpeech2)**: A non-autoregressive TTS architecture that utilizes feed-forward Transformer blocks.
- **[VITS](VITS)**: An end-to-end TTS architecture that utilizes conditional variational autoencoder with adversarial learning
- **[VALL-E](VALLE_V2)**: A zero-shot TTS architecture that uses a neural codec language model with discrete codes. This model is our updated VALL-E implementation as of June 2024 which uses Llama as its underlying architecture. The previous version of VALL-E release can be found [here](VALLE)
- **[NaturalSpeech2](NaturalSpeech2)** (👨‍💻 developing): An architecture for TTS that utilizes a latent diffusion model to generate natural-sounding voices.
- **[Jets](Jets)**: An end-to-end TTS model that jointly trains FastSpeech2 and HiFi-GAN with an alignment module.

## Amphion TTS Demo
Here are some [TTS samples](https://openhlt.github.io/Amphion_TTS_Demo/) from Amphion.
