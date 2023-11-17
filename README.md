# Amphion

Amphion (/æmˈfaɪən/) is a toolkit for Audio, Music, and Speech Generation. Its purpose is to support reproducible research and help junior researchers and engineers get started in the field of audio, music, and speech generation research and development. Amphion offers a unique feature: visualizations of classic models or architectures. We believe that these visualizations are beneficial for junior researchers and engineers who wish to gain a better understanding of the model.

The North-Star objective of Amphion is to offer a platform for studying the conversion of various inputs into audio. Amphion is designed to support individual generation tasks, including but not limited to,

- TTS: Text to Speech Synthesis (supported)
- SVS: Singing Voice Synthesis (planning)
- VC: Voice Conversion (planning)
- SVC: Singing Voice Conversion (supported)
- TTA: Text to Audio (supported)
- TTM: Text to Music (planning)
- more…

In addition to the specific generation tasks, Amphion also includes several vocoders and evaluation metrics. A vocoder is an important module for producing high-quality audio signals, while evaluation metrics are critical for ensuring consistent metrics in generation tasks.

## Key Features

### TTS: Text to speech

- Amphion achieves state-of-the-art performance when compared with existing open-source repositories on text-to-speech (TTS) systems.
- It supports the following models or architectures,
    - **[FastSpeech2](https://arxiv.org/abs/2006.04558)**: A non-autoregressive TTS architecture that utilizes feed-forward Transformer blocks.
    - **[VITS](https://arxiv.org/abs/2106.06103)**: An end-to-end TTS architecture that utilizes conditional variational autoencoder with adversarial learning
    - **[Vall-E](https://arxiv.org/abs/2301.02111)**: A zero-shot TTS architecture that uses a neural codec language model with discrete codes.
    - **[NaturalSpeech2](https://arxiv.org/abs/2304.09116)**: An architecture for TTS that utilizes a latent diffusion model to generate natural-sounding voices.

### SVC: Singing Voice Conversion

- It supports multiple content-based features from various pretrained models, including [WeNet](https://github.com/wenet-e2e/wenet), [Whisper](https://github.com/openai/whisper), and [ContentVec](https://github.com/auspicious3000/contentvec).
- It implements several state-of-the-art model architectures, including diffusion-based and Transformer-based models. The diffusion-based architecture uses [Bidirectoinal dilated CNN](https://openreview.net/pdf?id=a-xFK8Ymz5J) and [U-Net](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) as a backend and supports [DDPM](https://arxiv.org/pdf/2006.11239.pdf), [DDIM](https://arxiv.org/pdf/2010.02502.pdf), and [PNDM](https://arxiv.org/pdf/2202.09778.pdf). Additionally, it supports single-step inference based on the [Consistency Model](https://openreview.net/pdf?id=FmqFfMTNnv).

### TTA: Text to Audio

- **Supply TTA with latent diffusion model**, including:
    - **[AudioLDM](https://arxiv.org/abs/2301.12503)**: a two stage model with an autoencoder and a latent diffusion model

### Vocoder

- Amphion supports both classic and state-of-the-art neural vocoders, including
  - GAN-based vocoders: **[MelGAN](https://arxiv.org/abs/1910.06711)**, **[HiFi-GAN](https://arxiv.org/abs/2010.05646)**, **[NSF-HiFiGAN](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts)**, **[BigVGAN](https://arxiv.org/abs/2206.04658)**, **[APNet](https://arxiv.org/abs/2305.07952)**
  - Flow-based vocoders: **[WaveGlow](https://arxiv.org/abs/1811.00002)**
  - Diffusion-based vocoders: **[Diffwave](https://arxiv.org/abs/2009.09761)**
  - Auto-regressive based vocoders: **[WaveNet](https://arxiv.org/abs/1609.03499)**, **[WaveRNN](https://arxiv.org/abs/1802.08435v1)**

### Evaluation

We supply a comprehensive objective evaluation for the generated audios. The evaluation metrics contain:

- **F0 Modeling**
    - F0 Pearson Coefficients
    - F0 Periodicity Root Mean Square Error
    - F0 Root Mean Square Error
    - Voiced/Unvoiced F1 Score
- **Energy Modeling**
    - Energy Pearson Coefficients
    - Energy Root Mean Square Error
- **Intelligibility**
    - Character/Word Error Rate based [Whisper](https://github.com/openai/whisper)
- **Spectrogram Distortion**
    - Frechet Audio Distance (FAD)
    - Mel Cepstral Distortion (MCD)
    - Multi-Resolution STFT Distance (MSTFT)
    - Perceptual Evaluation of Speech Quality (PESQ)
    - Short Time Objective Intelligibility (STOI)
    - Signal to Noise Ratio (SNR)
- **Speaker Similarity**
    - Cosine similarity based [RawNet3](https://github.com/Jungjee/RawNet)

