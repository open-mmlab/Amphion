# Amphion Visualization Recipe

## Quick Start

We provides a **[beginner recipe](SingVisio/)** to demonstrate how to implement interactive visualization for classic audio, music and speech generative models. Specifically, it is also an official implementation of the paper "SingVisio: Visual Analytics of the Diffusion Model for Singing Voice Conversion", which can be accessed via [arXiv](https://arxiv.org/abs/2402.12660) or [Computers & Graphics](https://www.sciencedirect.com/science/article/pii/S0097849324001936). The **SingVisio** can be experienced [here](https://openxlab.org.cn/apps/detail/Amphion/SingVisio).

## Supported Models

As the unique feature of Amphion, visualization aims to introduce interactive visual analysis of some classical models for educational purposes, helping newcomers understand their inner workings. 

Until now, Amphion has supported the visualization tool for the following models:

- **SVC**:
    - **[MultipleContentsSVC](../svc/MultipleContentsSVC)**: A diffusion-based model for sining voice conversion
- **TTS**:
    - **[FastSpeech 2](../tts/FastSpeech2/)** (👨‍💻 developing): A typical transformer-based TTS model.
    - **[VITS](../tts/VITS/)** (👨‍💻 developing): A typical flow-based end-to-end TTS model.


