## MaskGCT: Zero-Shot Text-to-Speech with Masked Generative Codec Transformer

[![arXiv](https://img.shields.io/badge/arXiv-Paper-COLOR.svg)](https://arxiv.org/abs/2409.00750)
[![hf](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-model-yellow)](https://huggingface.co/amphion/maskgct)
[![hf](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-demo-pink)](https://huggingface.co/spaces/amphion/maskgct)
[![readme](https://img.shields.io/badge/README-Key%20Features-blue)](models/tts/maskgct/README.md)

## Overview

MaskGCT (**Mask**ed **G**enerative **C**odec **T**ransformer) is *a fully non-autoregressive TTS model that eliminates the need for explicit alignment information between text and speech supervision, as well as phone-level duration prediction*. MaskGCT is a two-stage model: in the first stage, the model uses text to predict semantic tokens extracted from a speech self-supervised learning (SSL) model, and in the second stage, the model predicts acoustic tokens conditioned on these semantic tokens. MaskGCT follows the *mask-and-predict* learning paradigm. During training, MaskGCT learns to predict masked semantic or acoustic tokens based on given conditions and prompts. During inference, the model generates tokens of a specified length in a parallel manner. Experiments with 100K hours of in-the-wild speech demonstrate that MaskGCT outperforms the current state-of-the-art zero-shot TTS systems in terms of quality, similarity, and intelligibility. Audio samples are available at [url](https://maskgct.github.io/).

<br>
<div align="center">
<img src="../../../imgs/maskgct/maskgct.png" width="100%">
</div>
<br>

## News

## Quickstart

## Pretrained Models

## Evaluation Results of MASKGCT