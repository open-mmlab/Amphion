# Amphion Singing Voice Conversion (VC) Recipe

## Quick Start

We provide a **[beginner recipe](Noro)** to demonstrate how to train a cutting edge VC model. Specifically, it is an official implementation of the paper "NORO: A Noise-Robust One-Shot Voice Conversion System with Hidden Speaker Representation Capabilities".

## Supported Model Architectures

Until now, Amphion has supported a noise-robust VC model with the following architecture:

<br>
<div align="center">
  <img src="../../imgs/vc/NoroVC.png" width="80%">
</div>
<br>

It has the following features:
1. Noise-Robust Voice Conversion: Utilizes a dual-branch reference encoding module and noise-agnostic contrastive speaker loss to maintain high-quality voice conversion in noisy environments.
2. One-shot Voice Conversion: Achieves timbre conversion using only one reference speech sample.
3. Speaker Representation Learning: Explores the potential of the reference encoder as a self-supervised speaker encoder.
