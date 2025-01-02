# Debatts - Mandarin Debate TTS Model

## Introduction
Debatts is an advanced text-to-speech (TTS) model specifically designed for Mandarin debate contexts. This innovative model leverages short audio prompts to learn and replicate speaker characteristics while dynamically adjusting speaking style by analyzing the audio of debate opponents. This capability allows Debatts to integrate seamlessly into debate scenarios, offering not just speech synthesis but a responsive adaptation to the changing dynamics of debate interactions.

## Environment Setup
To set up the necessary environment to run Debatts, please use the provided `env.sh` file. This file contains all the required dependencies and can be easily set up with the following Conda command:

**Clone and install**

```bash
git clone https://github.com/open-mmlab/Amphion.git
# create env
bash ./models/tts/debatts/env.sh
```

**Application**
We provide model application within the try_inference python code, with the supported example speeches. For more debating speech samples, users can refer to huggingface [Debatts-Data](https://huggingface.co/datasets/amphion/Debatts-Data). Modify the corresponding speech path in inference code.

## Continuous Updates
The Debatts project is actively being developed, with continuous updates aimed at enhancing model performance and expanding features. We encourage users to regularly check our repository for the latest updates and improvements to ensure optimal functionality and to take advantage of new capabilities as they become available.

## Citations
If you use MaskGCT in your research, please cite the following paper:

```bibtex
@misc{huang2024debattszeroshotdebatingtexttospeech,
      title={Debatts: Zero-Shot Debating Text-to-Speech Synthesis}, 
      author={Yiqiao Huang and Yuancheng Wang and Jiaqi Li and Haotian Guo and Haorui He and Shunsi Zhang and Zhizheng Wu},
      year={2024},
      eprint={2411.06540},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2411.06540}, 
}
```
