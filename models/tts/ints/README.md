# Ints

## Overview

Ints is a text-to-speech model that can generate speech from text.

## Clone and Environment

This part, follow the steps below to clone the repository and install the environment.

```bash
git clone https://github.com/open-mmlab/Amphion.git

# enter the repositry directory
cd Amphion
```

Now, create a new conda environment and activate it.

```bash
conda create -n ints python=3.10
conda activate ints
```

Then, install the dependencies.

```bash
bash models/tts/ints/env.sh
```

## Run Gradio 🤗 Playground Locally

You can run the following command to interact with the playground:

```bash
python -m models.tts.ints.gradio_app --port 7860 --use_vllm
```

## Citations

If you use Ints in your research, please cite the following paper:

```bibtex
@inproceedings{intp,
  title={Advancing Zero-shot Text-to-Speech Intelligibility across Diverse Domains via Preference Alignment},
  author={Xueyao Zhang and Yuancheng Wang and Chaoren Wang and Ziniu Li and Zhuo Chen and Zhizheng Wu},
  booktitle    = {ACL},
  publisher    = {Association for Computational Linguistics},
  year={2025}
}

@article{amphion_v0.2,
  title        = {Overview of the Amphion Toolkit (v0.2)},
  author       = {Jiaqi Li and Xueyao Zhang and Yuancheng Wang and Haorui He and Chaoren Wang and Li Wang and Huan Liao and Junyi Ao and Zeyu Xie and Yiqiao Huang and Junan Zhang and Zhizheng Wu},
  year         = {2025},
  journal      = {arXiv preprint arXiv:2501.15442},
}

@inproceedings{amphion,
  author={Zhang, Xueyao and Xue, Liumeng and Gu, Yicheng and Wang, Yuancheng and Li, Jiaqi and He, Haorui and Wang, Chaoren and Song, Ting and Chen, Xi and Fang, Zihao and Chen, Haopeng and Zhang, Junan and Tang, Tze Ying and Zou, Lexiao and Wang, Mingxuan and Han, Jun and Chen, Kai and Li, Haizhou and Wu, Zhizheng},
  title={Amphion: An Open-Source Audio, Music and Speech Generation Toolkit},
  booktitle={{IEEE} Spoken Language Technology Workshop, {SLT} 2024},
  year={2024}
}
```
