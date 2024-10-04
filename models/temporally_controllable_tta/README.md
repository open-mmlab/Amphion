# PicoAudio: Enabling Precise Timing and Frequency Controllability of Audio Events in Text-to-audio Generation
Duplicate of [github repo](https://github.com/zeyuxie29/PicoAudio)
[![arXiv](https://img.shields.io/badge/arXiv-2407.02869v2-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2407.02869v2)
[![githubio](https://img.shields.io/badge/GitHub.io-Audio_Samples-blue?logo=Github&style=flat-square)](https://zeyuxie29.github.io/PicoAudio.github.io/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/ZeyuXie/PicoAudio)
[![Hugging Face data](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/ZeyuXie/PicoAudio/tree/main)

**Bullet contribution**:
* A data simulation pipeline tailored specifically for controllable audio generation frameworks;
* Propose a timing-controllable audio generation framework, enabling precise control over the timing and frequency of sound event;
* Achieve any precise control related to timing by integrating of large language models.
  
## Inference
You can see the demo on the website [Huggingface Online Inference](https://huggingface.co/spaces/ZeyuXie/PicoAudio) and [Github Demo](https://zeyuxie29.github.io/PicoAudio.github.io).
Or you can use the *"inference.py"* script provided by website [Huggingface Inference](https://huggingface.co/spaces/ZeyuXie/PicoAudio/tree/main) to generate.
Huggingface Online Inference uses Gemini as a preprocessor, and we also provide a GPT preprocessing script consistent with the paper in *"llm_preprocess.py"*

## Simulated Dataset
Simulated data can be downloaded from (1) [HuggingfaceDataset](https://huggingface.co/datasets/ZeyuXie/PicoAudio/tree/main) or (2) [BaiduNetDisk](https://pan.baidu.com/s/1rGrcjtQCEYFpr3o6y9wI8Q?pwd=pico) with the extraction code "pico". 
The metadata is stored in *"data/meta_data/{}.json"*, one instance is as follows:
```python
{
  "filepath": "data/multi_event_test/syn_1.wav",
  "onoffCaption": "cat meowing at 0.5-2.0, 3.0-4.5 and whistling at 5.0-6.5 and explosion at 7.0-8.0, 8.5-9.5",
  "frequencyCaption": "cat meowing two times and whistling one times and explosion two times"
}
```
where:
* *"filepath"* indicates the path to the audio file.  
* *"frequencyCaption"* contains information about the occurrence frequency.
* *"onoffCaption"* contains on- & off-set information.
* For test file *"test-frequency-control_onoffFromGpt_{}.json"*, the *"onoffCaption"* is derived from *"frequencyCaption"* transformed by GPT-4, which is used for evaluation in the frequency control task.

## Training 
Download data into the *"data"* folder. 
The training and inference code can be found in the *"picoaudio"* folder.
```shell
cd picoaudio
pip install -r requirements.txt
```
To start traning:
```python
  accelerate launch runner/controllable_train.py
```

## Acknowledgement
Our code referred to the [AudioLDM](https://github.com/haoheliu/AudioLDM) and [Tango](https://github.com/declare-lab/tango). We appreciate their open-sourcing of their code.

<!--
### Hi there 👋
**PicoAudio/PicoAudio** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
