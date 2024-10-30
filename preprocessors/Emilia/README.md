# Emilia: An Extensive, Multilingual, and Diverse Speech Dataset for Large-Scale Speech Generation
[![arXiv](https://img.shields.io/badge/arXiv-Paper-COLOR.svg)](https://arxiv.org/abs/2407.05361)  [![hf](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/amphion/Emilia-Dataset)  [![OpenDataLab](https://img.shields.io/badge/OpenDataLab-Dataset-blue)](https://opendatalab.com/Amphion/Emilia)  [![GitHub](https://img.shields.io/badge/GitHub-Repo-green)](https://github.com/open-mmlab/Amphion/tree/main/preprocessors/Emilia)  [![demo](https://img.shields.io/badge/WebPage-Demo-red)](https://emilia-dataset.github.io/Emilia-Demo-Page/)

This is the official repository 👑 for the **Emilia** dataset and the source code for **Emilia-Pipe** speech data preprocessing pipeline. 

<div align="center"><img width="500px" src="https://github.com/user-attachments/assets/b1c1a1f8-3149-4f96-8eb4-af470152a9b7" /></div>

## News 🔥
- **2024/09/01**: [Emilia](https://arxiv.org/abs/2407.05361) got accepted by IEEE SLT 2024! 🤗
- **2024/08/28**: Welcome to join Amphion's [Discord channel](https://discord.com/invite/drhW7ajqAG) to stay connected and engage with our community!
- **2024/08/27**: *The Emilia dataset is now publicly available!* Discover the most extensive and diverse speech generation dataset with 101k hours of in-the-wild speech data now at [![hf](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/amphion/Emilia-Dataset) or [![OpenDataLab](https://img.shields.io/badge/OpenDataLab-Dataset-blue)](https://opendatalab.com/Amphion/Emilia)! 👑👑👑
- **2024/07/08**: Our preprint [paper](https://arxiv.org/abs/2407.05361) is now available! 🔥🔥🔥
- **2024/07/03**: We welcome everyone to check our [homepage](https://emilia-dataset.github.io/Emilia-Demo-Page/) for our brief introduction for Emilia dataset and our demos!
- **2024/07/01**: We release of Emilia and Emilia-Pipe! We welcome everyone to explore it on our [GitHub](https://github.com/open-mmlab/Amphion/tree/main/preprocessors/Emilia)! 🎉🎉🎉

## Emilia Overview ⭐️
The **Emilia** dataset is a comprehensive, multilingual dataset with the following features:
- containing over *101k* hours of speech data;
- covering six different languages: *English (En), Chinese (Zh), German (De), French (Fr), Japanese (Ja), and Korean (Ko)*;
- containing diverse speech data with *various speaking styles* from diverse video platforms and podcasts on the Internet, covering various content genres such as talk shows, interviews, debates, sports commentary, and audiobooks.

The table below provides the duration statistics for each language in the dataset.

|   Language  | Duration (hours) |
|:-----------:|:----------------:|
|   English   |     46,828       |
|   Chinese   |     49,922       |
|   German    |     1,590        |
|   French    |     1,381        |
|   Japanese  |     1,715        |
|   Korean    |      217         |


The **Emilia-Pipe** is the first open-source preprocessing pipeline designed to transform raw, in-the-wild speech data into high-quality training data with annotations for speech generation. This pipeline can process one hour of raw audio into model-ready data in just a few minutes, requiring only the raw speech data. 

Detailed description for the Emilia and Emilia-Pipe could be found in our [paper](https://arxiv.org/abs/2407.05361).

## Emilia Dataset Usage 📖
The Emilia dataset is now publicly available at [![hf](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/amphion/Emilia-Dataset)! Users in mainland China can also download Emilia from [![OpenDataLab](https://img.shields.io/badge/OpenDataLab-Dataset-blue)](https://opendatalab.com/Amphion/Emilia)!

- To download from HuggingFace, you must first gain access to the dataset by completing the request form and accepting the terms of access. Please note that due to HuggingFace's file size limit of 50 GB, the `EN/EN_B00008.tar.gz` file has been split into `EN/EN_B00008.tar.gz.0` and `EN/EN_B00008.tar.gz.1`. Before extracting the files, you will need to run the following command to combine the parts: `cat EN/EN_B00008.tar.gz.* > EN/EN_B00008.tar.gz`

- To download from OpenDataLab (i.e., OpenXLab), please follow the guidence [here](https://speechteam.feishu.cn/wiki/PC8Ew5igviqBiJkElMJcJxNonJc) to gain access.

**ENJOY USING EMILIA!!!** 🔥

If you wish to re-build Emilia from scratch, you may download the raw audio files from the [provided URL list](https://huggingface.co/datasets/amphion/Emilia) and use our open-source [Emilia-Pipe](https://github.com/open-mmlab/Amphion/tree/main/preprocessors/Emilia) preprocessing pipeline to preprocess the raw data. Additionally, users can easily use Emilia-Pipe to preprocess their own raw speech data for custom needs. By open-sourcing the Emilia-Pipe code, we aim to enable the speech community to collaborate on large-scale speech generation research.

*Please note that Emilia does not own the copyright to the audio files; the copyright remains with the original owners of the videos or audio. Users are permitted to use this dataset only for non-commercial purposes under the CC BY-NC-4.0 license.*

## Emilia Dataset Structure ⛪️
The Emilia dataset will be structured as follows:

Structure example:
```
|-- openemilia_all.tar.gz (all .JSONL files are gzipped with directory structure in this file)
|-- EN (114 batches)
|   |-- EN_B00000.jsonl
|   |-- EN_B00000 (= EN_B00000.tar.gz)
|   |   |-- EN_B00000_S00000
|   |   |   `-- mp3
|   |   |       |-- EN_B00000_S00000_W000000.mp3
|   |   |       `-- EN_B00000_S00000_W000001.mp3
|   |   |-- ...
|   |-- ...
|   |-- EN_B00113.jsonl
|   `-- EN_B00113
|-- ZH (92 batches)
|-- DE (9 batches)
|-- FR (10 batches)
|-- JA (7 batches)
|-- KO (4 batches)

```
JSONL files example:
```
{"id": "EN_B00000_S00000_W000000", "wav": "EN_B00000/EN_B00000_S00000/mp3/EN_B00000_S00000_W000000.mp3", "text": " You can help my mother and you- No. You didn't leave a bad situation back home to get caught up in another one here. What happened to you, Los Angeles?", "duration": 6.264, "speaker": "EN_B00000_S00000", "language": "en", "dnsmos": 3.2927}
{"id": "EN_B00000_S00000_W000001", "wav": "EN_B00000/EN_B00000_S00000/mp3/EN_B00000_S00000_W000001.mp3", "text": " Honda's gone, 20 squads done. X is gonna split us up and put us on different squads. The team's come and go, but 20 squad, can't believe it's ending.", "duration": 8.031, "speaker": "EN_B00000_S00000", "language": "en", "dnsmos": 3.0442}
```

 
## Emilia-Pipe Overview 👀
The Emilia-Pipe includes the following major steps:

0. Standardization：Audio normalization
1. Source Separation: Long audio -> Long audio without BGM
2. Speaker Diarization: Get medium-length single-speaker speech data
3. Fine-grained Segmentation by VAD: Get 3-30s single-speaker speech segments
4. ASR: Get transcriptions of the speech segments
5. Filtering: Obtain the final processed dataset

## Setup Steps 👨‍💻

### 0. Prepare Environment

1. Install Python and CUDA.
2. Run the following commands to install the required packages:

    ```bash
    conda create -y -n AudioPipeline python=3.9 
    conda activate AudioPipeline

    bash env.sh
    ```

3. Download the model files from the third-party repositories.
    - Manually download the checkpoints of UVR-MDX-NET-Inst_HQ_3 ([UVR-MDX-NET-Inst_3.onnx](https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Inst_HQ_3.onnx)) and DNSMOS P.835 ([sig_bak_ovr.onnx](https://github.com/microsoft/DNS-Challenge/blob/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx)), then save their path for the next step configuration (i.e. #2  and #3 TODO).
    - Creat the access token to pyannote/speaker-diarization-3.1 following [the guide](https://huggingface.co/pyannote/speaker-diarization-3.1#requirements), then save it for the next step configuration (i.e. #4 TODO).
    - Make sure you have stable connection to GitHub and HuggingFace. The checkpoints of Silero and Whisperx-medium will be downloaded automatically on the pipeline's first run. 


### 1. Modify Config File

Change the config.json file according to the following TODOs.

```json
{
    "language": {
        "multilingual": true,
        "supported": [
            "zh",
            "en",
            "fr",
            "ja",
            "ko",
            "de"
        ]
    },
    "entrypoint": {
        // TODO: Fill in the input_folder_path. 
        "input_folder_path": "examples", // #1: Data input folder for processing
        "SAMPLE_RATE": 24000
    },
    "separate": {
        "step1": {
            // TODO: Fill in the source separation model's path. 
            "model_path": "/path/to/model/separate_model/UVR-MDX-NET-Inst_HQ_3.onnx", // #2: Model path
            "denoise": true,
            "margin": 44100,
            "chunks": 15,
            "n_fft": 6144,
            "dim_t": 8,
            "dim_f": 3072
        }
    },
    "mos_model": {
        // TODO: Fill in the DNSMOS prediction model's path. 
        "primary_model_path": "/path/to/model/mos_model/DNSMOS/sig_bak_ovr.onnx" // #3: Model path
    },
     // TODO: Fill in your huggingface access token for pynannote. 
    "huggingface_token": "<HUGGINGFACE_ACCESS_TOKEN>" // #4: Huggingface access token for pyannote
}
```

### 2. Run Script

1. Change the `input_folder_path` in `config.json` to the folder path where the downloaded audio files are stored (i.e. #1 TODO).
2. Run the following command to process the audio files:

```bash
conda activate AudioPipeline
export CUDA_VISIBLE_DEVICES=0  # Setting the GPU to run the pipeline, separate by comma

python main.py
```

3. Processed audio will be saved into `input_folder_path`_processed folder.


### 3. Check the Results

The processed audio (default 24k sample rate) files will be saved into `input_folder_path`_processed folder. The results for a single audio will be saved in a same folder with its original name and include the following information:

1. **MP3 file**: `<original_name>_<idx>.mp3` where `idx` is corresponding to the index in the JSON-encoded array.
2. **JSON file**: `<original_name>.json`

```json
[
    {
        "text": "So, don't worry about that. But, like for instance, like yesterday was very hard for me to say, you know what, I should go to bed.", // Transcription
        "start": 67.18, // Start timestamp, in second unit
        "end": 74.41, // End timestamp, in second unit
        "language": "en", // Language
        "dnsmos": 3.44 // DNSMOS P.835 score
    }
]
```

## TODOs 📝

Here are some potential improvements for the Emilia-Pipe pipeline:

- [x] Optimize the pipeline for better processing speed.
- [ ] Support input audio files larger than 4GB (calculated in WAVE format).
- [ ] Update source separation model to better handle noisy audio (e.g., reverberation).
- [ ] Ensure single speaker in each segment in the speaker diarization step.
- [ ] Move VAD to the first step to filter out non-speech segments. (for better speed)
- [ ] Extend ASR supported max length over 30s while keeping the speed.
- [ ] Fine-tune the ASR model to improve transcription accuracy on puctuation.
- [ ] Adding multimodal features to the pipeline for better transcription accuracy.
- [ ] Filter segments with unclean background noise, speaker overlap, hallucination transcriptions, etc.
- [ ] Labeling the data: speaker info (e.g., gender, age, native language, health), emotion, speaking style (pitch, rate, accent), acoustic features (e.g., fundamental frequency, formants), and environmental factors (background noise, microphone setup). Besides, non-verbal cues (e.g., laughter, coughing, silence, filters) and paralinguistic features could be labeled as well.

## Acknowledgement 🔔
We acknowledge the wonderful work by these excellent developers!
- Source Separation: [UVR-MDX-NET-Inst_HQ_3](https://github.com/TRvlvr/model_repo/releases/tag/all_public_uvr_models)
- VAD: [snakers4/silero-vad](https://github.com/snakers4/silero-vad)
- Speaker Diarization: [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- ASR: [m-bain/whisperX](https://github.com/m-bain/whisperX), using [faster-whisper](https://github.com/guillaumekln/faster-whisper) and [CTranslate2](https://github.com/OpenNMT/CTranslate2) backend.
- DNSMOS Prediction: [DNSMOS P.835](https://github.com/microsoft/DNS-Challenge)


## Reference 📖
If you use the Emilia dataset or the Emilia-Pipe pipeline, please cite the following papers:
```bibtex
@inproceedings{emilia,
    author={He, Haorui and Shang, Zengqiang and Wang, Chaoren and Li, Xuyuan and Gu, Yicheng and Hua, Hua and Liu, Liwei and Yang, Chen and Li, Jiaqi and Shi, Peiyang and Wang, Yuancheng and Chen, Kai and Zhang, Pengyuan and Wu, Zhizheng},
    title={Emilia: An Extensive, Multilingual, and Diverse Speech Dataset for Large-Scale Speech Generation},
    booktitle={Proc.~of SLT},
    year={2024}
}
```
```bibtex
@inproceedings{amphion,
    author={Zhang, Xueyao and Xue, Liumeng and Gu, Yicheng and Wang, Yuancheng and Li, Jiaqi and He, Haorui and Wang, Chaoren and Song, Ting and Chen, Xi and Fang, Zihao and Chen, Haopeng and Zhang, Junan and Tang, Tze Ying and Zou, Lexiao and Wang, Mingxuan and Han, Jun and Chen, Kai and Li, Haizhou and Wu, Zhizheng},
    title={Amphion: An Open-Source Audio, Music and Speech Generation Toolkit},
    booktitle={{IEEE} Spoken Language Technology Workshop, {SLT} 2024},
    year={2024}
}
```
