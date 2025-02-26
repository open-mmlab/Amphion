# Emilia: A Large-Scale, Extensive, Multilingual, and Diverse Dataset for Speech Generation
[![arXiv](https://img.shields.io/badge/arXiv-Paper-COLOR.svg)](https://arxiv.org/abs/2407.05361)  [![hf](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/amphion/Emilia-Dataset)  [![OpenDataLab](https://img.shields.io/badge/OpenDataLab-Dataset-blue)](https://opendatalab.com/Amphion/Emilia)  [![GitHub](https://img.shields.io/badge/GitHub-Repo-green)](https://github.com/open-mmlab/Amphion/tree/main/preprocessors/Emilia)  [![demo](https://img.shields.io/badge/WebPage-Demo-red)](https://emilia-dataset.github.io/Emilia-Demo-Page/)

This is the official repository 👑 for the **Emilia** dataset and the source code for **Emilia-Pipe** speech data preprocessing pipeline. 

<div align="center"><img width="500px" src="https://github.com/user-attachments/assets/b1c1a1f8-3149-4f96-8eb4-af470152a9b7" /></div>

## News 🔥
- **2025/02/26**: *The Emilia-Large dataset, featuring over 200,000 hours of data, is now available!!!* Emilia-Large combines the original 101k-hour Emilia dataset (licensed under `CC BY-NC 4.0`) with the brand-new 114k-hour **Emilia-YODAS dataset** (licensed under `CC BY 4.0`)!!!
- **2025/01/27**: We release the extened version of Emilia's paper on [arXiv](https://arxiv.org/abs/2501.15907)! More experiments and more insights! 
- **2024/12/04**: We present Emilia at the [IEEE SLT 2024](https://2024.ieeeslt.org/)! 
- **2024/09/01**: [Emilia](https://arxiv.org/abs/2407.05361) got accepted by IEEE SLT 2024! 🤗
- **2024/08/28**: Welcome to join Amphion's [Discord channel](https://discord.com/invite/drhW7ajqAG) to stay connected and engage with our community!
- **2024/08/27**: *The Emilia dataset is now publicly available!* Discover the most extensive and diverse speech generation dataset with 101k hours of in-the-wild speech data now at [![hf](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/amphion/Emilia-Dataset) or [![OpenDataLab](https://img.shields.io/badge/OpenDataLab-Dataset-blue)](https://opendatalab.com/Amphion/Emilia)! 👑👑👑
- **2024/07/08**: Our preprint [paper](https://arxiv.org/abs/2407.05361) is now available! 🔥🔥🔥
- **2024/07/03**: We welcome everyone to check our [homepage](https://emilia-dataset.github.io/Emilia-Demo-Page/) for our brief introduction for Emilia dataset and our demos!
- **2024/07/01**: We release of Emilia and Emilia-Pipe! We welcome everyone to explore it on our [GitHub](https://github.com/open-mmlab/Amphion/tree/main/preprocessors/Emilia)! 🎉🎉🎉


## Emilia-Large Overview ⭐️

The **Emilia-Large** dataset is a comprehensive, multilingual dataset with the following features:
- with *Emilia* containing over *101k* hours and *Emilia-YODAS* containing over *114k* hours of speech data;
- covering six different languages: *English (En), Chinese (Zh), German (De), French (Fr), Japanese (Ja), and Korean (Ko)*;
- containing diverse speech data with *various speaking styles* from diverse video platforms and podcasts on the Internet, covering various content genres such as talk shows, interviews, debates, sports commentary, and audiobooks.

The table below provides the duration statistics for each language in the dataset.

|   Language  | Emilia Duration (hours) | Emilia-YODAS Duration (hours) | Total Duration (hours) |
|:-----------:|:-----------------------:|:----------------------------:|:----------------------:|
|   English   |       46.8k             |        92.2k                 |        139.0k         |
|   Chinese   |       49.9k             |        0.3k                 |        50.3k          |
|   German    |        1.6k             |        5.6k                  |        7.2k           |
|   French    |        1.4k             |        7.4k                  |        8.8k           |
|   Japanese  |        1.7k             |        1.1k                  |        2.8k           |
|   Korean    |        0.2k             |        7.3k                  |        7.5k           |
|   **Total** |     **101.7k**          |       **113.9k**             |      **215.6k**        |


The **Emilia-Pipe** is the first open-source preprocessing pipeline designed to transform raw, in-the-wild speech data into high-quality training data with annotations for speech generation. This pipeline can process one hour of raw audio into model-ready data in just a few minutes, requiring only the raw speech data. 

Detailed descriptions for the Emilia and Emilia-Pipe can be found in our [paper](https://arxiv.org/abs/2407.05361), and [extended version](https://arxiv.org/abs/2501.15907).

## Emilia Dataset Usage 📖
Emilia and Emilia-YODAS is publicly available at [HuggingFace](https://huggingface.co/datasets/amphion/Emilia-Dataset). Please check the README in HuggingFace for usage guideline.

If you wish to re-build Emilia from scratch, you may download the raw audio files from the [provided URL list](https://huggingface.co/datasets/amphion/Emilia) and use our open-source [Emilia-Pipe](https://github.com/open-mmlab/Amphion/tree/main/preprocessors/Emilia) preprocessing pipeline to preprocess the raw data. Additionally, users can easily use Emilia-Pipe to preprocess their own raw speech data for custom needs. By open-sourcing the Emilia-Pipe code, we aim to enable the speech community to collaborate on large-scale speech generation research.

*Please note that Emilia does not own the copyright to the audio files; the copyright remains with the original owners of the videos or audio. Users are permitted to use this dataset only for non-commercial purposes under the CC BY-NC-4.0 license.*

 
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
@inproceedings{emilialarge,
    author={He, Haorui and Shang, Zengqiang and Wang, Chaoren and Li, Xuyuan and Gu, Yicheng and Hua, Hua and Liu, Liwei and Yang, Chen and Li, Jiaqi and Shi, Peiyang and Wang, Yuancheng and Chen, Kai and Zhang, Pengyuan and Wu, Zhizheng},
    title={Emilia: A Large-Scale, Extensive, Multilingual, and Diverse Dataset for Speech Generation},
    booktitle={arXiv:2501.15907},
    year={2025}
}

@inproceedings{emilia,
    author={He, Haorui and Shang, Zengqiang and Wang, Chaoren and Li, Xuyuan and Gu, Yicheng and Hua, Hua and Liu, Liwei and Yang, Chen and Li, Jiaqi and Shi, Peiyang and Wang, Yuancheng and Chen, Kai and Zhang, Pengyuan and Wu, Zhizheng},
    title={Emilia: An Extensive, Multilingual, and Diverse Speech Dataset for Large-Scale Speech Generation},
    booktitle={Proc.~of SLT},
    year={2024}
}

@inproceedings{amphion,
    author={Zhang, Xueyao and Xue, Liumeng and Gu, Yicheng and Wang, Yuancheng and Li, Jiaqi and He, Haorui and Wang, Chaoren and Song, Ting and Chen, Xi and Fang, Zihao and Chen, Haopeng and Zhang, Junan and Tang, Tze Ying and Zou, Lexiao and Wang, Mingxuan and Han, Jun and Chen, Kai and Li, Haizhou and Wu, Zhizheng},
    title={Amphion: An Open-Source Audio, Music and Speech Generation Toolkit},
    booktitle={Proc.~of SLT},
    year={2024}
}
```
