## Emilia: An Extensive, Multilingual, and Diverse Speech Dataset for Large-Scale Speech Generation
[![arXiv](https://img.shields.io/badge/arXiv-Paper-COLOR.svg)](https://arxiv.org/abs/2407.05361) 
[![hf](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/amphion/Emilia) 
[![demo](https://img.shields.io/badge/WebPage-Demo-red)](https://emilia-dataset.github.io/Emilia-Demo-Page/)

This is the official repository 👑 for the **Emilia** dataset and the source code for **Emilia-Pipe** speech data preprocessing pipeline. 

## News 🔥
- **2024/07/08**: Our preprint [paper](https://arxiv.org/abs/2407.05361) is now available! 🔥🔥🔥
- **2024/07/03**: We welcome everyone to check our [homepage](https://emilia-dataset.github.io/Emilia-Demo-Page/) for our brief introduction for Emilia dataset and our demos!
- **2024/07/01**: We release of Emilia and Emilia-Pipe! We welcome everyone to explore it! 🎉🎉🎉

## About ⭐️
🎤 **Emilia** is a comprehensive, multilingual dataset with the following features:
- containing over *101k* hours of speech data;
- covering six different languages: *English (En), Chinese (Zh), German (De), French (Fr), Japanese (Ja), and Korean (Ko)*;
- containing diverse speech data with *various speaking styles*;
  
Detailed description for the dataset could be found in our [paper](https://arxiv.org/abs/2407.05361).

🛠️ **Emilia-Pipe** is the first open-source preprocessing pipeline designed to transform raw, in-the-wild speech data into high-quality training data with annotations for speech generation. This pipeline can process one hour of raw audio into model-ready data in just a few minutes, requiring only the raw speech data. 

*To use the Emilia dataset, you can download the raw audio files from our provided source URL list on [HuggingFace](https://huggingface.co/datasets/amphion/Emilia) and use our open-source [Emilia-Pipe](.) preprocessing pipeline to preprocess the raw data and rebuild the dataset.*

*Please note that Emilia doesn't own the copyright of the audios; the copyright remains with the original owners of the video or audio. Additionally, users can easily use Emilia-Pipe to preprocess their own raw speech data for custom needs.*

By open-sourcing the Emilia-Pipe code, we aim to enable the speech community to collaborate on large-scale speech generation research.

This following README will introduce the installation and usage guide of the Emilia-Pipe.

## Pipeline Overview 👀

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
@article{emilia,
      title={Emilia: An Extensive, Multilingual, and Diverse Speech Dataset for Large-Scale Speech Generation},
      author={He, Haorui and Shang, Zengqiang and Wang, Chaoren and Li, Xuyuan and Gu, Yicheng and Hua, Hua and Liu, Liwei and Yang, Chen and Li, Jiaqi and Shi, Peiyang and Wang, Yuancheng and Chen, Kai and Zhang, Pengyuan and Wu, Zhizheng},
      journal={arXiv},
      volume={abs/2407.05361}
      year={2024}
}
```
```bibtex
@article{amphion,
      title={Amphion: An Open-Source Audio, Music and Speech Generation Toolkit}, 
      author={Zhang, Xueyao and Xue, Liumeng and Gu, Yicheng and Wang, Yuancheng and He, Haorui and Wang, Chaoren and Chen, Xi and Fang, Zihao and Chen, Haopeng and Zhang, Junan and Tang, Tze Ying and Zou, Lexiao and Wang, Mingxuan and Han, Jun and Chen, Kai and Li, Haizhou and Wu, Zhizheng},
      journal={arXiv},
      volume={abs/2312.09911}
      year={2024},
}
```
