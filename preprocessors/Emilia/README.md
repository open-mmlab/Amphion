## Emilia: An Extensive, Multilingual, and Diverse Speech Dataset for Large-Scale Speech Generation
This is the official repository 👑 for the **Emilia** dataset and the source code for **Emilia-Pipe** speech data preprocessing pipeline. 

## News 🔥
- 24/07/03: We welcome everyone to check our [homepage](https://emilia-dataset.github.io/Emilia-Demo-Page/) for our brief introduction for Emilia dataset and our demos!
- 24/07/01: We release of Emilia and Emilia-Pipe! We welcome everyone to explore it! 🎉🎉🎉

## About ⭐️
🎤 **Emilia** is a comprehensive, multilingual dataset with the following features:
- containing over *101k* hours of speech data;
- covering six different languages: *English (En), Chinese (Zh), German (De), French (Fr), Japanese (Ja), and Korean (Ko)*;
- containing diverse speech data with *various speaking styles*;
  
Detailed description for the dataset could be found in our paper.

🛠️ **Emilia-Pipe** is the first open-source preprocessing pipeline designed to transform raw, in-the-wild speech data into high-quality training data with annotations for speech generation. This pipeline can process one hour of raw audio into model-ready data in just a few minutes, requiring only the URLs of the audio or video sources. 

*To use the Emilia dataset, you can download the raw audio files from the [provided URL list](https://huggingface.co/datasets/amphion/Emilia) and use our open-source [Emilia-Pipe](https://github.com/open-mmlab/Amphion/tree/main/preprocessors/Emilia) preprocessing pipeline to preprocess the raw data and rebuild the dataset. Please note that Emilia doesn't own the copyright of the audios; the copyright remains with the original owners of the video or audio. Additionally, users can easily use Emilia-Pipe to preprocess their own raw speech data for custom needs.*

By open-sourcing the Emilia-Pipe code, we aim to enable the speech community to collaborate on large-scale speech generation research.

This README file will introduce the usage of the Emilia-Pipe and provide an installation guide.

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
We acknowledge the wonderful work by these excellent developers!
- Source Separation: [UVR-MDX-NET-Inst_HQ_3](https://github.com/TRvlvr/model_repo/releases/tag/all_public_uvr_models)
- VAD: [Silero](https://github.com/snakers4/silero-vad)
- Speaker Diarization: [pyannote](https://github.com/pyannote/pyannote-audio)
- ASR: [whisperx-medium](https://github.com/m-bain/whisperX)
- DNSMOS Prediction: [DNSMOS P.835](https://github.com/microsoft/DNS-Challenge)

The checkpoints of UVR-MDX-NET-Inst_HQ_3([UVR-MDX-NET-Inst_3.onnx
](https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Inst_HQ_3.onnx)) and DNSMOS P.835([sig_bak_ovr.onnx](https://github.com/microsoft/DNS-Challenge/blob/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx)) need to be downloaded manually and their local storage paths need to be written to the config file as the next step.
The checkpoints of Silero and Whisperx-medium will be downloaded automatically when the pipeline is first run. 
The pyannote checkpoint also will be downloaded automatically if your huggingface access token has been written to the config file as the next step. 


### 1. Modify Config File

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
        "input_folder_path": "examples", // #1: Data input
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

1. Change the `input_folder_path` in `config.json` to the folder path where the downloaded audio files are stored
2. Run the following command to process the audio files:

```bash
conda activate AudioPipeline
export CUDA_VISIBLE_DEVICES=0  # Setting the GPU to run the pipeline

python main.py
```

3. Processed audio will be saved into `input_folder_path_processed`.


### 3. Check the Results

The processed audio (default 24k sample rate) files will be saved into `input_folder_path_processed`. The results will be saved in the same folder and include the following information:

1. **MP3 file**: `<original_name>_<idx>.mp3`
2. **JSON file**: `<original_name>.json`

```json
[
    {
        "text": "So, don't worry about that. But, like for instance, like yesterday was very hard for me to say, you know what, I should go to bed.", // Transcription
        "start": 67.18, // Start timestamp
        "end": 74.41, // End timestamp
        "language": "en", // Language
        "dnsmos": 3.44 // DNSMOS score
    }
]
```

## Reference 📖

```bibtex
@article{amphion,
      title={Amphion: An Open-Source Audio, Music and Speech Generation Toolkit}, 
      author={Zhang, Xueyao and Xue, Liumeng and Gu, Yicheng and Wang, Yuancheng and He, Haorui and Wang, Chaoren and Chen, Xi and Fang, Zihao and Chen, Haopeng and Zhang, Junan and Tang, Tze Ying and Zou, Lexiao and Wang, Mingxuan and Han, Jun and Chen, Kai and Li, Haizhou and Wu, Zhizheng},
      year={2024},
      volume={abs/2312.09911}
}
```
