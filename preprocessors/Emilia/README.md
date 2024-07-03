## Emilia

This is the official repository for the **Emilia** dataset and the **Emilia-Pipe** source code.

Emilia is a comprehensive, multilingual dataset featuring over 101k hours of speech in six languages: English (En), Chinese (Zh), German (De), French (Fr), Japanese (Ja), and Korean (Ko). The dataset includes diverse speech samples with various speaking styles.

Emilia-Pipe is the first open-source preprocessing pipeline designed to transform raw, in-the-wild speech data into high-quality training data with annotations for speech generation. This pipeline can process one hour of raw audio into model-ready data in just a few minutes, requiring only the URLs of the audio or video sources. 

By downloading the raw audio files from our provided list of URLs and processing them with Emilia-Pipe, users can obtain the Emilia dataset. Additionally, users can easily use Emilia-Pipe to preprocess their own raw speech data for custom needs. By open-sourcing the Emilia-Pipe code, we aim to enable the speech community to collaborate on large-scale speech generation research.

This README file will introduce the usage of the Emilia-Pipe and provide an installation guide.

## Pipeline Overview

The Emilia-Pipe includes the following major steps:

0. Standardization：Audio normalization
1. Source Separation: Long audio -> Long audio without BGM
2. Speaker Diarization: Get medium-length single-speaker speech data
3. Fine-grained Segmentation by VAD: Get 3-30s single-speaker speech segments
4. ASR: Get transcriptions of the speech segments
5. Filtering: Obtain the final processed dataset

## Setup Steps

### 0. Prepare Environment

1. Install Python and CUDA.
2. Run the following commands to install the required packages:

```bash
conda create -y -n AudioPipeline python=3.9 
conda activate AudioPipeline

bash env.sh
```

3. Download the model files.
Bgm Separator: [UVR-MDX-NET-Inst_HQ_3](https://github.com/TRvlvr/model_repo/releases/tag/all_public_uvr_models)
VAD: [Silero](https://github.com/snakers4/silero-vad)
Speaker Diarization: [pyannote](https://github.com/pyannote/pyannote-audio)
ASR: [whisperx-medium](https://github.com/m-bain/whisperX)
AutoMOS: [DNSMOS P. 835](https://github.com/microsoft/DNS-Challenge)

### 1. Config File

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
     // TODO: Fill in your huggingface acess token for pynannote. 
    "huggingface_token": "<HUGGINGFACE_ACCESS_TOKEN>" // #4: Huggingface access token for pyannote
}
```

- #1: Data to be processed
- #2 - #3: Model path configuration
- #4: Huggingface access token


### 2. Running Script

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
