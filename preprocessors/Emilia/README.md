## Emilia Pipeline

This is the source code of the Emilia Pipeline. This README file will introduce the project and provide an installation guide.

### Pipeline

0. Preprocess
1. Source Separation: Long audio -> Long audio without BGM
2. Generate Speaker Diarization: Speaker diarization of long audio to get medium-length single-speaker audio
3. VAD (silero-vad): Short-length single-speaker audio
4. ASR: Process short single-speaker audio to get text
5. MOS Predictions: Assign a MOS score

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

P.S. If Huggingface is not accessible, try: `export HF_ENDPOINT=https://hf-mirror.com`

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
        "input_folder_path": "examples", // #1: Data input
        "SAMPLE_RATE": 24000
    },
    "separate": {
        "step1": {
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
        "primary_model_path": "/path/to/model/mos_model/DNSMOS/sig_bak_ovr.onnx" // #3: Model path
    }, 
    "huggingface_token": "<HUGGINGFACE_ACCESS_TOKEN>" // #4: Huggingface access token
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

3. Processed audio will be saved into `input_folder_path`_processed.


### 3. Check the Results

The processed audio files will be saved into `input_folder_path`_processed. The results will be saved in the same folder and include the following information:

1. **Wav file**: `<original_name>_<idx>.wav`
2. **JSON file**: `<original_name>.json`

```json
[
    {
        "text": "So, don't worry about that. But, like for instance, like yesterday was very hard for me to say, you know what, I should go to bed.", // Text
        "start": 67.18, // Start time in seconds
        "end": 74.41, // End time in seconds
        "language": "en", // Language
        "dnsmos": 3.44 // MOS score
    }
]
```
