# Amphion Evaluation Recipe

## Supported Evaluation Metrics

Until now, Amphion Evaluation has supported the following objective metrics:

- **F0 Modeling**:
  - F0 Pearson Coefficients (FPC)
  - F0 Periodicity Root Mean Square Error (PeriodicityRMSE)
  - F0 Root Mean Square Error (F0RMSE)
  - Voiced/Unvoiced F1 Score (V/UV F1)
- **Energy Modeling**:
  - Energy Root Mean Square Error (EnergyRMSE)
  - Energy Pearson Coefficients (EnergyPC)
- **Intelligibility**:
  - Character Error Rate (CER) based on [Whipser](https://github.com/openai/whisper)
  - Word Error Rate (WER) based on [Whipser](https://github.com/openai/whisper)
- **Spectrogram Distortion**:
  - Frechet Audio Distance (FAD)
  - Mel Cepstral Distortion (MCD)
  - Multi-Resolution STFT Distance (MSTFT)
  - Perceptual Evaluation of Speech Quality (PESQ)
  - Short Time Objective Intelligibility (STOI)
  - Scale Invariant Signal to Distortion Ratio (SISDR)
  - Scale Invariant Signal to Noise Ratio (SISNR)
- **Speaker Similarity**:
  - Cosine similarity based on:
    - [Rawnet3](https://github.com/Jungjee/RawNet)
    - [Resemblyzer](https://github.com/resemble-ai/Resemblyzer)
    - [WavLM](https://huggingface.co/microsoft/wavlm-base-plus-sv)

We provide a recipe to demonstrate how to objectively evaluate your generated audios. There are three steps in total:

1. Pretrained Models Preparation
2. Audio Data Preparation
3. Evaluation

## 1. Pretrained Models Preparation

If you want to calculate `RawNet3` based speaker similarity, you need to download the pretrained model first, as illustrated [here](../../pretrained/README.md).

## 2. Audio Data Preparation

Prepare reference audios and generated audios in two folders, the `ref_dir` contains the reference audio and the `gen_dir` contains the generated audio. Here is an example.

```plaintext
 ┣ {ref_dir}
 ┃ ┣ sample1.wav
 ┃ ┣ sample2.wav
 ┣ {gen_dir}
 ┃ ┣ sample1.wav
 ┃ ┣ sample2.wav
```

You have to make sure that the pairwise **reference audio and generated audio are named the same**, as illustrated above (sample1 to sample1, sample2 to sample2).

## 3. Evaluation

Run the `run.sh` with specified refenrece folder, generated folder, dump folder and metrics.

```bash
cd Amphion
sh egs/metrics/run.sh \
	--reference_folder [Your path to the reference audios] \
	--generated_folder [Your path to the generated audios] \
	--dump_folder [Your path to dump the objective results] \
	--metrics [The metrics you need] \
	--fs [Optional. To calculate all metrics in the specified sampling rate] \
	--similarity_model [Optional. To choose the model for calculating the speaker similarity. Currently "rawnet", "wavlm" and "resemblyzer" are available. Default to "wavlm"] \
	--similarity_mode [Optional. To choose the mode for calculating the speaker similarity. "pairwith" for calculating a series of ground truth / prediction audio pairs to obtain the speaker similarity, and "overall" for computing the average score with all possible pairs between the refernece folder and generated folder. Default to "pairwith"] \
	--intelligibility_mode [Optionoal. To choose the mode for computing CER and WER. "gt_audio" means selecting the recognition content of the reference audio as the target, "gt_content" means using transcription as the target. Default to "gt_audio"] \
	--ltr_path [Optional. Path to the transcription file] \
	--language [Optional. Language for computing CER and WER. Default to "english"]
```

As for the metrics, an example is provided below:

```bash
--metrics "mcd pesq fad"
```

All currently available metrics keywords are listed below:

| Keys                      | Description                                |
| ------------------------- | ------------------------------------------ |
| `fpc`                     | F0 Pearson Coefficients                    |
| `f0_periodicity_rmse`     | F0 Periodicity Root Mean Square Error      |
| `f0rmse`                  | F0 Root Mean Square Error                  |
| `v_uv_f1`                 | Voiced/Unvoiced F1 Score                   |
| `energy_rmse`             | Energy Root Mean Square Error              |
| `energy_pc`               | Energy Pearson Coefficients                |
| `cer`                     | Character Error Rate                       |
| `wer`                     | Word Error Rate                            |
| `similarity`      | Speaker Similarity
| `fad`                     | Frechet Audio Distance                     |
| `mcd`                     | Mel Cepstral Distortion                    |
| `mstft`                   | Multi-Resolution STFT Distance             |
| `pesq`                    | Perceptual Evaluation of Speech Quality    |
| `si_sdr`                  | Scale Invariant Signal to Distortion Ratio |
| `si_snr`                  | Scale Invariant Signal to Noise Ratio      |
| `stoi`                    | Short Time Objective Intelligibility       |

For example, if want to calculate the speaker similarity between the synthesized audio and the reference audio with the same content, run:

```bash
sh egs/metrics/run.sh \
	--reference_folder [Your path to the reference audios] \
	--generated_folder [Your path to the generated audios] \
	--dump_folder [Your path to dump the objective results] \
	--metrics "similarity" \
	--similarity_model [Optional. To choose the model for calculating the speaker similarity. Currently "rawnet", "wavlm" and "resemblyzer" are available. Default to "wavlm"] \
	--similarity_mode "pairwith" \
```

If you don't have the reference audio with the same content, run the following to get the conteng-free similarity score:

```bash
sh egs/metrics/run.sh \
	--reference_folder [Your path to the reference audios] \
	--generated_folder [Your path to the generated audios] \
	--dump_folder [Your path to dump the objective results] \
	--metrics "similarity" \
	--similarity_model [Optional. To choose the model for calculating the speaker similarity. Currently "rawnet", "wavlm" and "resemblyzer" are available. Default to "wavlm"] \
	--similarity_mode "overall" \
```

## Troubleshooting
### FAD (Using Offline Models)
If your system is unable to access huggingface.co from the terminal, you might run into an error like "OSError: Can't load tokenizer for ...". To work around this, follow these steps to use local models:

1. Download the [bert-base-uncased](https://huggingface.co/bert-base-uncased), [roberta-base](https://huggingface.co/roberta-base), and [facebook/bart-base](https://huggingface.co/facebook/bart-base) models from `huggingface.co`. Ensure that the models are complete and uncorrupted. Place these directories within `Amphion/pretrained`. For a detailed file structure reference, see [This README](../../pretrained/README.md#optional-model-dependencies-for-evaluation) under `Amphion/pretrained`.
2. Inside the `Amphion/pretrained` directory, create a bash script with the content outlined below. This script will automatically update the tokenizer paths used by your system:
  ```bash
  #!/bin/bash

  BERT_DIR="bert-base-uncased"
  ROBERTA_DIR="roberta-base"
  BART_DIR="facebook/bart-base"
  PYTHON_SCRIPT="[YOUR ENV PATH]/lib/python3.9/site-packages/laion_clap/training/data.py"

  update_tokenizer_path() {
      local dir_name=$1
      local tokenizer_variable=$2
      local full_path

      if [ -d "$dir_name" ]; then
          full_path=$(realpath "$dir_name")
          if [ -f "$PYTHON_SCRIPT" ]; then
              sed -i "s|${tokenizer_variable}.from_pretrained(\".*\")|${tokenizer_variable}.from_pretrained(\"$full_path\")|" "$PYTHON_SCRIPT"
              echo "Updated ${tokenizer_variable} path to $full_path."
          else
              echo "Error: The specified Python script does not exist."
              exit 1
          fi
      else
          echo "Error: The directory $dir_name does not exist in the current directory."
          exit 1
      fi
  }

  update_tokenizer_path "$BERT_DIR" "BertTokenizer"
  update_tokenizer_path "$ROBERTA_DIR" "RobertaTokenizer"
  update_tokenizer_path "$BART_DIR" "BartTokenizer"

  echo "BERT, BART and RoBERTa Python script paths have been updated."

  ```

3. The script provided is intended to adjust the tokenizer paths in the `data.py` file, found under `/lib/python3.9/site-packages/laion_clap/training/`, within your specific environment. For those utilizing conda, you can determine your environment path by running `conda info --envs`. Then, substitute `[YOUR ENV PATH]` in the script with this path. If your environment is configured differently, you'll need to update the `PYTHON_SCRIPT` variable to correctly point to the `data.py` file.
4. Run the script. If it executes successfully, the tokenizer paths will be updated, allowing them to be loaded locally.

### WavLM-based Speaker Similarity (Using Offline Models)

If your system is unable to access huggingface.co from the terminal and you want to calculate `WavLM` based speaker similarity, you need to download the pretrained model first, as illustrated [here](../../pretrained/README.md).