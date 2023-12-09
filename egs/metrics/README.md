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
  - Cosine similarity based on [Rawnet3](https://github.com/Jungjee/RawNet)
  - Cosine similarity based on [WeSpeaker](https://github.com/wenet-e2e/wespeaker) (👨‍💻 developing)

We provide a recipe to demonstrate how to objectively evaluate your generated audios. There are three steps in total:

1. Pretrained Models Preparation
2. Audio Data Preparation
3. Evaluation

## 1. Pretrained Models Preparation

If you want to calculate `RawNet3` based speaker similarity, you need to download the pretrained model first, as illustrated [here](../../pretrained/README.md).

## 2. Aduio Data Preparation

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
	--fs [Optional. To calculate all metrics in the specified sampling rate]
```

As for the metrics, an example is provided below:

```bash
--metrics "mcd pesq fad"
```

All currently available metrics keywords are listed below:

| Keys                  | Description                                |
| --------------------- | ------------------------------------------ |
| `fpc`                 | F0 Pearson Coefficients                    |
| `f0_periodicity_rmse` | F0 Periodicity Root Mean Square Error      |
| `f0rmse`              | F0 Root Mean Square Error                  |
| `v_uv_f1`             | Voiced/Unvoiced F1 Score                   |
| `energy_rmse`         | Energy Root Mean Square Error              |
| `energy_pc`           | Energy Pearson Coefficients                |
| `cer`                 | Character Error Rate                       |
| `wer`                 | Word Error Rate                            |
| `speaker_similarity`  | Cos Similarity based on RawNet3            |
| `fad`                 | Frechet Audio Distance                     |
| `mcd`                 | Mel Cepstral Distortion                    |
| `mstft`               | Multi-Resolution STFT Distance             |
| `pesq`                | Perceptual Evaluation of Speech Quality    |
| `si_sdr`              | Scale Invariant Signal to Distortion Ratio |
| `si_snr`              | Scale Invariant Signal to Noise Ratio      |
| `stoi`                | Short Time Objective Intelligibility       |
