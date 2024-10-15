# UniCATS Vector to Wave Recipe

In this recipe, we will explore the training process for CTX-vec2wav model, which was introduced in the research paper [UniCATS: A Unified Context-Aware Text-to-Speech Framework with Contextual VQ-Diffusion and Vocoding](https://arxiv.org/abs/2306.07547). The CTX-vec2wav model employs the context-aware vocoding technique that transforms semantic vectors into realistic speech waveforms. This vocoder leverages contextual information, ensuring that the generated speech is not only high in quality but also coherently matches the surrounding acoustic characteristics. This allows for improved speech synthesis, particularly in applications requiring continuity and contextual relevance in voice outputs.

There are four stages in total:

1. Data preparation
2. Features extraction
3. Training
4. Inference

> **NOTE:** You need to run every command of this recipe in the `Amphion` root path:
> ```bash
> cd Amphion
> ```


## Data Preparation

### Dataset download

You can use the commonly used TTS dataset to train TTS model, e.g., LJSpeech, VCTK, LibriTTS, etc. However, the LibriTTS dataset is strongly recommended for learning the UniCATS project, as many of the data prepation steps are implemented so that you can focus on what really matter. And you don't need to download the dataset, which comes large in size, as the features are already extracted.  
If you are using dataset other than the LibriTTS, please first download the dataset. How to download dataset is detailed [here](../../datasets/README.md).

## Features Extraction

### If you're using the recommanded LibriTTS dataset.

#### data manifest

Please download the data manifest for the LibriTTS dataset from [here](https://huggingface.co/datasets/cantabile-kwok/libritts-all-kaldi-data/resolve/main/data.zip). 

#### features

For the features, we need:

* the VQ index (feats.ark and feats.scp) from fairseq's vq-wav2vec model (Details of this model and its usage at the [UniCATS-CTXtxt2vec's feature extration section](../CTXtxt2vec/Readme.md)). You can simply [download it](https://huggingface.co/datasets/cantabile-kwok/libritts-all-kaldi-data/resolve/main/vqidx.zip) and unzip to Amphion/feats/vqindex folder.

* PPE auxiliary features. PPE stands for probability of voice, pitch and energy (all in log scale). Please [download it](https://huggingface.co/datasets/cantabile-kwok/libritts-all-kaldi-data/resolve/main/normed_ppe.zip) and unzip it to Amphion/feats/normed_ppe folder. 

* Mel spectrograms (FBanks). Run the following command to extract the mel spectrograms from the data manifest and the data features. Note that the script assumes that data/ and feats/ folder are directly placed under the project's root folder Amphion/.

```bash
nj= 8 # parallel jobs. Set this according to your CPU cores.
bash egs/tts/UniCATS/CTXvec2wav/extract_fbank.sh --nj $nj --stage 0 --stop_stage 1  # Default: 80-dim with 10ms frame shift
# Stage 0 extracts fbank in parallel. Stage 1 performs normalization.
```

Note: the path in data/xxx_all/wav.scp and the path in feats/normed_ppe/xxx_all/feats.scp are not specified to the framework's path when you first download them. When you call run.sh for training and inference (see later section), it calls fix_path.py in utils/UniCATS/utils automatically, which assume that the dataset, data/, and feats/ folder are directly under the the project directory Amphion/.

### If you're using the dataset other than LibriTTS

#### Data manifest

Please run the following script to get spk2utt (speaker to utterance), utt2spk (utterance to speaker), and wav.scp for development set, evaluation set, and train set.

```bash
bash utils/UniCATS/vec2wav_local/data_prep.sh dataset_dir destination_dir
# dataset dir: the path to a subset of your dataset, example: /LibriTTS/train_all
# destination_dir: the path of the data manifest folder, example: data/train_all
```

#### Features

* VQ index
The VQ index feats.ark and feats.scp are generated using the [fairseq's vq-wav2vec model](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec#vq-wav2vec). You need to [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/vq-wav2vec_kmeans.pt) the model and extract them from your dataset. Your can follow the steps introduced in [UniCATS-CTXtxt2vec's Readme.md](../CTXtxt2vec/Readme.md) to get these file from your dataset.

* PPE auxiliary features.
Please run the following to get the PPE auxiliary features:

```bash
bash utils/UniCATS/vec2wav_local/make_ppe.sh your_dataset_path log_output_path feature_output_path
```

* Mel spectrograms (FBanks). 
This step is essentially the same as the mel spectrogram when using the LibriTTS dataset. Please refer to the previous section for instruction when you have your data/ and feats/ ready. 


## Training

You can start traing by running the following when you have your data/ and feats/ ready as introduced in the previous section.

```bash
bash egs/tts/UniCATS/CTXvec2wav/run.sh --stage 2 --stop_stage 2 
```

The checkpoint will be saved to Amphion/exp/train_all_ctxv2w.v1/*pkl. Checkpoint trained on the LibriTTS dataset with sampling rate 16K and 24K are provided: [16K](https://huggingface.co/cantabile-kwok/ctx_vec2wav_libritts_all/resolve/main/ctx_v2w.pkl?download=true), [24K](https://huggingface.co/cantabile-kwok/ctx_vec2wav_libritts_all/resolve/main/ctx_v2w_24k.pkl?download=true).
Please place the .pkl file under your expriment dir (example: exp/train_all_ctxv2w.v1/ctx_v2w.pkl)

## Inference

You can infer on the entire dataset by

```bash
bash egs/tts/UniCATS/CTXvec2wav/run.sh --stage 3 --stop_stage 3
```

Or only infer a subset by:

```bash
bash egs/tts/UniCATS/CTXvec2wav/run.sh --stage 3 --stop_stage 3 --eval_set $which_set
# For example: "--eval_set dev_all"
```

The program loads the latest checkpoint in the experiment dir exp/train_all_ctxv2w.v1/*pkl.
