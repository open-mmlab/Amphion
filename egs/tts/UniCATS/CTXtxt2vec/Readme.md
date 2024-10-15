# UniCATS Text-to-Vector Recipe

In this recipe, we will show how to train the context-aware acoustic model introduced in
[UniCATS: A Unified Context-Aware Text-to-Speech Framework with Contextual VQ-Diffusion and Vocoding](https://arxiv.org/abs/2306.07547)
The acoustic model, CTX-txt2vec, employs the contextual Vector Quantized (VQ) diffusion method to predict semantic tokens from the input text. It generates vectors that represent the semantic context that seamless integrates with the surroundings, enabling speech continuation and editting. 

There are five stages in total:

1. Data preparation
2. Features extraction
3. Training
4. Inference
5. Vocoding to waveform

> **NOTE:** You need to run every command of this recipe in the `Amphion` root path:
> ```bash
> cd Amphion
> ```

## 1. Data Preparation

### Dataset Download

You can use the commonly used TTS dataset to train TTS model, e.g., LJSpeech, VCTK, LibriTTS, etc. However, the LibriTTS dataset is strongly recommended for learning the UniCATS project, as many of the data prepation steps are implemented so that you can focus on what really matter. And you don't need to download the dataset, which comes large in size, as the features are already extracted. 

If you are using dataset other than the LibriTTS, please first download the dataset. How to download dataset is detailed [here](../../datasets/README.md).

## 2. Features Extraction

### If you're using the recommanded LibriTTS dataset.

Please download the data manifest and the extracted feature from [here](https://huggingface.co/datasets/cantabile-kwok/libritts-all-kaldi-data/resolve/main/data_ctxt2v.zip) and [here](https://huggingface.co/datasets/cantabile-kwok/libritts-all-kaldi-data/resolve/main/feats_ctxt2v.zip), respectively. 

After that, unzip them to the project's root directory, Amphion/. You should get data/ and feats/ folders, which means you are all-set for training. You can move-on to step 3. 

Note that the extracted feature is for the complete LibriTTS dataset. 

### If you're using dataset other than LibriTTS

Please prepare the following for the dataset you are using:

#### Date manifest:

```
data
├── train_all
│         ├── duration    # the integer duration for each utterance. Frame shift is 10ms.
│         ├── feats.scp   # the VQ index for each utterance. 
│         ├── text   # the phone sequence for each utterance
│         └── utt2num_frames   # the number of frames of each utterance.
├── eval_all
│         ...  # similar four files
│── dev_all
│         ...
└── lang_1phn
          └── train_all_units.txt  # mapping between valid phones and their indexes
```

#### Kaldi-style features

```
feats
├── train_all
│         ├── feats.ark   # features matrix file for each utterance in the dataset
│         └── feats.scp   # features specifications for indexing utterance in feats.ark, same as the one in data manifest. 
├── eval_all
│         ...  # similar two files
│── dev_all
│         ...
└── vqidx
          └── codebook.npy  # the vector space for VQ (vector quantization)
          └── labels2vqidx  # The mapping between the label index for the UniCATS-CTXtxt2vec model (0 ~ 23632) and original vq-wav2vec codebook index pairs (2 indexes, each is 0 ~ 319) 
```
Note: If you are using other names for the data/ and feats/ folder, please update the in the exp_config.json file, corrospondingly. 

#### Data manifest:

1. duration: 
The duration file is of the following format: 
```
116_288045_000004_000000 3 8 8 5 8 5 3 3 12
```
where 116_288045_000004_000000 refers to the utterance (wav file) in the dataset, the following numbers are the duration of each phonemes in the uttrance (in term of frame).

2. feats.scp:
The feature specification file should have the following format:
```
1001_134708_000013_000000 feats/label/train_all/feats.ark:26
```
where where 116_288045_000004_000000 refers to the utterance in the dataset. feats/label/train_all/feats.ark is the path to the feature matrix file and 26 is the byte-shift to index the uttrance's location in the feats.ark file. 
You will need to download the [fairseq's vq-wav2vec model](https://dl.fbaipublicfiles.com/fairseq/wav2vec/vq-wav2vec_kmeans.pt) to get this file and the feat.ark from your dataset (see the section for getting feats.ark in feats/)

3. text
The mapping from uttrance to the phonemes. 
```
116_288045_000004_000000 SIL1 AH0 P EH1 R AH0 B AH0 L
```
where 116_288045_000004_000000 is the uttrance and "SIL1 AH0 P EH1 R AH0 B AH0 L" are the phonemes in this uttrance.


4. utt2num_frames
Counts the length of each uttrance in the dataset in terms of frames
```
116_288045_000003_000000 67
```
where 116_288045_000003_000000 is the utterance name, 67 is the total number of frames this utterance last. The duration of each phoneme in the uttrance should add up to the total number of frames of a utterence. 

#### features:
1. feat.ark and feat.scp

We will need the [fairseq's vq-wav2vec model](https://dl.fbaipublicfiles.com/fairseq/wav2vec/vq-wav2vec_kmeans.pt) to get features and index. The UniCATS project utilizes the fairseq's model to get the codebook and VQ. Details of the model can be found [here](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec#vq-wav2vec).
You can use the provided script at Amphion/utils/UniCATS/utils/get_ark_scp.py to generate the feats.ark and feats.scp from your dataset.

2. codebook.npy

We also need the [fairseq's vq-wav2vec model](https://dl.fbaipublicfiles.com/fairseq/wav2vec/vq-wav2vec_kmeans.pt). You can extract the codebook using script at Amphion/utils/UniCATS/utils/get_codebook.py

3. labels2vqidx

This file converts vq index to labels for model training. You can generate this file using script Amphion/utils/UniCATS/txt2vec_local/get_labelsvqidx_dict.py "path/to/feats.ark"


## 3. Training

### Configuration

The detailed configuration for training parameter is provided at config/UniCATS_txt2vec.json.
The recommended max traing epoch is 50, which takes quite some time to train. 
After the improvement on loss is nuance. 

### Run

Run the `run.sh` as the training stage (set  `--stage 2`). After the training starts, checkpoints and logs will be saved in Amphion/OUTPUT/YourExptName.

```bash
bash egs/tts/UniCATS/CTXtxt2vec/run.sh --stage 2 --name [YourExptName]
```


## 4. Inference (Decoding to VQ indexes)
The decoding of CTX-txt2vec rely on prompts that provide contextual information. In other words, before decoding, there should be a utt2prompt file that looks like:
```
1089_134686_000002_000001 1089_134686_000032_000008
1089_134686_000007_000005 1089_134686_000032_000008
```
where every line is organized as utt-to-synthesize prompt-utt. utt-to-synthesize is the wave file providing acoustic features, and prompt-utt provide the textual content for continuation. The utt-to-synthesize and prompt-utt keys should both be present in feats.scp for indexing.

### If you are using the LibriTTS data set
You can the [official utt2prompt file](https://cpdu.github.io/unicats/resources/testsetB_utt2prompt) for test set B in the paper. You can download that and save to data/eval_all/utt2prompt.

### If you are using other dataset. 
You should prepare you own utt2prompt file, following the format given above, and place it under data/eval_all/. Note that the utt-to-synthesize and prompt-utt keys should both be present in feats.scp for indexing.

After that, decoding with context prepended (a.k.a. continuation) can be performed by:

```bash
bash egs/tts/UniCATS/CTXtxt2vec/run.sh --stage 3
```

## Vocoding to waveform
It is recommended to use the CTXvec2wav for vocoding. Please refer to [CTXvec2wav/Readme.md](../CTXvec2wav/Readme.md).