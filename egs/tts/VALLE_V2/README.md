# VALL-E
## Introduction
This is an unofficial PyTorch implementation of VALL-E, a zero-shot voice cloning model via neural codec language modeling ([paper link](https://arxiv.org/abs/2301.02111)). 
If trained properly, this model could match the performance specified in the original paper.

## Change notes
This is a refined version compared to the first version of VALL-E in Amphion, we have changed the underlying implementation to Llama
to provide better model performance, faster training speed, and more readable codes.
This can be a great tool if you want to learn speech language models and its implementation.

## Installation requirement 

Set up your environemnt as in Amphion README (you'll need a conda environment, and we recommend using Linux). A GPU is recommended if you want to train this model yourself.
For inferencing our pretrained models, you could generate samples even without a GPU.
To ensure your transformers library can run the code, we recommend additionally running:
```bash
pip install -U transformers==4.41.2
```

## Inferencing pretrained VALL-E models
### Download pretrained weights
You need to download our pretrained weights from huggingface. 

Script to download AR and NAR model checkpoint: 
```bash
huggingface-cli download amphion/valle valle_ar_mls_196000.bin valle_nar_mls_164000.bin --local-dir ckpts
```
Script to download codec model (SpeechTokenizer) checkpoint:
```bash
mkdir -p ckpts/speechtokenizer_hubert_avg && huggingface-cli download amphion/valle SpeechTokenizer.pt config.json --local-dir ckpts/speechtokenizer_hubert_avg
```

If you cannot access huggingface, consider using the huggingface mirror to download: 
```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download amphion/valle valle_ar_mls_196000.bin valle_nar_mls_164000.bin --local-dir ckpts
```
```bash
mkdir -p ckpts/speechtokenizer_hubert_avg && HF_ENDPOINT=https://hf-mirror.com huggingface-cli download amphion/valle SpeechTokenizer.pt config.json --local-dir ckpts/speechtokenizer_hubert_avg
```


### Inference in IPython notebook

We provide our pretrained VALL-E model that is trained on 45k hours MLS dataset, which contains 10-20s English speech.
The "demo.ipynb" file provides a working example of inferencing our pretrained VALL-E model. Give it a try!

## Examining the model files
Examining the model files of VALL-E is a great way to learn how it works.
We provide examples that allows you to overfit a single batch (so no dataset downloading is required). 

The AR model is essentially a causal language model that "continues" a speech. The NAR model is a modification from the AR model that allows for bidirectional attention.


File `valle_ar.py` and `valle_nar.py` in "models/tts/valle_v2" folder are models files, these files can be run directly via `python -m models.tts.valle_v2.valle_ar` (or `python -m models.tts.valle_v2.valle_nar`).
This will invoke a test which overfits it to a single example.

## Training VALL-E from scratch
### Preparing LibriTTS or LibriTTS-R dataset files

We have tested our training script on LibriTTS and LibriTTS-R.
You could download LibriTTS-R at [this link](https://www.openslr.org/141/) and LibriTTS at [this link](https://www.openslr.org/60).
The "train-clean-360" split is currently used by our configuration.
You can test dataset.py by run `python -m models.tts.valle_v2.libritts_dataset`.

For your reference, our unzipped dataset files has a file structure like this:
```
/path/to/LibriTTS_R
├── BOOKS.txt
├── CHAPTERS.txt
├── dev-clean
│   ├── 2412
│   │   ├── 153947
│   │   │   ├── 2412_153947_000014_000000.normalized.txt
│   │   │   ├── 2412_153947_000014_000000.original.txt
│   │   │   ├── 2412_153947_000014_000000.wav
│   │   │   ├── 2412_153947_000017_000001.normalized.txt
│   │   │   ├── 2412_153947_000017_000001.original.txt
│   │   │   ├── 2412_153947_000017_000001.wav
│   │   │   ├── 2412_153947_000017_000005.normalized.txt
├── train-clean-360
    ├── 422
│   │   └── 122949
│   │       ├── 422_122949_000009_000007.normalized.txt
│   │       ├── 422_122949_000009_000007.original.txt
│   │       ├── 422_122949_000009_000007.wav
│   │       ├── 422_122949_000013_000010.normalized.txt
│   │       ├── 422_122949_000013_000010.original.txt
│   │       ├── 422_122949_000013_000010.wav
│   │       ├── 422_122949.book.tsv
│   │       └── 422_122949.trans.tsv
```


Alternativelly, you could write your own dataloader for your dataset. 
You can reference the `__getitem__` method in `models/tts/VALLE_V2/mls_dataset.py`
It should return a dict of a 1-dimensional tensor 'speech', which is a 16kHz speech; and a 1-dimensional tensor of 'phone', which is the phoneme sequence of the speech.
As long as your dataset returns this in `__getitem__`, it should work.

### Changing batch size and dataset path in configuration file
Our configuration file for training VALL-E AR model is at "egs/tts/VALLE_V2/exp_ar_libritts.json", and NAR model at "egs/tts/VALLE_V2/exp_nar_libritts.json"

To train your model, you need to modify the `dataset` variable in the json configurations.
Currently it's at line 40, you should modify the "data_dir" to your dataset's root directory.
```
    "dataset": {
      "dataset_list":["train-clean-360"], // You can also change to other splits like "dev-clean"
      "data_dir": "/path/to/your/LibriTTS_R",
    },
```

You should also select a reasonable batch size at the "batch_size" entry (currently it's set at 5).


You can change other experiment settings in the `/egs/tts/VALLE_V2/exp_ar_libritts.json` such as the learning rate, optimizer and the dataset.

### Run the command to Train AR model
(Make sure your current directory is at the Amphion root directory).
Run:
```sh
sh egs/tts/VALLE_V2/train_ar_libritts.sh
```
Your initial model checkpoint could be found in places such as `ckpt/VALLE_V2/ar_libritts/checkpoint/epoch-0000_step-0000000_loss-7.397293/pytorch_model.bin`


### Resume from existing checkpoint
Our framework supports resuming from existing checkpoint.

Run:
```sh
sh egs/tts/VALLE_V2/train_ar_libritts.sh --resume
```

### Finetuning based on our AR model
We provide our AR model optimizer, and random_states checkpoints to support finetuning (No need to download these files if you're only inferencing from the pretrained model). First rename the models as "pytorch_model.bin", "optimizer.bin", and "random_states_0.pkl", then you could resume from these checkpoints. [Link to AR optimizer checkpoint](https://huggingface.co/amphion/valle/blob/main/optimizer_valle_ar_mls_196000.bin) and [Link to random_states.pkl](https://huggingface.co/amphion/valle/blob/main/random_states_0.pkl).


### Run the command to Train NAR model
(Make sure your current directory is at the Amphion root directory).
Run:
```sh
sh egs/tts/VALLE_V2/train_nar_libritts.sh
```

### Inference your models
Since our inference script is already given, you can change the paths
from our pretrained model to you newly trained models and do the inference.

## Future plans
- [ ] Support more languages
- [ ] More are coming...
