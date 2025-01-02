#!/bin/bash

sudo apt-get update
sudo apt-get install -y espeak-ng

pip install accelerate==0.24.1
pip install cn2an
pip install -U cos-python-sdk-v5
pip install datasets
pip install ffmpeg-python
pip install setuptools ruamel.yaml tqdm 
pip install tensorboard tensorboardX torch==2.3.1
pip install transformers===4.41.1
pip install -U encodec
pip install black==24.1.1
pip install -U funasr
pip install g2p-en
pip install jieba
pip install json5
pip install librosa
pip install matplotlib
pip install modelscope
pip install numba==0.60.0
pip install numpy
pip install omegaconf
pip install onnxruntime
pip install -U openai-whisper
pip install openpyxl
pip install pandas
pip install phonemizer
pip install protobuf
pip install pydub
pip install pypinyin
pip install pyworld
pip install ruamel.yaml
pip install scikit-learn scipy
pip install soundfile
pip install timm tokenizers
pip install torchaudio==2.3.1
pip install torchvision==0.18.1
pip install tqdm==4.66.4
pip install transformers==4.44.0
pip install unidecode
pip install zhconv zhon wandb

