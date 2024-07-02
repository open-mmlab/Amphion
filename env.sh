# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Raise error if any command fails
set -e

# Install ffmpeg in Linux
conda install -c conda-forge ffmpeg

# Pip packages
pip install setuptools ruamel.yaml tqdm colorama easydict tabulate loguru json5 Cython unidecode inflect argparse g2p_en tgt librosa==0.9.1 matplotlib typeguard einops omegaconf hydra-core humanfriendly pandas munch

pip install tensorboard tensorboardX torch==2.0.1 torchaudio==2.0.2 torchvision==0.15.2 accelerate==0.24.1 transformers==4.41.2 diffusers praat-parselmouth audiomentations pedalboard ffmpeg-python==0.2.0 pyworld diffsptk==1.0.1 nnAudio unidecode inflect ptwt

pip install encodec vocos speechtokenizer g2p_en descript-audio-codec

pip install torchmetrics pymcd openai-whisper frechet_audio_distance asteroid resemblyzer vector-quantize-pytorch==1.12.5

pip install https://github.com/vBaiCai/python-pesq/archive/master.zip

pip install fairseq

pip install git+https://github.com/lhotse-speech/lhotse

pip install -U encodec

pip install phonemizer==3.2.1 pypinyin==0.48.0

pip install black==24.1.1

# Uninstall nvidia-cublas-cu11 if there exist some bugs about CUDA version
# pip uninstall nvidia-cublas-cu11
