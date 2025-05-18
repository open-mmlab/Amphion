#!/bin/bash
conda activate ints

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/descriptinc/audiotools
pip install vllm==0.7.2 soundfile descript-audio-codec easydict pyworld librosa ffmpy importlib-resources json5 ruamel_yaml ipywidgets dualcodec
pip install gradio accelerate transformers==4.47.1

# optional:
pip install flash-attn --no-build-isolation
