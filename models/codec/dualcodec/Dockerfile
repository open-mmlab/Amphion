FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-devel 

ARG VENV_NAME="tts"
ENV VENV=$VENV_NAME
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV DEBIAN_FRONTEN=noninteractive
ENV PYTHONUNBUFFERED=1
SHELL ["/bin/bash", "--login", "-c"]

WORKDIR /workspace

ENV PYTHONPATH="${PYTHONPATH}"

RUN pip install dualcodec[tts]

# RUN conda activate ${VENV} && conda install -y -c conda-forge pynini==2.1.5
WORKDIR /workspace