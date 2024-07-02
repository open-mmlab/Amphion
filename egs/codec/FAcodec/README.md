# FAcodec

Pytorch implementation for the training of FAcodec, which was proposed in paper [NaturalSpeech 3: Zero-Shot Speech Synthesis
with Factorized Codec and Diffusion Models](https://arxiv.org/pdf/2403.03100)  

A dedicated repository for the FAcodec model can also be find [here](https://github.com/Plachtaa/FAcodec).

This implementation made some key improvements to the training pipeline, so that the requirements of any form of annotations, including 
transcripts, phoneme alignments, and speaker labels, are eliminated. All you need are simply raw speech files.  
With the new training pipeline, it is possible to train the model on more languages with more diverse timbre distributions.  
We release the code for training and inference, including a pretrained checkpoint on 50k hours speech data with over 1 million speakers.

## Model storage
We provide pretrained checkpoints on 50k hours speech data.  

| Model type        | Link                                                                                                                                   |
|-------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| FAcodec           | [![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-FAcodec-blue)](https://huggingface.co/Plachta/FAcodec)               |

## Demo
Try our model on [![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/Plachta/FAcodecV2)!

## Training
Prepare your data and put them under one folder, internal file structure does not matter.  
Then, change the `dataset` in `./egs/codec/FAcodec/exp_custom_data.json` to the path of your data folder.  
Finally, run the following command:
```bash
sh ./egs/codec/FAcodec/train.sh
```

## Inference
To reconstruct a speech file, run:
```bash
python ./bins/codec/inference.py --source <source_wav> --output_dir <output_dir> --checkpoint_path <checkpoint_path>
```
To use zero-shot voice conversion, run:
```bash
python ./bins/codec/inference.py --source <source_wav> --reference <reference_wav> --output_dir <output_dir> --checkpoint_path <checkpoint_path>
```

## Feature extraction
When running `./bins/codec/inference.py`, check the returned results of the `FAcodecInference` class: a tuple of `(quantized, codes)`
- `quantized` is the quantized representation of the input speech file.
- `quantized[0]` is the quantized representation of prosody
- `quantized[1]` is the quantized representation of content

- `codes` is the discrete code representation of the input speech file.
- `codes[0]` is the discrete code representation of prosody
- `codes[1]` is the discrete code representation of content

For the most clean content representation without any timbre, we suggest to use `codes[1][:, 0, :]`, which is the first layer of content codebooks.