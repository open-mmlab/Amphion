import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import os
import math
import json
import accelerate

from models.codec.kmeans.kmeans_model import KMeans, KMeansEMA
from models.codec.kmeans.repcodec_model import RepCodec
from models.codec.amphion_codec.codec import CodecEncoder, CodecDecoder
from transformers import Wav2Vec2BertModel
import safetensors
from utils.util import load_config
from tqdm import tqdm

from transformers import SeamlessM4TFeatureExtractor
processor = SeamlessM4TFeatureExtractor.from_pretrained("")
