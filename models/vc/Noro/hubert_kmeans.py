from pathlib import Path
import torch
from torch import nn
from torchaudio.functional import resample
from einops import rearrange, repeat
import torch.nn.functional as F

def noop(*args, **kwargs):
    pass


import warnings
import logging

logging.root.setLevel(logging.ERROR)

warnings.warn = noop

import joblib
import fairseq

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def round_down_nearest_multiple(num, divisor):
    return num // divisor * divisor


def curtail_to_multiple(t, mult, from_left=False):
    data_len = t.shape[-1]
    rounded_seq_len = round_down_nearest_multiple(data_len, mult)
    seq_slice = (
        slice(None, rounded_seq_len) if not from_left else slice(-rounded_seq_len, None)
    )
    return t[..., seq_slice]

# TODO: modify according to xueyao's comments

class HubertWithKmeans(nn.Module):
    """
    checkpoint and kmeans can be downloaded at https://github.com/facebookresearch/fairseq/tree/main/examples/hubert
    or you can train your own
    """

    def __init__(
        self,
        checkpoint_path= 'path/to/your/mhubert_model/mhubert_base_vp_en_es_fr_it3.pt',
        kmeans_path = 'path/to/your/mhubert/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin',
        target_sample_hz=16000,
        seq_len_multiple_of=None,
        output_layer=9,
    ):
        super().__init__()

        self.target_sample_hz = target_sample_hz
        self.seq_len_multiple_of = seq_len_multiple_of
        self.output_layer = output_layer

        model_path = Path(checkpoint_path)
        kmeans_path = Path(kmeans_path)

        assert model_path.exists(), f"path {checkpoint_path} does not exist"
        assert kmeans_path.exists(), f"path {kmeans_path} does not exist"

        checkpoint = torch.load(checkpoint_path)
        load_model_input = {checkpoint_path: checkpoint}
        model, *_ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            load_model_input
        )

        self.model = model[0]
        self.model.eval()

        kmeans = joblib.load(kmeans_path)

        self.kmeans = kmeans

        self.register_buffer(
            "cluster_centers", torch.from_numpy(kmeans.cluster_centers_)
        )

    @property
    def groups(self):
        return 1

    @property
    def codebook_size(self):
        return self.kmeans.n_clusters

    @property
    def downsample_factor(self):
        return 320

    @torch.inference_mode()
    def forward(self, wav_input, flatten=True, input_sample_hz=None):
        batch, device = wav_input.shape[0], wav_input.device
        wav_input = F.pad(wav_input, (40, 40), "reflect")

        if exists(input_sample_hz):
            wav_input = resample(wav_input, input_sample_hz, self.target_sample_hz)

        if exists(self.seq_len_multiple_of):
            wav_input = curtail_to_multiple(wav_input, self.seq_len_multiple_of)

        embed = self.model(
            wav_input,
            features_only=True,
            mask=False,  # thanks to @maitycyrus for noticing that mask is defaulted to True in the fairseq code
        )
        
        batched_cluster_centers = repeat(
            self.cluster_centers, "c d -> b c d", b=embed.shape[0]
        )
        
        dists = -torch.cdist(embed, batched_cluster_centers, p=2)
        clusters = dists.argmax(dim=-1) # (batch, seq_len)
        quantize = F.embedding(clusters, self.cluster_centers)

        if flatten:
            return clusters, quantize

        return rearrange(clusters, "b ... -> b (...)"), quantize