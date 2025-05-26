# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torchaudio
import torch.nn.functional as F
import os
from easydict import EasyDict as edict
from contextlib import nullcontext
import warnings


def _build_semantic_model(
    dualcodec_path,
    meanvar_fname="w2vbert2_mean_var_stats_emilia.pt",
    semantic_model_path="facebook/w2v-bert-2.0",
    device="cuda",
    **kwargs,
):
    """Build the w2v semantic model and load pretrained weights.
    Inputs:
    - dualcodec_path: str, path to the dualcodec model
    - meanvar_fname: str, filename of the mean and variance statistics
    - semantic_model_path: str, path to the semantic model, or a huggngface model name
    Outputs:
    cfg: edict, containing the semantic model, mean, std, and feature extractor.
    - model: Wav2Vec2BertModel instance for semantic feature extraction
    - layer_idx: int (15), index of the layer to extract features from
    - output_idx: int (17), index of the output layer (layer_idx + 2)
    - mean: torch.Tensor containing precomputed mean for feature normalization
    - std: torch.Tensor containing precomputed standard deviation for normalization
    - feature_extractor: SeamlessM4TFeatureExtractor for audio preprocessing
    """
    from transformers import Wav2Vec2BertModel

    if not torch.cuda.is_available():
        warnings.warn("CUDA is not available, running on CPU.")
        device = "cpu"

    # load semantic model
    semantic_model = Wav2Vec2BertModel.from_pretrained(semantic_model_path)
    semantic_model = semantic_model.eval().to(device)

    # load feature extractor
    from transformers import SeamlessM4TFeatureExtractor

    w2v_feat_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
        semantic_model_path
    )

    layer_idx = 15
    output_idx = layer_idx + 2

    # load mean and std
    meanvar_path = os.path.join(dualcodec_path, meanvar_fname)
    stat_mean_var = torch.load(meanvar_path)
    semantic_mean = stat_mean_var["mean"]
    semantic_std = torch.sqrt(stat_mean_var["var"])
    semantic_mean = semantic_mean
    semantic_std = semantic_std

    return edict(
        {
            "semantic_model": semantic_model,
            "layer_idx": layer_idx,
            "output_idx": output_idx,
            "mean": semantic_mean,
            "std": semantic_std,
            "feature_extractor": w2v_feat_extractor,
        }
    )


from cached_path import cached_path


class Inference:
    """
    Inference class for DualCodec.
    """

    def __init__(
        self,
        dualcodec_model,
        dualcodec_path="hf://amphion/dualcodec",
        w2v_path="hf://facebook/w2v-bert-2.0",
        device="cuda",
        autocast=True,
        **kwargs,
    ) -> None:
        """
        Inputs:
        - dualcodec_model: DualCodec instance, the model weight is loaded by safetensors
        - dualcodec_path: str, path to the dualcodec model
        - w2v_path: str, path to the w2v-bert model
        - device: str, device to run the model
        - autocast: bool, whether to use autocast to fp16 for model inference
        """
        dualcodec_path = cached_path(dualcodec_path)
        w2v_path = cached_path(w2v_path)

        if not torch.cuda.is_available():
            warnings.warn("CUDA is not available, running on CPU.")
            device = "cpu"

        self.semantic_cfg = _build_semantic_model(
            dualcodec_path=dualcodec_path,
            semantic_model_path=w2v_path,
            device=device,
            **kwargs,
        )

        self.model = dualcodec_model

        self.model.to(device)
        self.model.eval()

        for key in self.semantic_cfg:
            if isinstance(self.semantic_cfg[key], torch.nn.Module) or isinstance(
                self.semantic_cfg[key], torch.Tensor
            ):
                self.semantic_cfg[key] = self.semantic_cfg[key].to(device)
        self.device = device
        self.autocast = autocast

    @torch.no_grad()
    def encode(
        self,
        audio,
        n_quantizers=8,
    ):
        """
        Args:
        - audio: torch.Tensor, shape=(B, 1, T), dtype=torch.float32, input audio waveform
        - n_quantizers: int, number of RVQ quantizers to use
        Returns:
        - semantic_codes: torch.Tensor, shape=(B, 1, T), dtype=torch.int, semantic codes
        - acoustic_codes: torch.Tensor, shape=(B, num_vq-1, T), dtype=torch.int, acoustic codes
        """
        audio_16k = torchaudio.functional.resample(audio, 24000, 16000)

        feature_extractor = self.semantic_cfg.feature_extractor
        inputs = feature_extractor(
            audio_16k.cpu(), sampling_rate=16000, return_tensors="pt"
        )
        input_features = inputs["input_features"][0]
        attention_mask = inputs["attention_mask"][0]

        input_features = input_features.unsqueeze(0).to(self.device)
        attention_mask = attention_mask.unsqueeze(0).to(self.device)
        audio = audio.to(self.device)

        # by default, we use autocast for semantic feature extraction
        with torch.autocast(device_type=self.device, dtype=torch.float16):
            feat = self._extract_semantic_code(
                input_features, attention_mask
            ).transpose(1, 2)

            feat = torch.nn.functional.avg_pool1d(
                feat,
                self.model.semantic_downsample_factor,
                self.model.semantic_downsample_factor,
            )

        if self.autocast:
            ctx = torch.autocast(device_type=self.device, dtype=torch.float16)
        else:
            ctx = nullcontext()

        with ctx:
            semantic_codes, acoustic_codes = self.model.encode(
                audio, num_quantizers=n_quantizers, semantic_repr=feat
            )

        return semantic_codes, acoustic_codes

    def decode_from_codes(
        self,
        semantic_codes,
        acoustic_codes,
    ):
        """
        Args:
        - semantic_codes: torch.Tensor, shape=(B, 1, T), dtype=torch.int, semantic codes
        - acoustic_codes: torch.Tensor, shape=(B, num_vq-1, T), dtype=torch.int, acoustic codes
        Returns:
        - audio: torch.Tensor, shape=(B, 1, T), dtype=torch.float32, output audio waveform
        """
        audio = self.model.decode_from_codes(semantic_codes, acoustic_codes).to(
            torch.float32
        )
        return audio

    @torch.no_grad()
    def decode(self, semantic_codes, acoustic_codes):
        """
        Args:
        - semantic_codes: torch.Tensor, shape=(B, 1, T), dtype=torch.int, semantic codes
        - acoustic_codes: torch.Tensor, shape=(B, num_vq-1, T), dtype=torch.int, acoustic codes
        Returns:
        - audio: torch.Tensor, shape=(B, 1, T), dtype=torch.float32, output audio waveform
        """
        audio = self.model.decode_from_codes(semantic_codes, acoustic_codes).to(
            torch.float32
        )
        return audio

    @torch.no_grad()
    def _extract_semantic_code(self, input_features, attention_mask):
        """
        Extract semantic code from the input features.
        Args:
        - input_features: torch.Tensor, shape=(B, T, C), dtype=torch.float32, input features
        - attention_mask: torch.Tensor, shape=(B, T), dtype=torch.int, attention mask
        Returns:
        - feat: torch.Tensor, shape=(B, T, C), dtype=torch.float32, extracted semantic code
        """
        vq_emb = self.semantic_cfg["semantic_model"](
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[self.semantic_cfg["output_idx"]]  # (B, T, C)

        if (
            hasattr(self.semantic_cfg, "skip_semantic_normalize")
            and self.semantic_cfg.skip_semantic_normalize
        ):  # skip normalization
            pass
        else:
            feat = (feat - self.semantic_cfg["mean"]) / self.semantic_cfg["std"]
        return feat


@torch.no_grad()
def infer(audio, model=None, num_quantizers=8):
    audio = audio.reshape(1, 1, -1).cpu()
    out, codes = model.inference(audio, n_quantizers=num_quantizers)
    out = pad_to_length(out, audio.shape[-1])
    return out, codes


def pad_to_length(x, length, pad_value=0):
    # Get the current size along the last dimension
    current_length = x.shape[-1]

    # If the length is greater than current_length, we need to pad
    if length > current_length:
        pad_amount = length - current_length
        # Pad on the last dimension (right side), keeping all other dimensions the same
        x_padded = F.pad(x, (0, pad_amount), value=pad_value)
    else:
        # If no padding is required, simply slice the tensor
        x_padded = x[..., :length]

    return x_padded
