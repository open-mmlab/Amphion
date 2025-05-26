# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
from cached_path import cached_path
from functools import partial
from dualcodec.utils import device, package_dir
import torch.nn.functional as F


def load_voicebox_300M_model(device="cuda"):
    TTS_MODEL_CFG = {
        "model": "voicebox_300M",
        "ckpt_path": "hf://amphion/dualcodec-tts/voicebox_dualcodec12hzv1.safetensors",
    }
    from dualcodec.model_tts.voicebox.voicebox_models import voicebox_300M

    model = voicebox_300M().to(device)
    model.eval()
    # load model
    ckpt_path = TTS_MODEL_CFG["ckpt_path"]
    ckpt_path = cached_path(ckpt_path)
    import safetensors.torch

    safetensors.torch.load_model(model, ckpt_path)
    return model


def load_dualcodec_12hzv1_model(device="cuda"):
    import dualcodec

    dualcodec_model = dualcodec.get_model("12hz_v1")
    dualcodec_inference_obj = dualcodec.Inference(
        dualcodec_model=dualcodec_model, device=device, autocast=True
    )
    return dualcodec_inference_obj


def get_vocoder_decode_func_and_mel_spec(device="cuda"):
    from dualcodec.model_tts.voicebox.vocoder_model import (
        get_vocos_model_spectrogram,
        mel_to_wav_vocos,
    )

    vocos_model, mel_model = get_vocos_model_spectrogram(device=device)
    infer_vocos = partial(mel_to_wav_vocos, vocos_model)
    return infer_vocos, mel_model


@torch.inference_mode()
def voicebox_inference(
    voicebox_model_obj,
    vocoder_decode_func,
    mel_spec_extractor_func,
    combine_semantic_code,  # shape [b t]
    prompt_speech,  # shape [b t]
    device="cuda",
):
    def code2mel(self, combine_semantic_code: torch.Tensor, prompt_speech):
        cond_feature = voicebox_model_obj.cond_emb(combine_semantic_code)
        cond_feature = F.interpolate(
            cond_feature.transpose(1, 2),
            scale_factor=voicebox_model_obj.cond_scale_factor,
        ).transpose(1, 2)

        if prompt_speech is not None:
            prompt_mel_feat = mel_spec_extractor_func(
                torch.tensor(prompt_speech),  # [b t]
                device=device,
            )  # [b c t]
            prompt_mel_feat = prompt_mel_feat.transpose(1, 2)  # [b t c]
        else:
            prompt_mel_feat = None

        predict_mel = voicebox_model_obj.reverse_diffusion(
            cond_feature,
            prompt_mel_feat,
            n_timesteps=32,
            cfg=2.0,
            rescale_cfg=0.75,
        )

        return predict_mel  # [b t c]

    predicted_mel = code2mel(
        voicebox_model_obj,
        combine_semantic_code,
        prompt_speech,
    )  # [b, t, c]

    predicted_audio = vocoder_decode_func(predicted_mel.transpose(1, 2))
    return predicted_audio  # [b,t]


if __name__ == "__main__":
    from dualcodec.model_tts.voicebox.voicebox_models import (
        voicebox_300M,
        extract_normalized_mel_spec_50hz,
    )

    voicebox_model_obj = load_voicebox_300M_model(device=device)

    vocoder_decode_func, mel_model = get_vocoder_decode_func_and_mel_spec(device=device)

    # extract GT dualcodec tokens
    dualcodec_inference_obj = load_dualcodec_12hzv1_model(device=device)
    import torchaudio

    audio, sr = torchaudio.load(
        f"{package_dir}/infer/examples/basic/example_wav_en.wav"
    )
    # resample to 24kHz
    audio = torchaudio.functional.resample(audio, sr, 24000)
    audio = audio.reshape(1, 1, -1)
    audio = audio.to("cuda")
    # extract codes, for example, using 8 quantizers here:
    semantic_codes, acoustic_codes = dualcodec_inference_obj.encode(
        audio, n_quantizers=8
    )
    # semantic_codes shape: torch.Size([1, 1, T])
    # acoustic_codes shape: torch.Size([1, n_quantizers-1, T])

    # change semantic_codes to [b, t]
    semantic_codes = semantic_codes.squeeze(1)

    # use first 3s of acoustic codes as prompt
    audio = audio[:, :, : int(24000 * 2)]

    predicted = voicebox_inference(
        voicebox_model_obj=voicebox_model_obj,
        vocoder_decode_func=vocoder_decode_func,
        mel_spec_extractor_func=extract_normalized_mel_spec_50hz,
        combine_semantic_code=semantic_codes,
        prompt_speech=audio.squeeze(1),  # [b t]
    )

    torchaudio.save("predicted.wav", predicted.cpu(), 24000)
