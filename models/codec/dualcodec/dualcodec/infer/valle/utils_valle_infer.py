# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from dualcodec.utils.utils_infer import (
    tqdm,
    device,
    cross_fade_duration,
    torch,
    torchaudio,
    chunk_text,
)
from dualcodec.utils import normalize_text
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from einops import rearrange
import numpy as np

# -----------------------------------------

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"
target_rms = 0.1
cross_fade_duration = 0.15
ode_method = "euler"
nfe_step = 32  # 16, 32
cfg_strength = 2.0
sway_sampling_coef = -1.0
speed = 1.0
fix_duration = None

from dualcodec.utils.utils_infer import instantiate_model, load_checkpoint


# load models
def load_dualcodec_valle_ar_12hzv1():
    TTS_MODEL_CFG = {
        "model": "valle_ar",
        "ckpt_path": "hf://amphion/dualcodec-tts/dualcodec_valle_ar_12hzv1.safetensors",
        # "ckpt_path": "dualcodec_tts_ckpts/dualcodec_valle_ar_12hzv1.safetensors",
        "cfg_path": "../conf_tts/model/valle_ar/llama_250M.yaml",
    }
    model = (
        instantiate_model(
            model_cfg_path=TTS_MODEL_CFG["cfg_path"],
        )
        .half()
        .eval()
    )
    ckpt_path = TTS_MODEL_CFG["ckpt_path"]
    load_checkpoint(
        model,
        ckpt_path,
        use_ema=False,
        device=device,
    )
    return model


def load_dualcodec_valle_nar_12hzv1():
    TTS_MODEL_CFG = {
        "model": "valle_nar",
        "ckpt_path": "hf://amphion/dualcodec-tts/dualcodec_valle_nar_12hzv1.safetensors",
        "cfg_path": "../conf_tts/model/valle_nar/valle_nar.yaml",
    }
    model = (
        instantiate_model(
            model_cfg_path=TTS_MODEL_CFG["cfg_path"],
        )
        .half()
        .eval()
    )
    ckpt_path = TTS_MODEL_CFG["ckpt_path"]
    load_checkpoint(
        model,
        ckpt_path,
        use_ema=False,
        device=device,
    )
    return model


def infer_process(
    ref_audio,
    ref_text,
    gen_text,
    ar_model_obj,
    nar_model_obj,
    dualcodec_inference_obj,
    tokenizer_obj,
    show_info=print,
    progress=tqdm,
    cross_fade_duration=cross_fade_duration,
    target_rms=target_rms,
    lang="en",
    device=device,
    streaming=False,
    top_k=15,
    top_p=0.85,
    temperature=1.0,
    repeat_penalty=1.1,
):
    # Split the input text into batches
    audio, sr = torchaudio.load(ref_audio)
    max_chars = int(
        len(ref_text.encode("utf-8"))
        / (audio.shape[-1] / sr)
        * (22 - audio.shape[-1] / sr)
    )
    gen_text_batches = chunk_text(gen_text, max_chars=max_chars)
    for i, gen_text in enumerate(gen_text_batches):
        print(f"gen_text {i}", gen_text)
    print("\n")

    show_info(f"Generating audio in {len(gen_text_batches)} batches...")
    return next(
        infer_batch_process(
            (audio, sr),
            ref_text,
            gen_text_batches,
            ar_model_obj,
            nar_model_obj,
            dualcodec_inference_obj=dualcodec_inference_obj,
            tokenizer_obj=tokenizer_obj,
            progress=progress,
            cross_fade_duration=cross_fade_duration,
            target_rms=target_rms,
            device=device,
            lang=lang,
            streaming=streaming,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repeat_penalty=repeat_penalty,
        )
    )


# infer batches


def infer_batch_process(
    ref_audio,
    ref_text,
    gen_text_batches,
    ar_model_obj,
    nar_model_obj,
    dualcodec_inference_obj,
    tokenizer_obj,
    target_rms=0.1,
    progress=tqdm,
    cross_fade_duration=0.15,
    device="cuda",
    streaming=False,
    chunk_size=2048,
    lang="en",
    top_k=15,
    top_p=0.85,
    temperature=1.0,
    repeat_penalty=1.1,
):
    audio, sr = ref_audio
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
    audio = audio.to(device)

    generated_waves = []
    spectrograms = []

    if ref_text[-1] != " ":
        ref_text = ref_text + " "

    def process_batch(gen_text):
        local_speed = speed
        if len(gen_text.encode("utf-8")) < 10:
            local_speed = 0.3

        # Prepare the text
        text_list = [ref_text + gen_text]
        final_text = normalize_text(text_list, lang=lang)
        logger.debug(f"final_text: {final_text}")
        # final_text_list = convert_char_to_pinyin(text_list)

        # inference
        with torch.autocast(device_type=device, dtype=torch.float16):
            with torch.inference_mode():
                (
                    generated,
                    prompt_semantic_code,
                    prompt_acoustic_code,
                    prompt_text_tokens,
                ) = valle_ar_inference(
                    ar_model_obj=ar_model_obj,
                    dualcodec_inference_obj=dualcodec_inference_obj,
                    tokenizer=tokenizer_obj,
                    text=final_text,
                    prompt_speech=audio,
                    prompt_language=lang,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    repeat_penalty=repeat_penalty,
                    return_prompt=True,
                    device=device,
                )

                # nar sampling
                combine_semantic_code = torch.cat(
                    [prompt_semantic_code.squeeze(1), generated], dim=-1
                )
                generated = valle_nar_inference(
                    nar_model_obj=nar_model_obj,
                    dualcodec_inference_obj=dualcodec_inference_obj,
                    combine_semantic_code=combine_semantic_code,
                    prompt_acoustic_code=prompt_acoustic_code,
                    prompt_text_tokens=None,
                    use_text_prompt=True,
                    prompt_language=lang,
                    device=device,
                )  # [1,1,T]

                generated = generated.to(torch.float32)  # generated mel spectrogram

                # wav -> numpy
                generated_wave = generated.squeeze().squeeze().cpu()

                # get spectrogram
                generated_cpu = torchaudio.transforms.MelSpectrogram(
                    sample_rate=target_sample_rate,
                    n_fft=n_fft,
                    win_length=win_length,
                    hop_length=hop_length,
                    n_mels=n_mel_channels,
                    f_min=0,
                    f_max=target_sample_rate // 2,
                    pad=0,
                    power=1,
                    norm="slaney",
                )(
                    generated_wave
                )  # [H,T]

                generated_wave = generated_wave.numpy()

                if streaming:
                    for j in range(0, len(generated_wave), chunk_size):
                        yield generated_wave[j : j + chunk_size], target_sample_rate
                else:
                    generated_cpu = generated.cpu().numpy()
                    del generated
                    yield generated_wave, generated_cpu

    if streaming:
        for gen_text in (
            progress.tqdm(gen_text_batches)
            if progress is not None
            else gen_text_batches
        ):
            for chunk in process_batch(gen_text):
                yield chunk
    else:
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_batch, gen_text)
                for gen_text in gen_text_batches
            ]
            for future in progress.tqdm(futures) if progress is not None else futures:
                result = future.result()
                if result:
                    generated_wave, generated_mel_spec = next(result)
                    generated_waves.append(generated_wave)
                    spectrograms.append(generated_mel_spec)

        if generated_waves:
            if cross_fade_duration <= 0:
                # Simply concatenate
                final_wave = np.concatenate(generated_waves)
            else:
                # Combine all generated waves with cross-fading
                final_wave = generated_waves[0]
                for i in range(1, len(generated_waves)):
                    prev_wave = final_wave
                    next_wave = generated_waves[i]

                    # Calculate cross-fade samples, ensuring it does not exceed wave lengths
                    cross_fade_samples = int(cross_fade_duration * target_sample_rate)
                    cross_fade_samples = min(
                        cross_fade_samples, len(prev_wave), len(next_wave)
                    )

                    if cross_fade_samples <= 0:
                        # No overlap possible, concatenate
                        final_wave = np.concatenate([prev_wave, next_wave])
                        continue

                    # Overlapping parts
                    prev_overlap = prev_wave[-cross_fade_samples:]
                    next_overlap = next_wave[:cross_fade_samples]

                    # Fade out and fade in
                    fade_out = np.linspace(1, 0, cross_fade_samples)
                    fade_in = np.linspace(0, 1, cross_fade_samples)

                    # Cross-faded overlap
                    cross_faded_overlap = (
                        prev_overlap * fade_out + next_overlap * fade_in
                    )

                    # Combine
                    new_wave = np.concatenate(
                        [
                            prev_wave[:-cross_fade_samples],
                            cross_faded_overlap,
                            next_wave[cross_fade_samples:],
                        ]
                    )

                    final_wave = new_wave

            # Create a combined spectrogram
            combined_spectrogram = np.concatenate(spectrograms, axis=1)

            yield final_wave, target_sample_rate, combined_spectrogram

        else:
            yield None, target_sample_rate, None


@torch.inference_mode()
def valle_ar_inference(
    ar_model_obj,
    dualcodec_inference_obj,
    tokenizer,
    text,
    prompt_speech,
    prompt_language,
    temperature=1.0,
    top_k=1000,
    top_p=0.85,
    repeat_penalty=1.1,
    return_prompt=False,
    device="cuda",
):
    """
    Generate text given speech and text prompts.
    Args:
        ar_model_obj: The autoregressive model object.
        tokenizer: The whisper tokenizer object.
        text: The text prompt.
        prompt_speech: The speech prompt.
        prompt_language: The language of the prompt.
        temp: Temperature for sampling.
        top_k: Top-k sampling parameter.
        top_p: Top-p sampling parameter.
        repeat_penalty: Penalty for repeated tokens.
        return_prompt: Whether to return the semantic and acoustic prompt code.
    """
    prompt_text_tokens = torch.tensor(
        [[tokenizer.to_language_token(prompt_language)] + tokenizer.encode(text)],
        dtype=torch.int32,
        device=device,
    )  # [B, T]
    prompt_text_len = torch.tensor([prompt_text_tokens.shape[-1]], device=device)

    # prompt semantic codes
    prompt_semantic_code, prompt_acoustic_code = dualcodec_inference_obj.encode(
        prompt_speech.reshape(1, 1, -1),
    )
    # semantic_codes shape: torch.Size([1, 1, T])
    # acoustic_codes shape: torch.Size([1, n_quantizers-1, T])

    out = ar_model_obj.inference(
        text=prompt_text_tokens.clone(),
        text_len=prompt_text_len,
        prompt_text=None,
        prompt_text_len=None,
        prompt_speech_token=prompt_semantic_code.squeeze(0),
        prompt_speech_token_len=torch.tensor([prompt_semantic_code.shape[-1]]),
        top_k=top_k,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        temperature=temperature,
    )
    if return_prompt:
        # out: [B, T], prompt_semantic_code: [1, 1, T], prompt_acoustic_code: [1, n_quantizers-1, T]
        return out, prompt_semantic_code, prompt_acoustic_code, prompt_text_tokens
    else:
        return out


@torch.inference_mode()
def valle_nar_inference(
    nar_model_obj,
    dualcodec_inference_obj,
    combine_semantic_code,  # shape [b t]
    prompt_acoustic_code,  # shape [1, q, t]
    prompt_text_tokens=None,
    use_text_prompt=True,
    prompt_language="en",
    device="cuda",
    use_prompt_text=False,
):
    if prompt_text_tokens is not None:
        raise NotImplementedError("prompt_text_tokens is not None")
        # prompt_text_mask = torch.ones(1, prompt_text_tokens.shape[-1], device=device, dtype=torch.bool)
    else:
        B = 1
        prompt_text_tokens = torch.zeros(B, 1, dtype=torch.long).to(device)
        prompt_text_mask = torch.zeros(B, 1, dtype=torch.bool).to(device)
    prompt_acoustic_code = rearrange(prompt_acoustic_code, "b q t -> q b t")
    prompt_semantic_code = combine_semantic_code[:, : prompt_acoustic_code.shape[-1]]
    predict_semantic_code = combine_semantic_code[:, prompt_acoustic_code.shape[-1] :]
    prompt_semantic_code = rearrange(prompt_semantic_code, "b t -> 1 b t")
    prompt_acoustic_code = torch.cat(
        [prompt_semantic_code, prompt_acoustic_code], dim=0
    )  # [8,B,T]

    out = nar_model_obj.sample_hf(
        phone_ids=prompt_text_tokens,  # [B, T]
        phone_mask=prompt_text_mask,
        prompt_ids=prompt_acoustic_code,  # [8,B,T]
        first_stage_ids=predict_semantic_code,  # [B,T]
        use_text_prompt=use_text_prompt,
        # target_quantization_layer=1+i%6,
    )

    predict_full = rearrange(out, "q b t -> b q t")  # [8,B,T]

    # combine_semantic_code = rearrange(combine_semantic_code, 'b t -> b 1 t')
    # predict_full = rearrange(predict_full, 'q b t -> b q t')

    combine_audio = dualcodec_inference_obj.decode(
        predict_full[:, :1],
        predict_full[:, 1:],
    )
    return combine_audio  # [1, 1, T]
