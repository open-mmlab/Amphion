# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
import os


# %%
import hydra
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
import torchaudio
import torchaudio.transforms as T
from einops import rearrange
import random
import numpy as np
import torch

from loguru import logger

# get dualcodec-valle ar model
with initialize(version_base="1.3", config_path="./conf_tts/model/valle_ar"):
    cfg = compose(config_name="llama_250M")

llama = hydra.utils.instantiate(cfg.model)
logger.debug(llama)

random.seed(0)
np.random.seed(0)
torch.random.manual_seed(0)


# %%
device = "cuda" if torch.cuda.is_available() else "cpu"

from dualcodec.tts_model.valle_ar.valle_ar_inference import ValleARInference

ar_inference = ValleARInference(
    model=llama,
    # TODO ckpt_path=
)

# %%
sys.path.append("zero_shot_tts_training")

# %%
cfg.inference

# %%
t2s_trainer = hydra.utils.instantiate(cfg.inference)
tokenizer = hydra.utils.instantiate(cfg.cfg.get_tokenizer)()

# %%
t2s_trainer.model

# %%
with initialize(version_base="1.3", config_path="../../conf/"):
    cfg = compose(
        config_name="valle_nar_infer_12hz_fast_with_text", overrides=[]
    )  # swtich to text prompted mode: valle_nar_infer_12hz_fast(_with_text)
    print(cfg)

# %%
nar_inference = hydra.utils.instantiate(cfg.inference)

# %%
nar_inference

# %%
import torch
import librosa
import random

# %%
# Directory where the example prompts are stored
data_dir = "/ssd2/lijiaqi18/baidu/simeji-nlp/zero-shot-tts-training/example_prompts"


# Collect all files with matching .wav and .txt filenames
def get_file_pairs(directory):
    # List all .wav and .txt files
    wav_files = [f for f in os.listdir(directory) if f.endswith(".wav")]
    txt_files = [f for f in os.listdir(directory) if f.endswith(".txt")]

    # Create a dictionary mapping base filenames to file pairs
    file_pairs = {}
    for wav_file in wav_files:
        base_name = os.path.splitext(wav_file)[0]
        txt_file = base_name + ".txt"
        if txt_file in txt_files:
            file_pairs[base_name] = (
                os.path.join(directory, wav_file),
                os.path.join(directory, txt_file),
            )

    return file_pairs


# Function to get text from a .txt file
def get_txt_from_file(file_path):
    with open(file_path, "r") as f:
        return f.read().strip()


def generate_random_data():
    # Get all file pairs from the directory
    file_pairs = get_file_pairs(data_dir)

    if not file_pairs:
        raise ValueError("No matching .wav and .txt files found in the directory.")

    # Randomly select a file pair
    random_key = random.choice(list(file_pairs.keys()))
    wav_file, txt_file = file_pairs[random_key]

    # Load the .wav file
    # speech = librosa.load(wav_file, sr=16000)[0]

    # Load the text data
    txt = get_txt_from_file(txt_file)

    return wav_file, txt


def text_preprocess(prompt_text, prompt_language, target_text, prompt_len_tmp=0):
    """
    split target text and normalize
    ["prompt text | target text 1", "",...]
    """
    prompt_text = prompt_text.strip()
    target_text = target_text.replace("\n", "")
    target_text = target_text.replace("\t", "")
    prompt_len_tmp = len(tokenizer.encode(prompt_text)) // 2
    if prompt_language == "zh":
        from cosyvoice.utils.frontend_utils import split_paragraph

        texts = split_paragraph(
            target_text,
            None,
            "zh",
            token_max_n=60 - prompt_len_tmp,
            token_min_n=40 - prompt_len_tmp,
            merge_len=20,
            comma_split=False,
        )
    elif prompt_language == "ja":
        from cosyvoice.utils.frontend_utils import split_paragraph

        texts = split_paragraph(
            target_text,
            None,
            "zh",
            token_max_n=70,
            token_min_n=60,
            merge_len=20,
            comma_split=False,
        )
    elif prompt_language == "en":
        from cosyvoice.utils.frontend_utils import split_paragraph

        texts = split_paragraph(
            target_text,
            tokenizer.encode,
            "en",
            token_max_n=50 - prompt_len_tmp,
            token_min_n=30 - prompt_len_tmp,
            merge_len=20,
            comma_split=True,
        )
    else:
        texts = [target_text]
    if prompt_language == "en":
        texts = [prompt_text + " " + t for t in texts]
    else:
        texts = [prompt_text + t for t in texts]
    print(texts)

    for i in range(len(texts)):
        from cosyvoice.dataset.processor import normalize

        texts[i] = list(
            normalize(
                [
                    {
                        "language": prompt_language,
                        "text": texts[i],
                    }
                ],
                en_punct=True,
                use_kana=True,
            )
        )[0]["text"]
    print(texts)
    return texts


# %%
import logging
import numpy as np

logger = logging.getLogger(__name__)


def preprocess_audio_torch(
    waveform, original_sample_rate, target_sample_rate=16000, apply_loudness_norm=False
):
    # Resample the waveform to the target sample rate (16kHz)
    resample = T.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
    waveform = resample(waveform)

    # Ensure the audio is mono (downmix stereo if needed)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Convert waveform to NumPy array for further processing
    waveform_np = waveform.numpy().flatten().astype(np.float32)

    if apply_loudness_norm:
        # Calculate dBFS (decibel relative to full scale) of the audio
        rms = np.sqrt(np.mean(waveform_np**2))
        dBFS = 20 * np.log10(rms) if rms > 0 else -float("inf")

        # Calculate the gain to be applied (target dBFS is -20)
        target_dBFS = -20
        gain = target_dBFS - dBFS
        logger.info(f"Calculating the gain needed for the audio: {gain} dB")

        # Apply gain, limiting it between -3 and 3 dB
        gain = min(max(gain, -3), 3)
        waveform_np = waveform_np * (10 ** (gain / 20))

        # Normalize waveform (max absolute amplitude should be 1.0)
        max_amplitude = np.max(np.abs(waveform_np))
        if max_amplitude > 0:
            waveform_np /= max_amplitude

    logger.debug(f"waveform shape: {waveform_np.shape}")
    logger.debug(f"waveform dtype: {waveform_np.dtype}")

    return waveform_np, target_sample_rate


# %%
import time
import soundfile as sf

from tools.whisper_usage import get_prompt_text


@torch.no_grad()
def infer(
    speech_path,
    prompt_text,
    target_text,
    target_language="zh",
    temperature=1.0,
    top_k=20,
    top_p=0.9,
    repeat_penalty=1.0,
    concat_prompt=False,
):
    # Load the audio file (original sample rate is detected automatically)
    waveform, original_sample_rate = torchaudio.load(speech_path)

    if prompt_text is None or prompt_text == "":
        prompt_text_detected, prompt_lang_detected, _, _ = get_prompt_text(speech_path)
        prompt_text = prompt_text_detected
    if target_language == "auto":
        try:
            target_language = prompt_lang_detected
        except Exception:
            raise ValueError(
                "Please specified target language as well, if you specified reference text."
            )
    # Preprocess audio for 16 kHz
    speech_16k, sample_rate_16k = preprocess_audio_torch(
        waveform, original_sample_rate, target_sample_rate=16000
    )

    # Preprocess audio for 24 kHz (if needed)
    speech, sample_rate_24k = preprocess_audio_torch(
        waveform, original_sample_rate, target_sample_rate=24000
    )

    # Convert the resampled waveforms to NumPy arrays
    # prompt speech
    speech_16k = speech_16k.flatten()
    speech = speech.flatten()
    # breakpoint()
    with torch.cuda.amp.autocast():
        time1 = time.time()
        # combine_semantic_code, prompt_semantic_code = text2semantic(speech_16k, prompt_text,  target_language, target_text, target_language, temp=temperature, top_k=top_k, top_p=top_p, repeat_penalty=repeat_penalty)
        texts = text_preprocess(prompt_text, target_language, target_text)
        for text in texts:
            predict_semantic, prompt_semantic_code, input_features, attention_mask = (
                t2s_trainer.inference(
                    text,
                    speech_16k,
                    target_language,
                    temp=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repeat_penalty=repeat_penalty,
                )
            )
            time2 = time.time()
            semantic_codes, acoustic_codes = nar_inference._extract_codes_dac(
                (torch.tensor(speech).unsqueeze(0).to(device)),
                input_features,
                attention_mask,
            )
            time3 = time.time()

            semantic_codes = rearrange(semantic_codes, "b t -> b t 1")
            acoustic_codes = torch.cat([semantic_codes, acoustic_codes], dim=-1)

            # print(acoustic_code.shape)32

            recovered_audio = nar_inference.inference(
                predict_semantic, acoustic_codes, text, target_language
            )  # if not with_text 'text' 'target_language' will not be used in the inference function
            # breakpoint()
            recovered_audio = recovered_audio.cpu().reshape(1, -1).to(torch.float32)
            time4 = time.time()
            print(
                f"text2semantic: {time2-time1}, extract_acoustic_code: {time3-time2}, semantic2acoustic: {time4-time3}"
            )
            print(f"All time cost: {time4-time1:.2f} seconds")
            save_path = f"/tmp/{int(time.time())}.wav"
            torchaudio.save(save_path, recovered_audio, 24000)
            yield save_path, prompt_text
    yield None, None


# %%
import gradio as gr

with gr.Blocks() as app:
    gr.Markdown("## TTS Demo")

    with gr.Tabs():
        with gr.TabItem("Fast_Valle_Streaming"):
            with gr.Column():
                # 添加一个状态变量
                audio_state = gr.State([])

                # Define input components individually (unwrapped)
                reference_audio = gr.Audio(
                    label="Reference Audio", type="filepath", autoplay=True
                )
                text_to_generate = gr.Textbox(
                    label="Text to Generate", type="text", value=""
                )
                transcription_label = gr.Textbox(
                    label="Transcribed Content",
                    placeholder="The prompt transcription will appear here.",
                    interactive=False,
                )

                # Advanced settings section
                with gr.Accordion("Advanced settings", open=False):
                    # Add interactive components within the accordion
                    generated_language_dropdown = gr.Dropdown(
                        label="Generated Language",
                        choices=["auto", "zh", "en", "ja"],
                        value="auto",
                    )
                    reference_text = gr.Textbox(
                        label="(Optional) Reference Text", type="text", placeholder=""
                    )
                    temperature_slider = gr.Slider(
                        label="temperature", value=0.9, minimum=0, maximum=5
                    )
                    top_k_slider = gr.Slider(
                        label="top_k", value=10, minimum=1, maximum=150
                    )
                    top_p_slider = gr.Slider(
                        label="top_p", value=0.9, minimum=0.0, maximum=1.0
                    )
                    repeat_penalty_slider = gr.Slider(
                        label="repeat_penalty", value=1.0, minimum=0.0, maximum=3.0
                    )

                    # Button to generate random prompt
                    generate_input_btn = gr.Button("Generate a random prompt")
                    generate_input_btn.click(
                        fn=generate_random_data,
                        inputs=[],
                        outputs=[reference_audio, reference_text],
                    )

                # Output components
                demo_outputs = gr.Audio(
                    label="Generated Audio", streaming=True, autoplay=True
                )

                submit_btn = gr.Button("Submit")

                # 使用单个事件处理器
                submit_btn.click(
                    fn=infer,
                    inputs=[
                        reference_audio,
                        reference_text,
                        text_to_generate,
                        generated_language_dropdown,
                        temperature_slider,
                        top_k_slider,
                        top_p_slider,
                        repeat_penalty_slider,
                    ],
                    outputs=[demo_outputs, transcription_label],
                )

app.launch(server_name="0.0.0.0", server_port=8071, show_error=True)


# if __name__ == "__main__":

#     # 设置示例参数
#     example_speech_path = "/ssd2/lijiaqi18/baidu/simeji-nlp/zero-shot-tts-training/example_prompts/1.wav"
#     prompt_text = "大丈夫！今日は休日だってことをちゃんと伝えれば、みんなも理解してくれると思う。"  # 留空让系统自动检测
#     target_text = "ありがとう、私の暗い世界の小さな太陽、ありがとう、ずっと私を温めてくれました"
#     target_language = "ja"
#     # example_speech_path = "/ssd2/lijiaqi18/baidu/simeji-nlp/zero-shot-tts-training/example_prompts/wk1.wav"
#     # prompt_text = "你们太没信誉了,这都五天了还不发货,我要投诉你。"  # 留空让系统自动检测
#     # target_text = "理工学院下属学生社团物理协会诚挚邀请您参加我们即将举办的首期梧桐讲堂。"
#     # target_language = "zh"

#     # 运行推理
#     while(True):
#         output_path, detected_prompt = infer(
#             speech_path=example_speech_path,
#             prompt_text=prompt_text,
#             target_text=target_text,
#             target_language=target_language,
#             temperature=0.9,
#             top_k=10,
#             top_p=0.9,
#             repeat_penalty=1.0
#         )

#     print(f"推理完成！")
#     print(f"检测到的提示文本: {detected_prompt}")
#     print(f"生成的音频保存在: {output_path}")
