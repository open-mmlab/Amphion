import gradio as gr
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
import tempfile
from loguru import logger
import torchaudio

from dualcodec.utils.utils_infer import (
    device,
    cross_fade_duration,
    target_rms,
    nfe_step,
    speed,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)
from dualcodec.infer.valle.utils_valle_infer import (
    load_dualcodec_valle_ar_12hzv1,
    load_dualcodec_valle_nar_12hzv1,
    infer_process,
)
from dualcodec.utils import get_whisper_tokenizer
import dualcodec

# Load models
logger.info("Loading Valle models...")
ar_model = load_dualcodec_valle_ar_12hzv1()
nar_model = load_dualcodec_valle_nar_12hzv1()
tokenizer_model = get_whisper_tokenizer()
dualcodec_model = dualcodec.get_model("12hz_v1")
dualcodec_inference_obj = dualcodec.Inference(
    dualcodec_model=dualcodec_model, device=device, autocast=True
)
logger.info("Valle models loaded.")


def process_tts(
    ref_audio,
    ref_text,
    gen_text,
    remove_silence=False,
    cross_fade_duration=0.15,
    temperature=1.0,
    top_k=15,
    top_p=0.85,
    repeat_penalty=1.1,
    progress=gr.Progress(),
):
    if not ref_audio:
        gr.Warning("Please provide reference audio.")
        return None, None

    if not gen_text.strip():
        gr.Warning("Please enter text to generate.")
        return None, None

    # Preprocess reference audio and text
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio, ref_text)

    # Generate audio
    final_wave, final_sample_rate, combined_spectrogram = infer_process(
        ar_model_obj=ar_model,
        nar_model_obj=nar_model,
        dualcodec_inference_obj=dualcodec_inference_obj,
        tokenizer_obj=tokenizer_model,
        ref_audio=ref_audio,
        ref_text=ref_text,
        gen_text=gen_text,
        cross_fade_duration=cross_fade_duration,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
    )

    # Remove silence if requested
    if remove_silence and final_wave is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
            final_wave = final_wave.squeeze().cpu().numpy()

    return (final_sample_rate, final_wave) if final_wave is not None else None


# Create Gradio interface
with gr.Blocks(title="Valle TTS Demo") as demo:
    gr.Markdown("# Valle TTS Demo")
    gr.Markdown("Generate speech using reference audio and text.")

    with gr.Row():
        with gr.Column():
            ref_audio = gr.Audio(
                label="Reference Audio",
                type="filepath",
                format="wav",
            )
            ref_text = gr.Textbox(
                label="Reference Text",
                placeholder="Enter the transcript of the reference audio...",
                lines=2,
            )
            gen_text = gr.Textbox(
                label="Text to Generate",
                placeholder="Enter the text you want to generate speech for...",
                lines=4,
            )

            with gr.Accordion("Generation Parameters", open=False):
                with gr.Row():
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Temperature",
                        info="Higher values make output more random, lower values more deterministic",
                    )
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=15,
                        step=1,
                        label="Top-K",
                        info="Number of highest probability tokens to consider",
                    )
                with gr.Row():
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.85,
                        step=0.05,
                        label="Top-P",
                        info="Cumulative probability threshold for token selection",
                    )
                    repeat_penalty = gr.Slider(
                        minimum=1.0,
                        maximum=2.0,
                        value=1.1,
                        step=0.1,
                        label="Repeat Penalty",
                        info="Penalty for repeated tokens (higher = less repetition)",
                    )

            with gr.Row():
                remove_silence = gr.Checkbox(
                    label="Remove Silence",
                    value=False,
                    info="Remove long silence from generated audio",
                )
                cross_fade = gr.Slider(
                    minimum=0.0,
                    maximum=0.5,
                    value=0.15,
                    step=0.01,
                    label="Cross-fade Duration",
                    info="Duration of cross-fade between audio segments (seconds)",
                )

            generate_btn = gr.Button("Generate Speech", variant="primary")

        with gr.Column():
            output_audio = gr.Audio(
                label="Generated Audio",
                type="numpy",
                format="wav",
            )

    # Set up event handlers
    generate_btn.click(
        fn=process_tts,
        inputs=[
            ref_audio,
            ref_text,
            gen_text,
            remove_silence,
            cross_fade,
            temperature,
            top_k,
            top_p,
            repeat_penalty,
        ],
        outputs=[output_audio],
    )

if __name__ == "__main__":
    demo.launch(share=True)
