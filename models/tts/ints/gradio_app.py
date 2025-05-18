# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import gradio as gr
from huggingface_hub import snapshot_download
from models.tts.ints.ints import Ints
from utils.util import load_config


def text_to_speech(
    prompt_text, text, prompt_audio, top_k=20, top_p=0.98, temperature=1.0
):
    gen_audio, debug_dict = ins_model(
        text, prompt_audio, prompt_text, top_k, top_p, temperature
    )
    # (sample rate in Hz, audio data as numpy array)
    output_audio = (24000, gen_audio)
    return output_audio, debug_dict


def create_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# Ints Text-to-Speech")

        with gr.Row():
            with gr.Column():
                prompt_text = gr.Textbox(
                    label="Prompt Text",
                    placeholder="Enter the prompt text here...",
                    lines=2,
                )
                text = gr.Textbox(
                    label="Text to Synthesize",
                    placeholder="Enter the text you want to convert to speech...",
                    lines=3,
                )
                prompt_audio = gr.Audio(label="Prompt Audio", type="filepath")

                with gr.Row():
                    top_k = gr.Slider(
                        minimum=10, maximum=100, value=20, step=1, label="Top-k"
                    )
                    top_p = gr.Slider(
                        minimum=0.5, maximum=1.0, value=0.98, step=0.01, label="Top-p"
                    )
                    temperature = gr.Slider(
                        minimum=0.5,
                        maximum=1.5,
                        value=1.0,
                        step=0.01,
                        label="Temperature",
                    )

                generate_btn = gr.Button("Generate Speech")

            with gr.Column():
                output_audio = gr.Audio(label="Generated Speech")
                debug_dict = gr.JSON(label="Debug Dict")

        generate_btn.click(
            fn=text_to_speech,
            inputs=[
                prompt_text,
                text,
                prompt_audio,
                top_k,
                top_p,
                temperature,
            ],
            outputs=[output_audio, debug_dict],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--use_flash_attn", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--config_path", type=str, default="models/tts/ints/ints.json")
    parser.add_argument("--model_name", type=str, default="ints_v2")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.4)
    args = parser.parse_args()

    ins_cfg = load_config(args.config_path)

    base_folder = snapshot_download("amphion/Ints")
    llm_path = os.path.join(base_folder, args.model_name)
    print(f"llm_path: {llm_path}")

    ins_model = Ints(
        llm_path=llm_path,
        cfg=ins_cfg,
        device=args.device,
        use_vllm=args.use_vllm,
        use_flash_attn=args.use_flash_attn,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    text_to_speech(
        prompt_text="We do not break. We never give in. We never back down.",
        prompt_audio="models/tts/maskgct/wav/prompt.wav",
        text="I will make America great again.",
    )

    demo = create_demo()
    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
    )
