# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import gradio as gr
import torch
import torchaudio
import dualcodec
import base64
import soundfile as sf
import io

# Model configuration
MODEL_CONFIGS = {"12hz_v1": {"max_quantizers": 8}, "25hz_v1": {"max_quantizers": 12}}

w2v_path = "./w2v-bert-2.0"
dualcodec_model_path = "./dualcodec_ckpts"

# Global model variables
current_model = None
current_inference = None


def load_model(model_id):
    global current_model, current_inference
    current_model = dualcodec.get_model(model_id, dualcodec_model_path)
    current_inference = dualcodec.Inference(
        dualcodec_model=current_model,
        dualcodec_path=dualcodec_model_path,
        w2v_path=w2v_path,
        device="cuda",
    )
    return MODEL_CONFIGS[model_id]["max_quantizers"]


def process_audio(audio_file, model_id, n_quantizers, session_state):
    global current_model, current_inference
    if current_model is None or current_inference is None:
        load_model(model_id)

    # Load and process audio
    audio, sr = torchaudio.load(audio_file)
    audio = torchaudio.functional.resample(audio, sr, 24000)
    audio = audio.reshape(1, 1, -1)

    # Encode and decode
    semantic_codes, acoustic_codes = current_inference.encode(
        audio, n_quantizers=n_quantizers
    )
    out_audio = current_model.decode_from_codes(semantic_codes, acoustic_codes)

    # Prepare outputs
    generated_audio = (24000, out_audio.cpu().numpy().squeeze())

    # Update session state
    if session_state is None:
        session_state = {"history": []}

    # Add new entry to history
    new_entry = {
        "audio": generated_audio,
        "metadata": f"Model: {model_id}, VQs: {n_quantizers}",
    }
    session_state["history"].append(new_entry)

    # Limit history to 10 entries
    if len(session_state["history"]) > 10:
        session_state["history"].pop(0)  # Remove the oldest entry

    return generated_audio, session_state


def update_slider(model_id):
    return gr.update(maximum=MODEL_CONFIGS[model_id]["max_quantizers"])


def generate_history_html(session_state):
    if session_state is None or "history" not in session_state:
        return ""
    history_list = session_state["history"]
    if not history_list:
        return ""
    html = []
    for idx, entry in enumerate(history_list):
        sr, audio_data = entry["audio"]
        # Convert numpy array to bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sr, format="wav")
        buffer.seek(0)
        data_uri = "data:audio/wav;base64," + base64.b64encode(buffer.read()).decode()
        html.append(
            f'<div style="border: 1px solid #ccc; padding: 10px; margin: 10px;">'
            f"<h4>History Entry {idx+1}</h4>"
            f'<audio controls><source src="{data_uri}" type="audio/wav"></audio>'
            f'<p>{entry["metadata"]}</p>'
            f"</div>"
        )
    return "".join(html)


def clear_history(session_state):
    if session_state is not None and "history" in session_state:
        session_state["history"] = []
    return session_state, ""


# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# DualCodec Audio Demo")

    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=list(MODEL_CONFIGS.keys()), value="12hz_v1", label="Model"
        )
        n_quantizers = gr.Slider(
            minimum=1,
            maximum=MODEL_CONFIGS["12hz_v1"]["max_quantizers"],
            step=1,
            value=8,
            label="Number of Quantizers",
        )

    audio_input = gr.Audio(type="filepath", label="Input Audio")
    inference_button = gr.Button("Run Inference")

    # Reconstructed audio output
    audio_output_recon = gr.Audio(label="Reconstructed Audio")

    # History section
    gr.Markdown("## History Outputs")
    history_display = gr.HTML(label="History Audios")

    # Session state to store history audios (unique to each user)
    session_state = gr.State({"history": []})

    # Set up interactions
    model_dropdown.change(fn=update_slider, inputs=model_dropdown, outputs=n_quantizers)
    inference_button.click(
        fn=process_audio,
        inputs=[audio_input, model_dropdown, n_quantizers, session_state],
        outputs=[audio_output_recon, session_state],
    )
    session_state.change(
        fn=generate_history_html, inputs=session_state, outputs=history_display
    )

    # Clear history button
    clear_button = gr.Button("Clear History Audios")
    clear_button.click(
        fn=clear_history, inputs=session_state, outputs=[session_state, history_display]
    )


def main():
    demo.launch()


if __name__ == "__main__":
    main()
