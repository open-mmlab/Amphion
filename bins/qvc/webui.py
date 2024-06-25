# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gradio as gr
import os
import shutil
from inf_preprocess import *
from inference import *
from post_process import *
from preprocess import *
from train import *

def processing_audio(infsource, tarsource, shifts):
    global cfg, args
    target_folder = 'temp'
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)
    inf_preprocess(infsource, tarsource)
    preprocess(cfg, args)
    if shifts is not None:
        args.trans_key = shifts
    if (
        type(cfg.preprocess.data_augment) == list
        and len(cfg.preprocess.data_augment) > 0
    ):
        new_datasets_list = []
        for dataset in cfg.preprocess.data_augment:
            new_datasets = [
                f"{dataset}_pitch_shift" if cfg.preprocess.use_pitch_shift else None,
                (
                    f"{dataset}_formant_shift"
                    if cfg.preprocess.use_formant_shift
                    else None
                ),
                f"{dataset}_equalizer" if cfg.preprocess.use_equalizer else None,
                f"{dataset}_time_stretch" if cfg.preprocess.use_time_stretch else None,
            ]
            new_datasets_list.extend(filter(None, new_datasets))
        cfg.dataset.extend(new_datasets_list)

    # CUDA settings
    cuda_relevant()
    # Build trainer
    trainer = build_trainer(args, cfg)
    trainer.train_loop()
    
    args.source = 'temp/temp0'
    if os.path.isdir(args.source):
        ### Infer from file

        # Get all the source audio files (.wav, .flac, .mp3)
        source_audio_dir = args.source
        audio_list = []
        for suffix in ["wav", "flac", "mp3"]:
            audio_list += glob.glob(
                os.path.join(source_audio_dir, "**/*.{}".format(suffix)), recursive=True
            )
        print("There are {} source audios: ".format(len(audio_list)))

        # Infer for every file as dataset
        output_root_path = args.output_dir
        for audio_path in tqdm(audio_list):
            audio_name = audio_path.split("/")[-1].split(".")[0]
            args.output_dir = os.path.join(output_root_path, audio_name)
            print("\n{}\nConversion for {}...\n".format("*" * 10, audio_name))

            cfg.inference.source_audio_path = audio_path
            cfg.inference.source_audio_name = audio_name
            cfg.inference.segments_max_duration = 10.0
            cfg.inference.segments_overlap_duration = 1.0

            # Prepare metadata and features
            args, cfg, cache_dir = prepare_for_audio_file(args, cfg)

            # Infer from file
            output_audio_files = infer(args, cfg, infer_type="from_file")

            # Merge the split segments
            result = merge_for_audio_segments(output_audio_files, args, cfg)

            # Keep or remove caches
            if not args.keep_cache:
                os.removedirs(cache_dir)

    else:
        ### Infer from dataset
        infer(args, cfg, infer_type="from_dataset")
    return result


def main():
    infsource_audio = gr.Audio(label="Source Audio", type="filepath")
    tarsource_audio = gr.Audio(label="Target Audio", type="filepath")
    options1 = gr.Dropdown(["autoshift", "-10", "-9", "-8", "-7", "-6", "-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"], label="How many semitones you want to transpose?")
    outputs =  gr.Audio(label="Output Audio")
    inputs = [tarsource_audio, infsource_audio, options1]
    title = "Amphion-QuickVC"
    
    gr.Interface(processing_audio, inputs, outputs, title=title).queue().launch(server_name="0.0.0.0", share=True)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config.json",
        help="json files for configurations.",
        required=True,
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="exp_name",
        help="A specific name to note the experiment",
        required=True,
    )
    parser.add_argument("--num_workers", type=int, default=int(cpu_count()))
    parser.add_argument("--prepare_alignment", type=bool, default=False)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If specified, to resume from the existing checkpoint.",
    )
    parser.add_argument(
        "--resume_from_ckpt_path",
        type=str,
        default="",
        help="The specific checkpoint path that you want to resume from.",
    )
    parser.add_argument(
        "--resume_type",
        type=str,
        default="",
        help="`resume` for loading all the things (including model weights, optimizer, scheduler, and random states). `finetune` for loading only the model weights",
    )
    parser.add_argument(
        "--log_level", default="warning", help="logging level (debug, info, warning)"
    )
    parser.add_argument(
        "--acoustics_dir",
        type=str,
        help="Acoustics model checkpoint directory. If a directory is given, "
        "search for the latest checkpoint dir in the directory. If a specific "
        "checkpoint dir is given, directly load the checkpoint.",
    )
    parser.add_argument(
        "--vocoder_dir",
        type=str,
        required=True,
        help="Vocoder checkpoint directory. Searching behavior is the same as "
        "the acoustics one.",
    )
    parser.add_argument(
        "--target_singer",
        type=str,
        required=True,
        help="convert to a specific singer (e.g. --target_singers singer_id).",
    )
    parser.add_argument(
        "--trans_key",
        default=0,
        help="0: no pitch shift; autoshift: pitch shift;  int: key shift.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="source_audio",
        help="Source audio file or directory. If a JSON file is given, "
        "inference from dataset is applied. If a directory is given, "
        "inference from all wav/flac/mp3 audio files in the directory is applied. "
        "Default: inference from all wav/flac/mp3 audio files in ./source_audio",
    )
    parser.add_argument(
        "--infsource",
        type=str,
        default="source_audio",
        help="Source audio file or directory. If a JSON file is given, "
        "inference from dataset is applied. If a directory is given, "
        "inference from all wav/flac/mp3 audio files in the directory is applied. "
        "Default: inference from all wav/flac/mp3 audio files in ./source_audio",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="conversion_results",
        help="Output directory. Default: ./conversion_results",
    )

    parser.add_argument(
        "--keep_cache",
        action="store_true",
        default=True,
        help="Keep cache files. Only applicable to inference from files.",
    )
    parser.add_argument(
        "--diffusion_inference_steps",
        type=int,
        default=1000,
        help="Number of inference steps. Only applicable to diffusion inference.",
    )
    
    args = parser.parse_args()
    cfg = load_config(args.config)
    main()
