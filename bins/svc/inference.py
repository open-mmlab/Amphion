# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import glob
from tqdm import tqdm
import json
import torch
import time

from models.svc.diffusion.diffusion_inference import DiffusionInference
from models.svc.comosvc.comosvc_inference import ComoSVCInference
from models.svc.transformer.transformer_inference import TransformerInference
from models.svc.vits.vits_inference import VitsInference
from utils.util import load_config
from utils.audio_slicer import split_audio, merge_segments_encodec
from processors import acoustic_extractor, content_extractor


def build_inference(args, cfg, infer_type="from_dataset"):
    supported_inference = {
        "DiffWaveNetSVC": DiffusionInference,
        "DiffComoSVC": ComoSVCInference,
        "TransformerSVC": TransformerInference,
        "VitsSVC": VitsInference,
    }

    inference_class = supported_inference[cfg.model_type]
    return inference_class(args, cfg, infer_type)


def prepare_for_audio_file(args, cfg, num_workers=1):
    preprocess_path = cfg.preprocess.processed_dir
    audio_name = cfg.inference.source_audio_name
    temp_audio_dir = os.path.join(preprocess_path, audio_name)

    ### eval file
    t = time.time()
    eval_file = prepare_source_eval_file(cfg, temp_audio_dir, audio_name)
    args.source = eval_file
    with open(eval_file, "r") as f:
        metadata = json.load(f)
    print("Prepare for meta eval data: {:.1f}s".format(time.time() - t))

    ### acoustic features
    t = time.time()
    acoustic_extractor.extract_utt_acoustic_features_serial(
        metadata, temp_audio_dir, cfg
    )
    if cfg.preprocess.use_min_max_norm_mel == True:
        acoustic_extractor.cal_mel_min_max(
            dataset=audio_name, output_path=preprocess_path, cfg=cfg, metadata=metadata
        )
    acoustic_extractor.cal_pitch_statistics_svc(
        dataset=audio_name, output_path=preprocess_path, cfg=cfg, metadata=metadata
    )
    print("Prepare for acoustic features: {:.1f}s".format(time.time() - t))

    ### content features
    t = time.time()
    content_extractor.extract_utt_content_features_dataloader(
        cfg, metadata, num_workers
    )
    print("Prepare for content features: {:.1f}s".format(time.time() - t))
    return args, cfg, temp_audio_dir


def merge_for_audio_segments(audio_files, args, cfg):
    audio_name = cfg.inference.source_audio_name
    target_singer_name = args.target_singer

    merge_segments_encodec(
        wav_files=audio_files,
        fs=cfg.preprocess.sample_rate,
        output_path=os.path.join(
            args.output_dir, "{}_{}.wav".format(audio_name, target_singer_name)
        ),
        overlap_duration=cfg.inference.segments_overlap_duration,
    )

    for tmp_file in audio_files:
        os.remove(tmp_file)


def prepare_source_eval_file(cfg, temp_audio_dir, audio_name):
    """
    Prepare the eval file (json) for an audio
    """

    audio_chunks_results = split_audio(
        wav_file=cfg.inference.source_audio_path,
        target_sr=cfg.preprocess.sample_rate,
        output_dir=os.path.join(temp_audio_dir, "wavs"),
        max_duration_of_segment=cfg.inference.segments_max_duration,
        overlap_duration=cfg.inference.segments_overlap_duration,
    )

    metadata = []
    for i, res in enumerate(audio_chunks_results):
        res["index"] = i
        res["Dataset"] = audio_name
        res["Singer"] = audio_name
        res["Uid"] = "{}_{}".format(audio_name, res["Uid"])
        metadata.append(res)

    eval_file = os.path.join(temp_audio_dir, "eval.json")
    with open(eval_file, "w") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False, sort_keys=True)

    return eval_file


def cuda_relevant(deterministic=False):
    torch.cuda.empty_cache()
    # TF32 on Ampere and above
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = True
    # Deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic)


def infer(args, cfg, infer_type):
    # Build inference
    t = time.time()
    trainer = build_inference(args, cfg, infer_type)
    print("Model Init: {:.1f}s".format(time.time() - t))

    # Run inference
    t = time.time()
    output_audio_files = trainer.inference()
    print("Model inference: {:.1f}s".format(time.time() - t))
    return output_audio_files


def build_parser():
    r"""Build argument parser for inference.py.
    Anything else should be put in an extra config YAML file.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="JSON/YAML file for configurations.",
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
        "--output_dir",
        type=str,
        default="conversion_results",
        help="Output directory. Default: ./conversion_results",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="warning",
        help="Logging level. Default: warning",
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
    return parser


def main():
    ### Parse arguments and config
    args = build_parser().parse_args()
    cfg = load_config(args.config)

    # CUDA settings
    cuda_relevant()

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
            merge_for_audio_segments(output_audio_files, args, cfg)

            # Keep or remove caches
            if not args.keep_cache:
                os.removedirs(cache_dir)

    else:
        ### Infer from dataset
        infer(args, cfg, infer_type="from_dataset")


if __name__ == "__main__":
    main()
