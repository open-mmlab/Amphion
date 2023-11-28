# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Denoise and cut into utterances for custom dataset
"""
import glob
from tqdm import tqdm
import os
from utils.audio_slicer import split_utterances_from_audio
from utils.io import save_audio
import torchaudio


def get_wav_files(input_dir):
    ## Files
    wav_files = []
    for suffix in ["wav", "flac"]:
        wav_files.extend(
            glob.glob(
                "{}".format(os.path.join(input_dir, "**", "*.{}".format(suffix))),
                recursive=True,
            )
        )

    print("wav_files sz: {}".format(len(wav_files)))
    return wav_files


def denoise(input_dir, output_dir):
    """
    Adopt DFSMN to denoise
    References: https://www.modelscope.cn/models/damo/speech_dfsmn_ans_psm_48k_causal/summary
    """

    ## Load Model
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    ans = pipeline(
        Tasks.acoustic_noise_suppression, model="damo/speech_dfsmn_ans_psm_48k_causal"
    )

    ## Handle
    for input_file in tqdm(get_wav_files(input_dir)):
        singer, song_file = input_file.split("/")[-2:]
        save_dir = os.path.join(output_dir, singer)
        os.makedirs(save_dir, exist_ok=True)
        output_file = os.path.join(save_dir, song_file)

        ans(input_file, output_path=output_file)


def cut_into_utterances(
    input_dir, output_dir, max_duration_of_utterance=10, db_threshold=-20
):
    for input_file in tqdm(get_wav_files(input_dir)):
        singer, song_file = input_file.split("/")[-2:]
        song = song_file.split(".")[0]

        save_dir = os.path.join(output_dir, singer, song)
        os.makedirs(save_dir, exist_ok=True)

        split_utterances_from_audio(
            wav_file=input_file,
            output_dir=save_dir,
            max_duration_of_utterance=max_duration_of_utterance,
            db_threshold=db_threshold,
        )


def resample(input_dir, output_dir, target_sr=24000):
    for input_file in tqdm(get_wav_files(input_dir)):
        singer, song, utterance_file = input_file.split("/")[-3:]

        save_dir = os.path.join(output_dir, singer, song)
        os.makedirs(save_dir, exist_ok=True)

        # Resample
        waveform, sample_rate = torchaudio.load(input_file)
        if sample_rate != target_sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sample_rate, new_freq=target_sr
            )

        # To mono, and adjust to the same volume peak
        save_audio(
            path=os.path.join(save_dir, utterance_file),
            waveform=waveform,
            fs=target_sr,
            turn_up=True,
        )


def vocalist_training_data():
    root = "/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/xueyaozhang/dataset/Vocalist/"
    s1_tag = "s1_separation"
    s2_tag = "s2_denoise"
    s3_tag = "s3_utterances"
    s4_tag = "s4_resample"

    s1 = os.path.join(root, s1_tag)
    s1_s2 = os.path.join(root, "-".join([s1_tag, s2_tag]))
    s1_s2_s3 = os.path.join(root, "-".join([s1_tag, s2_tag, s3_tag]))
    s1_s3 = os.path.join(root, "-".join([s1_tag, s3_tag]))
    s1_s3_s4 = os.path.join(root, "-".join([s1_tag, s3_tag, s4_tag]))

    for level in ["l1", "l2"]:
        ## Stage 2: Denoise
        # denoise(
        #     input_dir=os.path.join(s1, "vocalist_{}".format(level)),
        #     output_dir=os.path.join(s1_s2, "vocalist_{}".format(level)),
        # )

        ## Stage 3: Cut into utterances
        # cut_into_utterances(
        #     input_dir=os.path.join(s1_s2, "vocalist_{}".format(level)),
        #     output_dir=os.path.join(s1_s2_s3, "vocalist_{}".format(level)),
        # )

        # cut_into_utterances(
        #     input_dir=os.path.join(s1, "vocalist_{}".format(level)),
        #     output_dir=os.path.join(s1_s3, "vocalist_{}".format(level)),
        # )

        ## Stage 4: Resample
        resample(
            input_dir=os.path.join(s1_s3, "vocalist_{}".format(level)),
            output_dir=os.path.join(s1_s3_s4, "vocalist_{}".format(level)),
            target_sr=44100,
        )


if __name__ == "__main__":
    cut_into_utterances(
        input_dir=os.path.join(
            "/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/xueyaozhang/dataset/VocalistDemo/raw/"
        ),
        output_dir=os.path.join(
            "/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/xueyaozhang/dataset/VocalistDemo/utterances"
        ),
        max_duration_of_utterance=10,
        db_threshold=-20,
    )
