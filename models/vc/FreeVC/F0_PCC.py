import numpy as np
import pyworld as pw
import argparse
import librosa
import os


def get_f0(x, fs=16000, n_shift=160):
    x = x.astype(np.float64)
    frame_period = n_shift / fs * 1000
    f0, timeaxis = pw.dio(x, fs, frame_period=frame_period)  # type:ignore
    f0 = pw.stonemask(x, f0, timeaxis, fs)  # type:ignore
    return f0


def compute_f0(wav, sr=16000, frame_period=10.0):
    wav = wav.astype(np.float64)
    f0, timeaxis = pw.harvest(  # type:ignore
        wav, sr, frame_period=frame_period, f0_floor=20.0, f0_ceil=600.0
    )
    return f0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--txtpath", type=str, default="samples.txt", help="path to txt file")
    parser.add_argument(
        "--src_path",
        type=str,
        default=r"data\VCTK\test_data",
        help="path to src audio files",
    )
    parser.add_argument(
        "--tgt_path",
        type=str,
        default=r"data\VCTK\test_output",
        help="path to output audio files",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=r"result\pcc.txt",
        help="path tot the pcc output file",
    )
    args = parser.parse_args()

    pccs = []
    i = 0
    src_files = [f for f in os.listdir(args.src_path) if f.endswith(".wav")]

    for filename in src_files:
        # print(filename)
        path_of_src_audio = os.path.join(args.src_path, filename)
        path_of_res_audio = os.path.join(
            args.tgt_path, os.path.splitext(filename)[0] + "_sync.wav"
        )
        # print(os.path.exists(path_of_res_audio))

        if os.path.exists(path_of_res_audio):
            # print(path_of_res_audio)
            # 加载音频文件
            src = librosa.load(path_of_src_audio, sr=16000)[0]
            src_f0 = get_f0(src)
            tgt = librosa.load(path_of_res_audio, sr=16000)[0]
            tgt_f0 = get_f0(tgt)
            if sum(src_f0) == 0:
                src_f0 = compute_f0(src)
                tgt_f0 = compute_f0(tgt)
                # print(rawline)
            pcc = np.corrcoef(src_f0[: tgt_f0.shape[-1]], tgt_f0[: src_f0.shape[-1]])[
                0, 1
            ]
            print(pcc)
            # print(i, pcc)
            if not np.isnan(pcc.item()):
                pccs.append(pcc.item())

        else:
            print(f"Warning: No matching file for {filename} in result folder")

    with open(args.output_path, "w") as f:
        for pcc in pccs:
            f.write(f"{pcc}\n")
        pcc = sum(pccs) / len(pccs)
        f.write(f"mean pcc: {pcc}")

    print("mean: ", pcc)
