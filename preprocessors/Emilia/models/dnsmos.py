# Source: https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS
#
# Copyright (c) 2022 Microsoft
#
# This code is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.
# The full license text is available at the root of the source repository.
#
# Note: This code has been modified to fit the context of this repository.
#       This code is included in an MIT-licensed repository.
#       The repository's MIT license does not apply to this code.

import os
import librosa
import numpy as np
import onnxruntime as ort
import pandas as pd
import tqdm
import warnings


warnings.filterwarnings("ignore")

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01


class ComputeScore:
    """
    ComputeScore class for evaluating DNSMOS.
    """

    def __init__(self, primary_model_path, device="cpu") -> None:
        """
        Initialize the ComputeScore object.

        Args:
            primary_model_path (str): Path to the primary model.
            device (str): Device to run the models on ('cpu' or 'cuda').

        Returns:
            None

        Raises:
            RuntimeError: If the device is not supported.
        """
        if device == "cuda":
            self.onnx_sess = ort.InferenceSession(
                primary_model_path, providers=["CUDAExecutionProvider"]
            )
            print("Using CUDA:", self.onnx_sess.get_providers())
        else:
            self.onnx_sess = ort.InferenceSession(primary_model_path)

    def audio_melspec(
        self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True
    ):
        """
        Compute the mel spectrogram of an audio signal.

        Args:
            audio (np.ndarray): Input audio signal.
            n_mels (int): Number of mel bands.
            frame_size (int): Size of the FFT window.
            hop_length (int): Number of samples between successive frames.
            sr (int): Sampling rate.
            to_db (bool): Whether to convert the power spectrogram to decibel units.

        Returns:
            np.ndarray: Mel spectrogram.
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length, n_mels=n_mels
        )
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        """
        Apply polynomial fitting to MOS scores.

        Args:
            sig (float): Signal MOS score.
            bak (float): Background MOS score.
            ovr (float): Overall MOS score.
            is_personalized_MOS (bool): Flag for personalized MOS.

        Returns:
            tuple: Tuple containing the adjusted signal, background, and overall MOS scores.
        """
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def __call__(self, audio, sampling_rate, is_personalized_MOS):
        """
        Compute DNSMOS scores for an audio signal.

        Args:
            audio (np.ndarray or str): Input audio signal or path to audio file.
            sampling_rate (int): Sampling rate of the input audio.
            is_personalized_MOS (bool): Flag for personalized MOS.

        Returns:
            dict: Dictionary containing MOS scores.

        Raises:
            ValueError: If the input audio is not valid.
        """
        fs = SAMPLING_RATE
        if isinstance(audio, str):
            audio, _ = librosa.load(audio, sr=fs)
        elif sampling_rate != fs:
            # resample audio
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=fs)

        actual_audio_len = len(audio)

        len_samples = int(INPUT_LENGTH * fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / fs) - INPUT_LENGTH) + 1
        hop_len_samples = fs
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []

        for idx in range(num_hops):
            audio_seg = audio[
                int(idx * hop_len_samples) : int((idx + INPUT_LENGTH) * hop_len_samples)
            ]
            if len(audio_seg) < len_samples:
                continue
            input_features = np.array(audio_seg).astype("float32")[np.newaxis, :]
            oi = {"input_1": input_features}
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(
                mos_sig_raw, mos_bak_raw, mos_ovr_raw, is_personalized_MOS
            )
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)

        clip_dict = {
            "filename": "audio_clip",
            "len_in_sec": actual_audio_len / fs,
            "sr": fs,
            "num_hops": num_hops,
            "OVRL_raw": np.mean(predicted_mos_ovr_seg_raw),
            "SIG_raw": np.mean(predicted_mos_sig_seg_raw),
            "BAK_raw": np.mean(predicted_mos_bak_seg_raw),
            "OVRL": np.mean(predicted_mos_ovr_seg),
            "SIG": np.mean(predicted_mos_sig_seg),
            "BAK": np.mean(predicted_mos_bak_seg),
        }
        return clip_dict
