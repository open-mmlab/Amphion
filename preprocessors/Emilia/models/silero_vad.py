# Source: https://github.com/snakers4/silero-vad
#
# Copyright (c) 2024 snakers4
#
# This code is from a MIT-licensed repository. The full license text is available at the root of the source repository.
#
# Note: This code has been modified to fit the context of this repository.

import librosa
import torch
import numpy as np

VAD_THRESHOLD = 20
SAMPLING_RATE = 16000


class SileroVAD:
    """
    Voice Activity Detection (VAD) using Silero-VAD.
    """

    def __init__(self, local=False, model="silero_vad", device=torch.device("cpu")):
        """
        Initialize the VAD object.

        Args:
            local (bool, optional): Whether to load the model locally. Defaults to False.
            model (str, optional): The VAD model name to load. Defaults to "silero_vad".
            device (torch.device, optional): The device to run the model on. Defaults to 'cpu'.

        Returns:
            None

        Raises:
            RuntimeError: If loading the model fails.
        """
        try:
            vad_model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad" if not local else "vad/silero-vad",
                model=model,
                force_reload=False,
                onnx=True,
                source="github" if not local else "local",
            )
            self.vad_model = vad_model
            (get_speech_timestamps, _, _, _, _) = utils
            self.get_speech_timestamps = get_speech_timestamps
        except Exception as e:
            raise RuntimeError(f"Failed to load VAD model: {e}")

    def segment_speech(self, audio_segment, start_time, end_time, sampling_rate):
        """
        Segment speech from an audio segment and return a list of timestamps.

        Args:
            audio_segment (np.ndarray): The audio segment to be segmented.
            start_time (int): The start time of the audio segment in frames.
            end_time (int): The end time of the audio segment in frames.
            sampling_rate (int): The sampling rate of the audio segment.

        Returns:
            list: A list of timestamps, each containing the start and end times of speech segments in frames.

        Raises:
            ValueError: If the audio segment is invalid.
        """
        if audio_segment is None or not isinstance(audio_segment, (np.ndarray, list)):
            raise ValueError("Invalid audio segment")

        speech_timestamps = self.get_speech_timestamps(
            audio_segment, self.vad_model, sampling_rate=sampling_rate
        )

        adjusted_timestamps = [
            (ts["start"] + start_time, ts["end"] + start_time)
            for ts in speech_timestamps
        ]
        if not adjusted_timestamps:
            return []

        intervals = [
            end[0] - start[1]
            for start, end in zip(adjusted_timestamps[:-1], adjusted_timestamps[1:])
        ]

        segments = []

        def split_timestamps(start_index, end_index):
            if (
                start_index == end_index
                or adjusted_timestamps[end_index][1]
                - adjusted_timestamps[start_index][0]
                < 20 * sampling_rate
            ):
                segments.append([start_index, end_index])
            else:
                if not intervals[start_index:end_index]:
                    return
                max_interval_index = intervals[start_index:end_index].index(
                    max(intervals[start_index:end_index])
                )
                split_index = start_index + max_interval_index
                split_timestamps(start_index, split_index)
                split_timestamps(split_index + 1, end_index)

        split_timestamps(0, len(adjusted_timestamps) - 1)

        merged_timestamps = [
            [adjusted_timestamps[start][0], adjusted_timestamps[end][1]]
            for start, end in segments
        ]
        return merged_timestamps

    def vad(self, speakerdia, audio):
        """
        Process the audio based on the given speaker diarization dataframe.

        Args:
            speakerdia (pd.DataFrame): The diarization dataframe containing start, end, and speaker info.
            audio (dict): A dictionary containing the audio waveform and sample rate.

        Returns:
            list: A list of dictionaries containing processed audio segments with start, end, and speaker.
        """
        sampling_rate = audio["sample_rate"]
        audio_data = audio["waveform"]

        out = []
        last_end = 0
        speakers_seen = set()
        count_id = 0

        for index, row in speakerdia.iterrows():
            start = float(row["start"])
            end = float(row["end"])

            if end <= last_end:
                continue
            last_end = end

            start_frame = int(start * sampling_rate)
            end_frame = int(end * sampling_rate)
            if row["speaker"] not in speakers_seen:
                speakers_seen.add(row["speaker"])

            if end - start <= VAD_THRESHOLD:
                out.append(
                    {
                        "index": str(count_id).zfill(5),
                        "start": start,  # in seconds
                        "end": end,
                        "speaker": row["speaker"],  # same for all
                    }
                )
                count_id += 1
                continue

            temp_audio = audio_data[start_frame:end_frame]

            # resample from 24k to 16k
            temp_audio_resampled = librosa.resample(
                temp_audio, orig_sr=sampling_rate, target_sr=SAMPLING_RATE
            )

            for start_frame_sub, end_frame_sub in self.segment_speech(
                temp_audio_resampled,
                int(start * SAMPLING_RATE),
                int(end * SAMPLING_RATE),
                SAMPLING_RATE,
            ):
                out.append(
                    {
                        "index": str(count_id).zfill(5),
                        "start": start_frame_sub / SAMPLING_RATE,  # in seconds
                        "end": end_frame_sub / SAMPLING_RATE,
                        "speaker": row["speaker"],  # same for all
                    }
                )
                count_id += 1

        return out
