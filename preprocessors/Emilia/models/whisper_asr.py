# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import faster_whisper
from typing import List, Union, Optional, NamedTuple
import torch
import numpy as np
import tqdm
from whisperx.audio import N_SAMPLES, SAMPLE_RATE, load_audio, log_mel_spectrogram
from whisperx.types import TranscriptionResult, SingleSegment
from whisperx.asr import WhisperModel, FasterWhisperPipeline, find_numeral_symbol_tokens


class VadFreeFasterWhisperPipeline(FasterWhisperPipeline):
    """
    FasterWhisperModel without VAD
    """

    def __init__(
        self,
        model,
        options: NamedTuple,
        tokenizer=None,
        device: Union[int, str, "torch.device"] = -1,
        framework="pt",
        language: Optional[str] = None,
        suppress_numerals: bool = False,
        **kwargs,
    ):
        """
        Initialize the VadFreeFasterWhisperPipeline.

        Args:
            model: The Whisper model instance.
            options: Transcription options.
            tokenizer: The tokenizer instance.
            device: Device to run the model on.
            framework: The framework to use ('pt' for PyTorch).
            language: The language for transcription.
            suppress_numerals: Whether to suppress numeral tokens.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        super().__init__(
            model=model,
            vad=None,
            vad_params={},
            options=options,
            tokenizer=tokenizer,
            device=device,
            framework=framework,
            language=language,
            suppress_numerals=suppress_numerals,
            **kwargs,
        )

    def detect_language(self, audio: np.ndarray):
        """
        Detect the language of the audio.

        Args:
            audio (np.ndarray): The input audio signal.

        Returns:
            tuple: Detected language and its probability.
        """
        model_n_mels = self.model.feat_kwargs.get("feature_size")
        if audio.shape[0] > N_SAMPLES:
            # Randomly sample N_SAMPLES from the audio array
            start_index = np.random.randint(0, audio.shape[0] - N_SAMPLES)
            audio_sample = audio[start_index : start_index + N_SAMPLES]
        else:
            audio_sample = audio[:N_SAMPLES]
        padding = 0 if audio.shape[0] >= N_SAMPLES else N_SAMPLES - audio.shape[0]
        segment = log_mel_spectrogram(
            audio_sample,
            n_mels=model_n_mels if model_n_mels is not None else 80,
            padding=padding,
        )
        encoder_output = self.model.encode(segment)
        results = self.model.model.detect_language(encoder_output)
        language_token, language_probability = results[0][0]
        language = language_token[2:-2]
        return language, language_probability

    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        vad_segments: List[dict],
        batch_size=None,
        num_workers=0,
        language=None,
        task=None,
        chunk_size=30,
        print_progress=False,
        combined_progress=False,
    ) -> TranscriptionResult:
        """
        Transcribe the audio into text.

        Args:
            audio (Union[str, np.ndarray]): The input audio signal or path to audio file.
            vad_segments (List[dict]): List of VAD segments.
            batch_size (int, optional): Batch size for transcription. Defaults to None.
            num_workers (int, optional): Number of workers for loading data. Defaults to 0.
            language (str, optional): Language for transcription. Defaults to None.
            task (str, optional): Task type ('transcribe' or 'translate'). Defaults to None.
            chunk_size (int, optional): Size of chunks for processing. Defaults to 30.
            print_progress (bool, optional): Whether to print progress. Defaults to False.
            combined_progress (bool, optional): Whether to combine progress. Defaults to False.

        Returns:
            TranscriptionResult: The transcription result containing segments and language.
        """
        if isinstance(audio, str):
            audio = load_audio(audio)

        def data(audio, segments):
            for seg in segments:
                f1 = int(seg["start"] * SAMPLE_RATE)
                f2 = int(seg["end"] * SAMPLE_RATE)
                yield {"inputs": audio[f1:f2]}

        if self.tokenizer is None:
            language = language or self.detect_language(audio)
            task = task or "transcribe"
            self.tokenizer = faster_whisper.tokenizer.Tokenizer(
                self.model.hf_tokenizer,
                self.model.model.is_multilingual,
                task=task,
                language=language,
            )
        else:
            language = language or self.tokenizer.language_code
            task = task or self.tokenizer.task
            if task != self.tokenizer.task or language != self.tokenizer.language_code:
                self.tokenizer = faster_whisper.tokenizer.Tokenizer(
                    self.model.hf_tokenizer,
                    self.model.model.is_multilingual,
                    task=task,
                    language=language,
                )

        if self.suppress_numerals:
            previous_suppress_tokens = self.options.suppress_tokens
            numeral_symbol_tokens = find_numeral_symbol_tokens(self.tokenizer)
            new_suppressed_tokens = numeral_symbol_tokens + self.options.suppress_tokens
            new_suppressed_tokens = list(set(new_suppressed_tokens))
            self.options = self.options._replace(suppress_tokens=new_suppressed_tokens)

        segments: List[SingleSegment] = []
        batch_size = batch_size or self._batch_size
        total_segments = len(vad_segments)
        progress = tqdm.tqdm(total=total_segments, desc="Transcribing")
        for idx, out in enumerate(
            self.__call__(
                data(audio, vad_segments),
                batch_size=batch_size,
                num_workers=num_workers,
            )
        ):
            if print_progress:
                progress.update(1)
            text = out["text"]
            if batch_size in [0, 1, None]:
                text = text[0]
            segments.append(
                {
                    "text": text,
                    "start": round(vad_segments[idx]["start"], 3),
                    "end": round(vad_segments[idx]["end"], 3),
                    "speaker": vad_segments[idx].get("speaker", None),
                }
            )

        # revert the tokenizer if multilingual inference is enabled
        if self.preset_language is None:
            self.tokenizer = None

        # revert suppressed tokens if suppress_numerals is enabled
        if self.suppress_numerals:
            self.options = self.options._replace(
                suppress_tokens=previous_suppress_tokens
            )

        return {"segments": segments, "language": language}


def load_asr_model(
    whisper_arch: str,
    device: str,
    device_index: int = 0,
    compute_type: str = "float16",
    asr_options: Optional[dict] = None,
    language: Optional[str] = None,
    vad_model=None,
    vad_options=None,
    model: Optional[WhisperModel] = None,
    task: str = "transcribe",
    download_root: Optional[str] = None,
    threads: int = 4,
) -> VadFreeFasterWhisperPipeline:
    """
    Load a Whisper model for inference.

    Args:
        whisper_arch (str): The name of the Whisper model to load.
        device (str): The device to load the model on.
        device_index (int, optional): The device index. Defaults to 0.
        compute_type (str, optional): The compute type to use for the model. Defaults to "float16".
        asr_options (Optional[dict], optional): Options for ASR. Defaults to None.
        language (Optional[str], optional): The language of the model. Defaults to None.
        vad_model: The VAD model instance. Defaults to None.
        vad_options: Options for VAD. Defaults to None.
        model (Optional[WhisperModel], optional): The WhisperModel instance to use. Defaults to None.
        task (str, optional): The task type ('transcribe' or 'translate'). Defaults to "transcribe".
        download_root (Optional[str], optional): The root directory to download the model to. Defaults to None.
        threads (int, optional): The number of CPU threads to use per worker. Defaults to 4.

    Returns:
        VadFreeFasterWhisperPipeline: The loaded Whisper pipeline.

    Raises:
        ValueError: If the whisper architecture is not recognized.
    """

    if whisper_arch.endswith(".en"):
        language = "en"

    model = model or WhisperModel(
        whisper_arch,
        device=device,
        device_index=device_index,
        compute_type=compute_type,
        download_root=download_root,
        cpu_threads=threads,
    )
    if language is not None:
        tokenizer = faster_whisper.tokenizer.Tokenizer(
            model.hf_tokenizer,
            model.model.is_multilingual,
            task=task,
            language=language,
        )
    else:
        print(
            "No language specified, language will be detected for each audio file (increases inference time)."
        )
        tokenizer = None

    default_asr_options = {
        "beam_size": 5,
        "best_of": 5,
        "patience": 1,
        "length_penalty": 1,
        "repetition_penalty": 1,
        "no_repeat_ngram_size": 0,
        "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": False,
        "prompt_reset_on_temperature": 0.5,
        "initial_prompt": None,
        "prefix": None,
        "suppress_blank": True,
        "suppress_tokens": [-1],
        "without_timestamps": True,
        "max_initial_timestamp": 0.0,
        "word_timestamps": False,
        "prepend_punctuations": "\"'“¿([{-",
        "append_punctuations": "\"'.。,，!！?？:：”)]}、",
        "suppress_numerals": False,
        "max_new_tokens": None,
        "clip_timestamps": None,
        "hallucination_silence_threshold": None,
    }

    if asr_options is not None:
        default_asr_options.update(asr_options)

    suppress_numerals = default_asr_options["suppress_numerals"]
    del default_asr_options["suppress_numerals"]

    default_asr_options = faster_whisper.transcribe.TranscriptionOptions(
        **default_asr_options
    )

    return VadFreeFasterWhisperPipeline(
        model=model,
        options=default_asr_options,
        tokenizer=tokenizer,
        language=language,
        suppress_numerals=suppress_numerals,
    )
