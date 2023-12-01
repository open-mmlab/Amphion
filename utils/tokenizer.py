# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This code is modified from 
# https://github.com/lifeiteng/vall-e/blob/9c69096d603ce13174fb5cb025f185e2e9b36ac7/valle/data/tokenizer.py

import re
from typing import Any, Dict, List, Optional, Pattern, Union

import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.language_switch import LanguageSwitch
from phonemizer.backend.espeak.words_mismatch import WordMismatch
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator

try:
    from pypinyin import Style, pinyin
    from pypinyin.style._utils import get_finals, get_initials
except Exception:
    pass


class PypinyinBackend:
    """PypinyinBackend for Chinese. Most codes is referenced from espnet.
    There are two types pinyin or initials_finals, one is
    just like "ni1 hao3", the other is like "n i1 h ao3".
    """

    def __init__(
        self,
        backend="initials_finals",
        punctuation_marks: Union[str, Pattern] = Punctuation.default_marks(),
    ) -> None:
        self.backend = backend
        self.punctuation_marks = punctuation_marks

    def phonemize(
        self, text: List[str], separator: Separator, strip=True, njobs=1
    ) -> List[str]:
        assert isinstance(text, List)
        phonemized = []
        for _text in text:
            _text = re.sub(" +", " ", _text.strip())
            _text = _text.replace(" ", separator.word)
            phones = []
            if self.backend == "pypinyin":
                for n, py in enumerate(
                    pinyin(
                        _text, style=Style.TONE3, neutral_tone_with_five=True
                    )
                ):
                    if all([c in self.punctuation_marks for c in py[0]]):
                        if len(phones):
                            assert phones[-1] == separator.syllable
                            phones.pop(-1)

                        phones.extend(list(py[0]))
                    else:
                        phones.extend([py[0], separator.syllable])
            elif self.backend == "pypinyin_initials_finals":
                for n, py in enumerate(
                    pinyin(
                        _text, style=Style.TONE3, neutral_tone_with_five=True
                    )
                ):
                    if all([c in self.punctuation_marks for c in py[0]]):
                        if len(phones):
                            assert phones[-1] == separator.syllable
                            phones.pop(-1)
                        phones.extend(list(py[0]))
                    else:
                        if py[0][-1].isalnum():
                            initial = get_initials(py[0], strict=False)
                            if py[0][-1].isdigit():
                                final = (
                                    get_finals(py[0][:-1], strict=False)
                                    + py[0][-1]
                                )
                            else:
                                final = get_finals(py[0], strict=False)
                            phones.extend(
                                [
                                    initial,
                                    separator.phone,
                                    final,
                                    separator.syllable,
                                ]
                            )
                        else:
                            assert ValueError
            else:
                raise NotImplementedError
            phonemized.append(
                "".join(phones).rstrip(f"{separator.word}{separator.syllable}")
            )
        return phonemized


class G2PModule:
    """Phonemize Text."""

    def __init__(
        self,
        language="en-us",
        backend="espeak",
        separator=Separator(word="_", syllable="-", phone="|"),
        preserve_punctuation=True,
        punctuation_marks: Union[str, Pattern] = Punctuation.default_marks(),
        with_stress: bool = False,
        tie: Union[bool, str] = False,
        language_switch: LanguageSwitch = "keep-flags",
        words_mismatch: WordMismatch = "ignore",
    ) -> None:
        
        self.backend = self._initialize_backend(
            backend, language, punctuation_marks, preserve_punctuation,
            with_stress, tie, language_switch, words_mismatch
        )
        self.separator = separator

    def _initialize_backend(
        self, backend, language, punctuation_marks, preserve_punctuation,
        with_stress, tie, language_switch, words_mismatch
    ):
        if backend == "espeak":
            return EspeakBackend(
                language,
                punctuation_marks=punctuation_marks,
                preserve_punctuation=preserve_punctuation,
                with_stress=with_stress,
                tie=tie,
                language_switch=language_switch,
                words_mismatch=words_mismatch,
            )
        elif backend in ["pypinyin", "pypinyin_initials_finals"]:
            return PypinyinBackend(
                backend=backend,
                punctuation_marks=punctuation_marks + self.separator.word,
            )
        else:
            raise NotImplementedError(f"{backend}")

    def to_list(self, phonemized: str) -> List[str]:
        fields = []
        for word in phonemized.split(self.separator.word):
            pp = re.findall(r"\w+|[^\w\s]", word, re.UNICODE)
            fields.extend(
                [p for p in pp if p != self.separator.phone]
                + [self.separator.word]
            )
        assert len("".join(fields[:-1])) == len(phonemized) - phonemized.count(
            self.separator.phone
        )
        return fields[:-1]
    

    def __call__(self, text, strip=True) -> List[List[str]]:
        if isinstance(text, str):
            text = [text]

        phonemized = self.backend.phonemize(
            text, separator=self.separator, strip=strip, njobs=1
        )
        return [self.to_list(p) for p in phonemized]



def tokenize_text(tokenizer: G2PModule, text: str) -> List[str]:
    phonemes = tokenizer([text.strip()])
    return phonemes[0] 



# class AudioTokenizer:
#     """EnCodec audio."""

#     def __init__(
#         self,
#         device: Any = None,
#     ) -> None:
#         # Instantiate a pretrained EnCodec model
#         model = EncodecModel.encodec_model_24khz()
#         model.set_target_bandwidth(6.0)
#         remove_encodec_weight_norm(model)

#         if not device:
#             device = torch.device("cpu")
#             if torch.cuda.is_available():
#                 device = torch.device("cuda:0")

#         self._device = device

#         self.codec = model.to(device)
#         self.sample_rate = model.sample_rate
#         self.channels = model.channels

#     @property
#     def device(self):
#         return self._device

#     def encode(self, wav: torch.Tensor) -> torch.Tensor:
#         return self.codec.encode(wav.to(self.device))

#     def decode(self, frames: torch.Tensor) -> torch.Tensor:
#         return self.codec.decode(frames)


# def tokenize_audio(tokenizer: AudioTokenizer, audio_path: str):
#     # Load and pre-process the audio waveform
#     wav, sr = torchaudio.load(audio_path)
#     wav = convert_audio(wav, sr, tokenizer.sample_rate, tokenizer.channels)
#     wav = wav.unsqueeze(0)

#     # Extract discrete codes from EnCodec
#     with torch.no_grad():
#         encoded_frames = tokenizer.encode(wav)
#     return encoded_frames




class AudioTokenizer:
    """EnCodec audio tokenizer for encoding and decoding audio.

    Attributes:
        device: The device on which the codec model is loaded.
        codec: The pretrained EnCodec model.
        sample_rate: Sample rate of the model.
        channels: Number of audio channels in the model.
    """

    def __init__(self, device: Any = None) -> None:
        model = EncodecModel.encodec_model_24khz()
        model.set_target_bandwidth(6.0)
        remove_encodec_weight_norm(model)

        if not device:
            device = torch.device("cpu")
            if torch.cuda.is_available():
                device = torch.device("cuda:0")

        self._device = device

        # self.device = self._select_device(device)
        self.codec = model.to(device)
        self.sample_rate = model.sample_rate
        self.channels = model.channels

    # def _select_device(self, device: Any) -> torch.device:
    #     """Select the device for the model."""
    #     if not device:
    #         device = torch.device("cpu")
    #         if torch.cuda.is_available():
    #             device = torch.device("cuda:0")
        # return device        
        # return device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @property
    def device(self):
        return self._device

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """Encode the audio waveform.

        Args:
            wav: A tensor representing the audio waveform.

        Returns:
            A tensor representing the encoded audio.
        """
        return self.codec.encode(wav.to(self.device))

    def decode(self, frames: torch.Tensor) -> torch.Tensor:
        """Decode the encoded audio frames.

        Args:
            frames: A tensor representing the encoded audio frames.

        Returns:
            A tensor representing the decoded audio waveform.
        """
        return self.codec.decode(frames)



def tokenize_audio(tokenizer: AudioTokenizer, audio_path: str):
    """
    Tokenize the audio waveform using the given AudioTokenizer.

    Args:
        tokenizer: An instance of AudioTokenizer.
        audio_path: Path to the audio file.

    Returns:
        A tensor of encoded frames from the audio.

    Raises:
        FileNotFoundError: If the audio file is not found.
        RuntimeError: If there's an error processing the audio data.
    """
    # try:
        # Load and preprocess the audio waveform
    wav, sr = torchaudio.load(audio_path)
    wav = convert_audio(wav, sr, tokenizer.sample_rate, tokenizer.channels)
    wav = wav.unsqueeze(0)

    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = tokenizer.encode(wav)
    return encoded_frames

    # except FileNotFoundError:
    #     raise FileNotFoundError(f"Audio file not found at {audio_path}")
    # except Exception as e:
    #     raise RuntimeError(f"Error processing audio data: {e}")



def remove_encodec_weight_norm(model):
    from encodec.modules import SConv1d
    from encodec.modules.seanet import SConvTranspose1d, SEANetResnetBlock
    from torch.nn.utils import remove_weight_norm

    encoder = model.encoder.model
    for key in encoder._modules:
        if isinstance(encoder._modules[key], SEANetResnetBlock):
            remove_weight_norm(encoder._modules[key].shortcut.conv.conv)
            block_modules = encoder._modules[key].block._modules
            for skey in block_modules:
                if isinstance(block_modules[skey], SConv1d):
                    remove_weight_norm(block_modules[skey].conv.conv)
        elif isinstance(encoder._modules[key], SConv1d):
            remove_weight_norm(encoder._modules[key].conv.conv)

    decoder = model.decoder.model
    for key in decoder._modules:
        if isinstance(decoder._modules[key], SEANetResnetBlock):
            remove_weight_norm(decoder._modules[key].shortcut.conv.conv)
            block_modules = decoder._modules[key].block._modules
            for skey in block_modules:
                if isinstance(block_modules[skey], SConv1d):
                    remove_weight_norm(block_modules[skey].conv.conv)
        elif isinstance(decoder._modules[key], SConvTranspose1d):
            remove_weight_norm(decoder._modules[key].convtr.convtr)
        elif isinstance(decoder._modules[key], SConv1d):
            remove_weight_norm(decoder._modules[key].conv.conv)


def extract_encodec_token(wav_path):
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)

    wav, sr = torchaudio.load(wav_path)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0)
    if torch.cuda.is_available():
        model = model.cuda()
        wav = wav.cuda()
    with torch.no_grad():
        encoded_frames = model.encode(wav)
        codes_ = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
        codes = codes_.cpu().numpy()[0,:,:].T # [T, 8]
        
        return codes