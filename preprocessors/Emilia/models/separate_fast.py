# Copyright (c) 2023 seanghay
#
# This code is from an unliscensed repository.
#
# Note: This code has been modified to fit the context of this repository.
#       This code is included in an MIT-licensed repository.
#       The repository's MIT license does not apply to this code.

# This code is modified from https://github.com/seanghay/uvr-mdx-infer/blob/main/separate.py

import torch
import numpy as np
import onnxruntime as ort
from tqdm import tqdm


class ConvTDFNet:
    """
    ConvTDFNet - Convolutional Temporal Frequency Domain Network.
    """

    def __init__(self, target_name, L, dim_f, dim_t, n_fft, hop=1024):
        """
        Initialize ConvTDFNet.

        Args:
            target_name (str): The target name for separation.
            L (int): Number of layers.
            dim_f (int): Dimension in the frequency domain.
            dim_t (int): Dimension in the time domain (log2).
            n_fft (int): FFT size.
            hop (int, optional): Hop size. Defaults to 1024.

        Returns:
            None
        """
        super(ConvTDFNet, self).__init__()
        self.dim_c = 4
        self.dim_f = dim_f
        self.dim_t = 2**dim_t
        self.n_fft = n_fft
        self.hop = hop
        self.n_bins = self.n_fft // 2 + 1
        self.chunk_size = hop * (self.dim_t - 1)
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True)
        self.target_name = target_name

        out_c = self.dim_c * 4 if target_name == "*" else self.dim_c

        self.freq_pad = torch.zeros([1, out_c, self.n_bins - self.dim_f, self.dim_t])
        self.n = L // 2

    def stft(self, x):
        """
        Perform Short-Time Fourier Transform (STFT).

        Args:
            x (torch.Tensor): Input waveform.

        Returns:
            torch.Tensor: STFT of the input waveform.
        """
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            center=True,
            return_complex=True,
        )
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape(
            [-1, self.dim_c, self.n_bins, self.dim_t]
        )
        return x[:, :, : self.dim_f]

    def istft(self, x, freq_pad=None):
        """
        Perform Inverse Short-Time Fourier Transform (ISTFT).

        Args:
            x (torch.Tensor): Input STFT.
            freq_pad (torch.Tensor, optional): Frequency padding. Defaults to None.

        Returns:
            torch.Tensor: Inverse STFT of the input.
        """
        freq_pad = (
            self.freq_pad.repeat([x.shape[0], 1, 1, 1])
            if freq_pad is None
            else freq_pad
        )
        x = torch.cat([x, freq_pad], -2)
        c = 4 * 2 if self.target_name == "*" else 2
        x = x.reshape([-1, c, 2, self.n_bins, self.dim_t]).reshape(
            [-1, 2, self.n_bins, self.dim_t]
        )
        x = x.permute([0, 2, 3, 1])
        x = x.contiguous()
        x = torch.view_as_complex(x)
        x = torch.istft(
            x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True
        )
        return x.reshape([-1, c, self.chunk_size])


class Predictor:
    """
    Predictor class for source separation using ConvTDFNet and ONNX Runtime.
    """

    def __init__(self, args, device):
        """
        Initialize the Predictor.

        Args:
            args (dict): Configuration arguments.
            device (str): Device to run the model ('cuda' or 'cpu').

        Returns:
            None

        Raises:
            ValueError: If the provided device is not 'cuda' or 'cpu'.
        """
        self.args = args
        self.model_ = ConvTDFNet(
            target_name="vocals",
            L=11,
            dim_f=args["dim_f"],
            dim_t=args["dim_t"],
            n_fft=args["n_fft"],
        )

        if device == "cuda":
            self.model = ort.InferenceSession(
                args["model_path"], providers=["CUDAExecutionProvider"]
            )
        elif device == "cpu":
            self.model = ort.InferenceSession(
                args["model_path"], providers=["CPUExecutionProvider"]
            )
        else:
            raise ValueError("Device must be either 'cuda' or 'cpu'")

    def demix(self, mix):
        """
        Separate the sources from the input mix.

        Args:
            mix (np.ndarray): Input mixture signal.

        Returns:
            np.ndarray: Separated sources.

        Raises:
            AssertionError: If margin is zero.
        """
        samples = mix.shape[-1]
        margin = self.args["margin"]
        chunk_size = self.args["chunks"] * 44100

        assert margin != 0, "Margin cannot be zero!"

        if margin > chunk_size:
            margin = chunk_size

        segmented_mix = {}

        if self.args["chunks"] == 0 or samples < chunk_size:
            chunk_size = samples

        counter = -1
        for skip in range(0, samples, chunk_size):
            counter += 1
            s_margin = 0 if counter == 0 else margin
            end = min(skip + chunk_size + margin, samples)
            start = skip - s_margin
            segmented_mix[skip] = mix[:, start:end].copy()
            if end == samples:
                break

        sources = self.demix_base(segmented_mix, margin_size=margin)
        return sources

    def demix_base(self, mixes, margin_size):
        """
        Base function for source separation.

        Args:
            mixes (dict): Dictionary of segmented mixtures.
            margin_size (int): Size of the margin.

        Returns:
            np.ndarray: Separated sources.
        """
        chunked_sources = []
        progress_bar = tqdm(total=len(mixes))
        progress_bar.set_description("Source separation")

        for mix in mixes:
            cmix = mixes[mix]
            sources = []
            n_sample = cmix.shape[1]
            model = self.model_
            trim = model.n_fft // 2
            gen_size = model.chunk_size - 2 * trim
            pad = gen_size - n_sample % gen_size
            mix_p = np.concatenate(
                (np.zeros((2, trim)), cmix, np.zeros((2, pad)), np.zeros((2, trim))), 1
            )
            mix_waves = []
            i = 0
            while i < n_sample + pad:
                waves = np.array(mix_p[:, i : i + model.chunk_size])
                mix_waves.append(waves)
                i += gen_size

            mix_waves = torch.tensor(np.array(mix_waves), dtype=torch.float32)

            with torch.no_grad():
                _ort = self.model
                spek = model.stft(mix_waves)
                if self.args["denoise"]:
                    spec_pred = (
                        -_ort.run(None, {"input": -spek.cpu().numpy()})[0] * 0.5
                        + _ort.run(None, {"input": spek.cpu().numpy()})[0] * 0.5
                    )
                    tar_waves = model.istft(torch.tensor(spec_pred))
                else:
                    tar_waves = model.istft(
                        torch.tensor(_ort.run(None, {"input": spek.cpu().numpy()})[0])
                    )
                tar_signal = (
                    tar_waves[:, :, trim:-trim]
                    .transpose(0, 1)
                    .reshape(2, -1)
                    .numpy()[:, :-pad]
                )

                start = 0 if mix == 0 else margin_size
                end = None if mix == list(mixes.keys())[::-1][0] else -margin_size

                if margin_size == 0:
                    end = None

                sources.append(tar_signal[:, start:end])

                progress_bar.update(1)

            chunked_sources.append(sources)
        _sources = np.concatenate(chunked_sources, axis=-1)

        progress_bar.close()
        return _sources

    def predict(self, mix):
        """
        Predict the separated sources from the input mix.

        Args:
            mix (np.ndarray): Input mixture signal.

        Returns:
            tuple: Tuple containing the mixture minus the separated sources and the separated sources.
        """
        if mix.ndim == 1:
            mix = np.asfortranarray([mix, mix])

        tail = mix.shape[1] % (self.args["chunks"] * 44100)
        if mix.shape[1] % (self.args["chunks"] * 44100) != 0:
            mix = np.pad(
                mix,
                (
                    (0, 0),
                    (
                        0,
                        self.args["chunks"] * 44100
                        - mix.shape[1] % (self.args["chunks"] * 44100),
                    ),
                ),
            )

        mix = mix.T
        sources = self.demix(mix.T)
        opt = sources[0].T

        if tail != 0:
            return ((mix - opt)[: -(self.args["chunks"] * 44100 - tail), :], opt)
        else:
            return ((mix - opt), opt)
