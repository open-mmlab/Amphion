import copy
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import Resample
from models.codec.discriminator.layers import (
    NLayerDiscriminator,
    NLayerSpecDiscriminator,
)
from torch.nn.utils import weight_norm, spectral_norm
from torch.nn import Conv1d


def stft(x, fft_size, hop_size, win_length, window, use_complex=False):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """

    x_stft = torch.stft(
        x, fft_size, hop_size, win_length, window.to(x.device), return_complex=True
    )

    # clamp is needed to avoid nan or inf
    if not use_complex:
        return torch.sqrt(
            torch.clamp(x_stft.real**2 + x_stft.imag**2, min=1e-7, max=1e3)
        ).transpose(2, 1)
    else:
        res = torch.cat([x_stft.real.unsqueeze(1), x_stft.imag.unsqueeze(1)], dim=1)
        res = res.transpose(2, 3)  # [B, 2, T, F]
        return res


class HiFiGANPeriodDiscriminator(torch.nn.Module):
    """HiFiGAN period discriminator module."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        period=3,
        kernel_sizes=[5, 3],
        channels=32,
        downsample_scales=[3, 3, 3, 3, 1],
        channel_increasing_factor=4,
        max_downsample_channels=1024,
        bias=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_weight_norm=True,
        use_spectral_norm=False,
        cfg=None,
    ):
        """Initialize HiFiGANPeriodDiscriminator module.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            period (int): Period.
            kernel_sizes (list): Kernel sizes of initial conv layers and the final conv layer.
            channels (int): Number of initial channels.
            downsample_scales (list): List of downsampling scales.
            max_downsample_channels (int): Number of maximum downsampling channels.
            use_additional_convs (bool): Whether to use additional conv layers in residual blocks.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_spectral_norm (bool): Whether to use spectral norm.
                If set to true, it will be applied to all of the conv layers.
        """
        super().__init__()

        in_channels = (
            cfg.in_channels
            if cfg is not None and hasattr(cfg, "in_channels")
            else in_channels
        )
        out_channels = (
            cfg.out_channels
            if cfg is not None and hasattr(cfg, "out_channels")
            else out_channels
        )
        period = cfg.period if cfg is not None and hasattr(cfg, "period") else period
        kernel_sizes = (
            cfg.kernel_sizes
            if cfg is not None and hasattr(cfg, "kernel_sizes")
            else kernel_sizes
        )
        channels = (
            cfg.channels if cfg is not None and hasattr(cfg, "channels") else channels
        )
        downsample_scales = (
            cfg.downsample_scales
            if cfg is not None and hasattr(cfg, "downsample_scales")
            else downsample_scales
        )
        channel_increasing_factor = (
            cfg.channel_increasing_factor
            if cfg is not None and hasattr(cfg, "channel_increasing_factor")
            else channel_increasing_factor
        )
        max_downsample_channels = (
            cfg.max_downsample_channels
            if cfg is not None and hasattr(cfg, "max_downsample_channels")
            else max_downsample_channels
        )
        bias = cfg.bias if cfg is not None and hasattr(cfg, "bias") else bias
        nonlinear_activation = (
            cfg.nonlinear_activation
            if cfg is not None and hasattr(cfg, "nonlinear_activation")
            else nonlinear_activation
        )
        nonlinear_activation_params = (
            cfg.nonlinear_activation_params
            if cfg is not None and hasattr(cfg, "nonlinear_activation_params")
            else nonlinear_activation_params
        )
        use_weight_norm = (
            cfg.use_weight_norm
            if cfg is not None and hasattr(cfg, "use_weight_norm")
            else use_weight_norm
        )
        use_spectral_norm = (
            cfg.use_spectral_norm
            if cfg is not None and hasattr(cfg, "use_spectral_norm")
            else use_spectral_norm
        )

        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1, "Kernel size must be odd number."
        assert kernel_sizes[1] % 2 == 1, "Kernel size must be odd number."

        self.period = period
        self.convs = torch.nn.ModuleList()
        in_chs = in_channels
        out_chs = channels
        for downsample_scale in downsample_scales:
            self.convs += [
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_chs,
                        out_chs,
                        (kernel_sizes[0], 1),
                        (downsample_scale, 1),
                        padding=((kernel_sizes[0] - 1) // 2, 0),
                    ),
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    ),
                )
            ]
            in_chs = out_chs
            # NOTE(kan-bayashi): Use downsample_scale + 1?
            out_chs = min(out_chs * channel_increasing_factor, max_downsample_channels)
        self.output_conv = torch.nn.Conv2d(
            in_chs,
            out_channels,
            (kernel_sizes[1] - 1, 1),
            1,
            padding=((kernel_sizes[1] - 1) // 2, 0),
        )

        if use_weight_norm and use_spectral_norm:
            raise ValueError("Either use use_weight_norm or use_spectral_norm.")

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # apply spectral norm
        if use_spectral_norm:
            self.apply_spectral_norm()

    def forward(self, x, mask=None):
        """Calculate forward propagation.
        Args:
            c (Tensor): Input tensor (B, in_channels, T).
        Returns:
            list: List of each layer's tensors.
        """
        if mask is not None:
            # mask: (B, T) -> (B, 1, T)
            x = x * mask.unsqueeze(1)
        # transform 1d to 2d -> (B, C, T/P, P)
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t += n_pad
        x = x.view(b, c, t // self.period, self.period)

        # forward conv
        outs = []
        for layer in self.convs:
            x = layer(x)
            outs += [x]
        x = self.output_conv(x)
        x = torch.flatten(x, 1, -1)
        outs += [x]

        return outs

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def apply_spectral_norm(self):
        """Apply spectral normalization module from all of the layers."""

        def _apply_spectral_norm(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.spectral_norm(m)
                logging.debug(f"Spectral norm is applied to {m}.")

        self.apply(_apply_spectral_norm)


class HiFiGANMultiPeriodDiscriminator(torch.nn.Module):
    """HiFiGAN multi-period discriminator module."""

    def __init__(
        self,
        periods=[2, 3, 5, 7, 11],
        cfg=None,
    ):
        """Initialize HiFiGANMultiPeriodDiscriminator module.
        Args:
            periods (list): List of periods.
            discriminator_params (dict): Parameters for hifi-gan period discriminator module.
                The period parameter will be overwritten.
        """
        super().__init__()

        periods = (
            cfg.periods if cfg is not None and hasattr(cfg, "periods") else periods
        )

        # TODO: use a more general way to pass the cfg
        self.discriminators = torch.nn.ModuleList()
        for period in periods:
            cfg_period = copy.deepcopy(cfg)
            # cfg_period is a dictionary
            # drop the periods key
            if "periods" in cfg_period:
                del cfg_period["periods"]
            cfg_period["period"] = period
            self.discriminators += [HiFiGANPeriodDiscriminator(cfg=cfg_period)]

    def forward(self, x, mask=None):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input noise signal (B, 1, T).
        Returns:
            List: List of list of each discriminator outputs, which consists of each layer output tensors.
        """
        outs = []
        for f in self.discriminators:
            outs += [f(x, mask=mask)]

        return outs


class SpecDiscriminator(nn.Module):
    def __init__(
        self,
        stft_params=None,
        in_channels=1,
        out_channels=1,
        kernel_sizes=(7, 3),
        channels=32,
        max_downsample_channels=512,
        downsample_scales=(2, 2, 2),
        use_weight_norm=True,
        use_complex=False,
        cfg=None,
    ):
        super().__init__()

        stft_params = (
            cfg.stft_params
            if cfg is not None and hasattr(cfg, "stft_params")
            else stft_params
        )
        in_channels = (
            cfg.in_channels
            if cfg is not None and hasattr(cfg, "in_channels")
            else in_channels
        )
        out_channels = (
            cfg.out_channels
            if cfg is not None and hasattr(cfg, "out_channels")
            else out_channels
        )
        kernel_sizes = (
            cfg.kernel_sizes
            if cfg is not None and hasattr(cfg, "kernel_sizes")
            else kernel_sizes
        )
        channels = (
            cfg.channels if cfg is not None and hasattr(cfg, "channels") else channels
        )
        max_downsample_channels = (
            cfg.max_downsample_channels
            if cfg is not None and hasattr(cfg, "max_downsample_channels")
            else max_downsample_channels
        )
        downsample_scales = (
            cfg.downsample_scales
            if cfg is not None and hasattr(cfg, "downsample_scales")
            else downsample_scales
        )
        use_weight_norm = (
            cfg.use_weight_norm
            if cfg is not None and hasattr(cfg, "use_weight_norm")
            else use_weight_norm
        )
        use_complex = (
            cfg.use_complex
            if cfg is not None and hasattr(cfg, "use_complex")
            else use_complex
        )

        if stft_params is None:
            stft_params = {
                "fft_sizes": [1024, 2048, 512],
                "hop_sizes": [120, 240, 50],
                "win_lengths": [600, 1200, 240],
                "window": "hann_window",
            }

        self.stft_params = stft_params

        in_channels = in_channels if not use_complex else 2
        self.use_complex = use_complex

        self.model = nn.ModuleDict()
        for i in range(len(stft_params["fft_sizes"])):
            self.model["disc_" + str(i)] = NLayerSpecDiscriminator(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_sizes=kernel_sizes,
                channels=channels,
                max_downsample_channels=max_downsample_channels,
                downsample_scales=downsample_scales,
            )

        if use_weight_norm:
            self.apply_weight_norm()
        self.reset_parameters()

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask.unsqueeze(1)
        results = []
        i = 0
        x = x.squeeze(1)
        for _, disc in self.model.items():
            spec = stft(
                x,
                self.stft_params["fft_sizes"][i],
                self.stft_params["hop_sizes"][i],
                self.stft_params["win_lengths"][i],
                window=getattr(torch, self.stft_params["window"])(
                    self.stft_params["win_lengths"][i]
                ),
                use_complex=self.use_complex,
            )
            if not self.use_complex:
                spec = spec.transpose(1, 2).unsqueeze(1)
            else:
                spec = spec.transpose(2, 3)
            results.append(disc(spec))
            i += 1
        return results

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if (
                isinstance(m, nn.Conv1d)
                or isinstance(m, nn.ConvTranspose1d)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.ConvTranspose2d)
            ):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        def _reset_parameters(m):
            if (
                isinstance(m, nn.Conv1d)
                or isinstance(m, nn.ConvTranspose1d)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.ConvTranspose2d)
            ):
                m.weight.data.normal_(0.0, 0.02)

        self.apply(_reset_parameters)
