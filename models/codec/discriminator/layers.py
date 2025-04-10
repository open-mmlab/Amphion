import torch.nn as nn
import numpy as np


class NLayerDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_sizes=(5, 3),
        channels=16,
        max_downsample_channels=512,
        downsample_scales=(4, 4, 4),
    ):
        super().__init__()

        # check kernel size is valid
        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1
        assert kernel_sizes[1] % 2 == 1

        model = nn.ModuleDict()

        model["layer_0"] = nn.Sequential(
            nn.Conv1d(
                in_channels,
                channels,
                kernel_size=np.prod(kernel_sizes),
                padding=(np.prod(kernel_sizes) - 1) // 2,
            ),
            nn.LeakyReLU(0.2, True),
        )

        in_chs = channels
        for i, downsample_scale in enumerate(downsample_scales):
            out_chs = min(in_chs * downsample_scale, max_downsample_channels)

            model["layer_%d" % (i + 1)] = nn.Sequential(
                nn.Conv1d(
                    in_chs,
                    out_chs,
                    kernel_size=downsample_scale * 10 + 1,
                    stride=downsample_scale,
                    padding=downsample_scale * 5,
                    groups=in_chs // 4,
                ),
                nn.LeakyReLU(0.2, True),
            )
            in_chs = out_chs

        out_chs = min(in_chs * 2, max_downsample_channels)
        model["layer_%d" % (len(downsample_scales) + 2)] = nn.Sequential(
            nn.Conv1d(
                in_chs,
                out_chs,
                kernel_size=kernel_sizes[0],
                padding=(kernel_sizes[0] - 1) // 2,
            ),
            nn.LeakyReLU(0.2, True),
        )

        model["layer_%d" % (len(downsample_scales) + 3)] = nn.Conv1d(
            out_chs,
            out_channels,
            kernel_size=kernel_sizes[1],
            padding=(kernel_sizes[1] - 1) // 2,
        )

        self.model = model

    def forward(self, x):
        results = []
        for _, layer in self.model.items():
            x = layer(x)
            results.append(x)
        return results


class NLayerSpecDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_sizes=(5, 3),
        channels=32,
        max_downsample_channels=512,
        downsample_scales=(2, 2, 2),
    ):
        super().__init__()

        # check kernel size is valid
        assert kernel_sizes[0] % 2 == 1
        assert kernel_sizes[1] % 2 == 1

        model = nn.ModuleDict()

        model["layer_0"] = nn.Sequential(
            nn.Conv2d(
                in_channels,
                channels,
                kernel_size=kernel_sizes[0],
                stride=2,
                padding=kernel_sizes[0] // 2,
            ),
            nn.LeakyReLU(0.2, True),
        )

        in_chs = channels
        for i, downsample_scale in enumerate(downsample_scales):
            out_chs = min(in_chs * downsample_scale, max_downsample_channels)

            model[f"layer_{i + 1}"] = nn.Sequential(
                nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size=downsample_scale * 2 + 1,
                    stride=downsample_scale,
                    padding=downsample_scale,
                ),
                nn.LeakyReLU(0.2, True),
            )
            in_chs = out_chs

        out_chs = min(in_chs * 2, max_downsample_channels)
        model[f"layer_{len(downsample_scales) + 1}"] = nn.Sequential(
            nn.Conv2d(
                in_chs,
                out_chs,
                kernel_size=kernel_sizes[1],
                padding=kernel_sizes[1] // 2,
            ),
            nn.LeakyReLU(0.2, True),
        )

        model[f"layer_{len(downsample_scales) + 2}"] = nn.Conv2d(
            out_chs,
            out_channels,
            kernel_size=kernel_sizes[1],
            padding=kernel_sizes[1] // 2,
        )

        self.model = model

    def forward(self, x):
        results = []
        for _, layer in self.model.items():
            x = layer(x)
            results.append(x)
        return results
