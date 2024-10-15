"""VGG2L module definition for transformer encoder."""

from typing import Tuple
from typing import Union

import torch


class VGG2L(torch.nn.Module):
    """VGG2L module for custom encoder.

    Args:
        idim: Dimension of inputs
        odim: Dimension of outputs
        pos_enc: Positional encoding class

    """

    def __init__(self, idim: int, odim: int, pos_enc: torch.nn.Module = None):
        """Construct a VGG2L object."""
        super().__init__()

        self.vgg2l = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((3, 2)),
            torch.nn.Conv2d(64, 128, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),
        )

        if pos_enc is not None:
            self.output = torch.nn.Sequential(
                torch.nn.Linear(128 * ((idim // 2) // 2), odim), pos_enc
            )
        else:
            self.output = torch.nn.Linear(128 * ((idim // 2) // 2), odim)

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
    ]:
        """VGG2L forward for x.

        Args:
            x: Input tensor (B, T, idim)
            x_mask: Input mask (B, 1, T)

        Returns:
            x: Output tensor (B, sub(T), odim)
                   or ((B, sub(T), odim), (B, sub(T), att_dim))
            x_mask: Output mask (B, 1, sub(T))

        """
        x = x.unsqueeze(1)
        x = self.vgg2l(x)

        b, c, t, f = x.size()

        x = self.output(x.transpose(1, 2).contiguous().view(b, t, c * f))

        if x_mask is not None:
            x_mask = self.create_new_mask(x_mask)

        return x, x_mask

    def create_new_mask(self, x_mask: torch.Tensor) -> torch.Tensor:
        """Create a subsampled version of x_mask.

        Args:
            x_mask: Input mask (B, 1, T)

        Returns:
            x_mask: Output mask (B, 1, sub(T))

        """
        x_t1 = x_mask.size(2) - (x_mask.size(2) % 3)
        x_mask = x_mask[:, :, :x_t1][:, :, ::3]

        x_t2 = x_mask.size(2) - (x_mask.size(2) % 2)
        x_mask = x_mask[:, :, :x_t2][:, :, ::2]

        return x_mask
