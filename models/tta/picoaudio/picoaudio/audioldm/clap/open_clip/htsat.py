# Ke Chen
# knutchen@ucsd.edu
# HTS-AT: A HIERARCHICAL TOKEN-SEMANTIC AUDIO TRANSFORMER FOR SOUND CLASSIFICATION AND DETECTION
# Some layers designed on the model
# below codes are based and referred from https://github.com/microsoft/Swin-Transformer
# Swin Transformer for Computer Vision: https://arxiv.org/pdf/2103.14030.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import repeat
import collections.abc
import math
import warnings

from torch.nn.init import _calculate_fan_in_and_fan_out
import torch.utils.checkpoint as checkpoint

import random

from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

from itertools import repeat
from .utils import do_mixup, interpolate

from .feature_fusion import iAFF, AFF, DAF

# from PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        patch_stride=16,
        enable_fusion=False,
        fusion_type="None",
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patch_stride = to_2tuple(patch_stride)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.grid_size = (
            img_size[0] // patch_stride[0],
            img_size[1] // patch_stride[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.enable_fusion = enable_fusion
        self.fusion_type = fusion_type

        padding = (
            (patch_size[0] - patch_stride[0]) // 2,
            (patch_size[1] - patch_stride[1]) // 2,
        )

        if (self.enable_fusion) and (self.fusion_type == "channel_map"):
            self.proj = nn.Conv2d(
                in_chans * 4,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_stride,
                padding=padding,
            )
        else:
            self.proj = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_stride,
                padding=padding,
            )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        if (self.enable_fusion) and (
            self.fusion_type in ["daf_2d", "aff_2d", "iaff_2d"]
        ):
            self.mel_conv2d = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=(patch_size[0], patch_size[1] * 3),
                stride=(patch_stride[0], patch_stride[1] * 3),
                padding=padding,
            )
            if self.fusion_type == "daf_2d":
                self.fusion_model = DAF()
            elif self.fusion_type == "aff_2d":
                self.fusion_model = AFF(channels=embed_dim, type="2D")
            elif self.fusion_type == "iaff_2d":
                self.fusion_model = iAFF(channels=embed_dim, type="2D")

    def forward(self, x, longer_idx=None):
        if (self.enable_fusion) and (
            self.fusion_type in ["daf_2d", "aff_2d", "iaff_2d"]
        ):
            global_x = x[:, 0:1, :, :]

            # global processing
            B, C, H, W = global_x.shape
            assert (
                H == self.img_size[0] and W == self.img_size[1]
            ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
            global_x = self.proj(global_x)
            TW = global_x.size(-1)
            if len(longer_idx) > 0:
                # local processing
                local_x = x[longer_idx, 1:, :, :].contiguous()
                B, C, H, W = local_x.shape
                local_x = local_x.view(B * C, 1, H, W)
                local_x = self.mel_conv2d(local_x)
                local_x = local_x.view(
                    B, C, local_x.size(1), local_x.size(2), local_x.size(3)
                )
                local_x = local_x.permute((0, 2, 3, 1, 4)).contiguous().flatten(3)
                TB, TC, TH, _ = local_x.size()
                if local_x.size(-1) < TW:
                    local_x = torch.cat(
                        [
                            local_x,
                            torch.zeros(
                                (TB, TC, TH, TW - local_x.size(-1)),
                                device=global_x.device,
                            ),
                        ],
                        dim=-1,
                    )
                else:
                    local_x = local_x[:, :, :, :TW]

                global_x[longer_idx] = self.fusion_model(global_x[longer_idx], local_x)
            x = global_x
        else:
            B, C, H, W = x.shape
            assert (
                H == self.img_size[0] and W == self.img_size[1]
            ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
            x = self.proj(x)

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="normal"):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        # constant is stddev of standard normal truncated to (-2, 2)
        trunc_normal_(tensor, std=math.sqrt(variance) / 0.87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

    def extra_repr(self):
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"


# We use the model based on Swintransformer Block, therefore we can use the swin-transformer pretrained model
class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_before_mlp="ln",
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm_before_mlp = norm_before_mlp
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if self.norm_before_mlp == "ln":
            self.norm2 = nn.LayerNorm(dim)
        elif self.norm_before_mlp == "bn":
            self.norm2 = lambda x: nn.BatchNorm1d(dim)(x.transpose(1, 2)).transpose(
                1, 2
            )
        else:
            raise NotImplementedError
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(
                img_mask, self.window_size
            )  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        # pdb.set_trace()
        H, W = self.input_resolution
        # print("H: ", H)
        # print("W: ", W)
        # pdb.set_trace()
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows, attn = self.attn(
            x_windows, mask=self.attn_mask
        )  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, attn

    def extra_repr(self):
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self):
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        norm_before_mlp="ln",
    ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    norm_before_mlp=norm_before_mlp,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer
            )
        else:
            self.downsample = None

    def forward(self, x):
        attns = []
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x, attn = blk(x)
                if not self.training:
                    attns.append(attn.unsqueeze(0))
        if self.downsample is not None:
            x = self.downsample(x)
        if not self.training:
            attn = torch.cat(attns, dim=0)
            attn = torch.mean(attn, dim=0)
        return x, attn

    def extra_repr(self):
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


# The Core of HTSAT
class HTSAT_Swin_Transformer(nn.Module):
    r"""HTSAT based on the Swin Transformer
    Args:
        spec_size (int | tuple(int)): Input Spectrogram size. Default 256
        patch_size (int | tuple(int)): Patch size. Default: 4
        path_stride (iot | tuple(int)): Patch Stride for Frequency and Time Axis. Default: 4
        in_chans (int): Number of input image channels. Default: 1 (mono)
        num_classes (int): Number of classes for classification head. Default: 527
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each HTSAT-Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 8
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        config (module): The configuration Module from config.py
    """

    def __init__(
        self,
        spec_size=256,
        patch_size=4,
        patch_stride=(4, 4),
        in_chans=1,
        num_classes=527,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[4, 8, 16, 32],
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        norm_before_mlp="ln",
        config=None,
        enable_fusion=False,
        fusion_type="None",
        **kwargs,
    ):
        super(HTSAT_Swin_Transformer, self).__init__()

        self.config = config
        self.spec_size = spec_size
        self.patch_stride = patch_stride
        self.patch_size = patch_size
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.ape = ape
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_layers = len(self.depths)
        self.num_features = int(self.embed_dim * 2 ** (self.num_layers - 1))

        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate

        self.qkv_bias = qkv_bias
        self.qk_scale = None

        self.patch_norm = patch_norm
        self.norm_layer = norm_layer if self.patch_norm else None
        self.norm_before_mlp = norm_before_mlp
        self.mlp_ratio = mlp_ratio

        self.use_checkpoint = use_checkpoint

        self.enable_fusion = enable_fusion
        self.fusion_type = fusion_type

        #  process mel-spec ; used only once
        self.freq_ratio = self.spec_size // self.config.mel_bins
        window = "hann"
        center = True
        pad_mode = "reflect"
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.interpolate_ratio = 32  # Downsampled ratio
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=config.window_size,
            hop_length=config.hop_size,
            win_length=config.window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )
        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=config.sample_rate,
            n_fft=config.window_size,
            n_mels=config.mel_bins,
            fmin=config.fmin,
            fmax=config.fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True,
        )
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2,
        )  # 2 2
        self.bn0 = nn.BatchNorm2d(self.config.mel_bins)

        # split spctrogram into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=self.spec_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            norm_layer=self.norm_layer,
            patch_stride=patch_stride,
            enable_fusion=self.enable_fusion,
            fusion_type=self.fusion_type,
        )

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.grid_size
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, self.embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=self.drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(self.embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=self.depths[i_layer],
                num_heads=self.num_heads[i_layer],
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_scale=self.qk_scale,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=dpr[
                    sum(self.depths[:i_layer]) : sum(self.depths[: i_layer + 1])
                ],
                norm_layer=self.norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                norm_before_mlp=self.norm_before_mlp,
            )
            self.layers.append(layer)

        self.norm = self.norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)

        SF = (
            self.spec_size
            // (2 ** (len(self.depths) - 1))
            // self.patch_stride[0]
            // self.freq_ratio
        )
        self.tscam_conv = nn.Conv2d(
            in_channels=self.num_features,
            out_channels=self.num_classes,
            kernel_size=(SF, 3),
            padding=(0, 1),
        )
        self.head = nn.Linear(num_classes, num_classes)

        if (self.enable_fusion) and (
            self.fusion_type in ["daf_1d", "aff_1d", "iaff_1d"]
        ):
            self.mel_conv1d = nn.Sequential(
                nn.Conv1d(64, 64, kernel_size=5, stride=3, padding=2),
                nn.BatchNorm1d(64),
            )
            if self.fusion_type == "daf_1d":
                self.fusion_model = DAF()
            elif self.fusion_type == "aff_1d":
                self.fusion_model = AFF(channels=64, type="1D")
            elif self.fusion_type == "iaff_1d":
                self.fusion_model = iAFF(channels=64, type="1D")

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x, longer_idx=None):
        # A deprecated optimization for using a hierarchical output from different blocks

        frames_num = x.shape[2]
        x = self.patch_embed(x, longer_idx=longer_idx)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for i, layer in enumerate(self.layers):
            x, attn = layer(x)
        # for x
        x = self.norm(x)
        B, N, C = x.shape
        SF = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[0]
        ST = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[1]
        x = x.permute(0, 2, 1).contiguous().reshape(B, C, SF, ST)
        B, C, F, T = x.shape
        # group 2D CNN
        c_freq_bin = F // self.freq_ratio
        x = x.reshape(B, C, F // c_freq_bin, c_freq_bin, T)
        x = x.permute(0, 1, 3, 2, 4).contiguous().reshape(B, C, c_freq_bin, -1)
        # get latent_output
        fine_grained_latent_output = torch.mean(x, dim=2)
        fine_grained_latent_output = interpolate(
            fine_grained_latent_output.permute(0, 2, 1).contiguous(),
            8 * self.patch_stride[1],
        )

        latent_output = self.avgpool(torch.flatten(x, 2))
        latent_output = torch.flatten(latent_output, 1)

        # display the attention map, if needed

        x = self.tscam_conv(x)
        x = torch.flatten(x, 2)  # B, C, T

        fpx = interpolate(
            torch.sigmoid(x).permute(0, 2, 1).contiguous(), 8 * self.patch_stride[1]
        )

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        output_dict = {
            "framewise_output": fpx,  # already sigmoided
            "clipwise_output": torch.sigmoid(x),
            "fine_grained_embedding": fine_grained_latent_output,
            "embedding": latent_output,
        }

        return output_dict

    def crop_wav(self, x, crop_size, spe_pos=None):
        time_steps = x.shape[2]
        tx = torch.zeros(x.shape[0], x.shape[1], crop_size, x.shape[3]).to(x.device)
        for i in range(len(x)):
            if spe_pos is None:
                crop_pos = random.randint(0, time_steps - crop_size - 1)
            else:
                crop_pos = spe_pos
            tx[i][0] = x[i, 0, crop_pos : crop_pos + crop_size, :]
        return tx

    # Reshape the wavform to a img size, if you want to use the pretrained swin transformer model
    def reshape_wav2img(self, x):
        B, C, T, F = x.shape
        target_T = int(self.spec_size * self.freq_ratio)
        target_F = self.spec_size // self.freq_ratio
        assert (
            T <= target_T and F <= target_F
        ), "the wav size should less than or equal to the swin input size"
        # to avoid bicubic zero error
        if T < target_T:
            x = nn.functional.interpolate(
                x, (target_T, x.shape[3]), mode="bicubic", align_corners=True
            )
        if F < target_F:
            x = nn.functional.interpolate(
                x, (x.shape[2], target_F), mode="bicubic", align_corners=True
            )
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.reshape(
            x.shape[0],
            x.shape[1],
            x.shape[2],
            self.freq_ratio,
            x.shape[3] // self.freq_ratio,
        )
        # print(x.shape)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3], x.shape[4])
        return x

    # Repeat the wavform to a img size, if you want to use the pretrained swin transformer model
    def repeat_wat2img(self, x, cur_pos):
        B, C, T, F = x.shape
        target_T = int(self.spec_size * self.freq_ratio)
        target_F = self.spec_size // self.freq_ratio
        assert (
            T <= target_T and F <= target_F
        ), "the wav size should less than or equal to the swin input size"
        # to avoid bicubic zero error
        if T < target_T:
            x = nn.functional.interpolate(
                x, (target_T, x.shape[3]), mode="bicubic", align_corners=True
            )
        if F < target_F:
            x = nn.functional.interpolate(
                x, (x.shape[2], target_F), mode="bicubic", align_corners=True
            )
        x = x.permute(0, 1, 3, 2).contiguous()  # B C F T
        x = x[:, :, :, cur_pos : cur_pos + self.spec_size]
        x = x.repeat(repeats=(1, 1, 4, 1))
        return x

    def forward(
        self, x: torch.Tensor, mixup_lambda=None, infer_mode=False, device=None
    ):  # out_feat_keys: List[str] = None):

        if self.enable_fusion and x["longer"].sum() == 0:
            # if no audio is longer than 10s, then randomly select one audio to be longer
            x["longer"][torch.randint(0, x["longer"].shape[0], (1,))] = True

        if not self.enable_fusion:
            x = x["waveform"].to(device=device, non_blocking=True)
            x = self.spectrogram_extractor(x)  # (batch_size, 1, time_steps, freq_bins)
            x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)
            if self.training:
                x = self.spec_augmenter(x)

            if self.training and mixup_lambda is not None:
                x = do_mixup(x, mixup_lambda)

            x = self.reshape_wav2img(x)
            output_dict = self.forward_features(x)
        else:
            longer_list = x["longer"].to(device=device, non_blocking=True)
            x = x["mel_fusion"].to(device=device, non_blocking=True)
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)
            longer_list_idx = torch.where(longer_list)[0]
            if self.fusion_type in ["daf_1d", "aff_1d", "iaff_1d"]:
                new_x = x[:, 0:1, :, :].clone().contiguous()
                if len(longer_list_idx) > 0:
                    # local processing
                    fusion_x_local = x[longer_list_idx, 1:, :, :].clone().contiguous()
                    FB, FC, FT, FF = fusion_x_local.size()
                    fusion_x_local = fusion_x_local.view(FB * FC, FT, FF)
                    fusion_x_local = torch.permute(
                        fusion_x_local, (0, 2, 1)
                    ).contiguous()
                    fusion_x_local = self.mel_conv1d(fusion_x_local)
                    fusion_x_local = fusion_x_local.view(
                        FB, FC, FF, fusion_x_local.size(-1)
                    )
                    fusion_x_local = (
                        torch.permute(fusion_x_local, (0, 2, 1, 3))
                        .contiguous()
                        .flatten(2)
                    )
                    if fusion_x_local.size(-1) < FT:
                        fusion_x_local = torch.cat(
                            [
                                fusion_x_local,
                                torch.zeros(
                                    (FB, FF, FT - fusion_x_local.size(-1)),
                                    device=device,
                                ),
                            ],
                            dim=-1,
                        )
                    else:
                        fusion_x_local = fusion_x_local[:, :, :FT]
                    # 1D fusion
                    new_x = new_x.squeeze(1).permute((0, 2, 1)).contiguous()
                    new_x[longer_list_idx] = self.fusion_model(
                        new_x[longer_list_idx], fusion_x_local
                    )
                    x = new_x.permute((0, 2, 1)).contiguous()[:, None, :, :]
                else:
                    x = new_x

            elif self.fusion_type in ["daf_2d", "aff_2d", "iaff_2d", "channel_map"]:
                x = x  # no change

            if self.training:
                x = self.spec_augmenter(x)
            if self.training and mixup_lambda is not None:
                x = do_mixup(x, mixup_lambda)

            x = self.reshape_wav2img(x)
            output_dict = self.forward_features(x, longer_idx=longer_list_idx)

        # if infer_mode:
        #     # in infer mode. we need to handle different length audio input
        #     frame_num = x.shape[2]
        #     target_T = int(self.spec_size * self.freq_ratio)
        #     repeat_ratio = math.floor(target_T / frame_num)
        #     x = x.repeat(repeats=(1,1,repeat_ratio,1))
        #     x = self.reshape_wav2img(x)
        #     output_dict = self.forward_features(x)
        # else:
        #     if x.shape[2] > self.freq_ratio * self.spec_size:
        #         if self.training:
        #             x = self.crop_wav(x, crop_size=self.freq_ratio * self.spec_size)
        #             x = self.reshape_wav2img(x)
        #             output_dict = self.forward_features(x)
        #         else:
        #             # Change: Hard code here
        #             overlap_size = (x.shape[2] - 1) // 4
        #             output_dicts = []
        #             crop_size = (x.shape[2] - 1) // 2
        #             for cur_pos in range(0, x.shape[2] - crop_size - 1, overlap_size):
        #                 tx = self.crop_wav(x, crop_size = crop_size, spe_pos = cur_pos)
        #                 tx = self.reshape_wav2img(tx)
        #                 output_dicts.append(self.forward_features(tx))
        #             clipwise_output = torch.zeros_like(output_dicts[0]["clipwise_output"]).float().to(x.device)
        #             framewise_output = torch.zeros_like(output_dicts[0]["framewise_output"]).float().to(x.device)
        #             for d in output_dicts:
        #                 clipwise_output += d["clipwise_output"]
        #                 framewise_output += d["framewise_output"]
        #             clipwise_output  = clipwise_output / len(output_dicts)
        #             framewise_output = framewise_output / len(output_dicts)
        #             output_dict = {
        #                 'framewise_output': framewise_output,
        #                 'clipwise_output': clipwise_output
        #             }
        #     else: # this part is typically used, and most easy one
        #         x = self.reshape_wav2img(x)
        #         output_dict = self.forward_features(x)
        # x = self.head(x)

        # We process the data in the dataloader part, in that here we only consider the input_T < fixed_T

        return output_dict


def create_htsat_model(audio_cfg, enable_fusion=False, fusion_type="None"):
    try:

        assert audio_cfg.model_name in [
            "tiny",
            "base",
            "large",
        ], "model name for HTS-AT is wrong!"
        if audio_cfg.model_name == "tiny":
            model = HTSAT_Swin_Transformer(
                spec_size=256,
                patch_size=4,
                patch_stride=(4, 4),
                num_classes=audio_cfg.class_num,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[4, 8, 16, 32],
                window_size=8,
                config=audio_cfg,
                enable_fusion=enable_fusion,
                fusion_type=fusion_type,
            )
        elif audio_cfg.model_name == "base":
            model = HTSAT_Swin_Transformer(
                spec_size=256,
                patch_size=4,
                patch_stride=(4, 4),
                num_classes=audio_cfg.class_num,
                embed_dim=128,
                depths=[2, 2, 12, 2],
                num_heads=[4, 8, 16, 32],
                window_size=8,
                config=audio_cfg,
                enable_fusion=enable_fusion,
                fusion_type=fusion_type,
            )
        elif audio_cfg.model_name == "large":
            model = HTSAT_Swin_Transformer(
                spec_size=256,
                patch_size=4,
                patch_stride=(4, 4),
                num_classes=audio_cfg.class_num,
                embed_dim=256,
                depths=[2, 2, 12, 2],
                num_heads=[4, 8, 16, 32],
                window_size=8,
                config=audio_cfg,
                enable_fusion=enable_fusion,
                fusion_type=fusion_type,
            )

        return model
    except:
        raise RuntimeError(
            f"Import Model for {audio_cfg.model_name} not found, or the audio cfg parameters are not enough."
        )
