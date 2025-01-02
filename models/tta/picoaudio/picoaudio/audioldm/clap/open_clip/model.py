""" CLAP Model

Adapted from CLIP: https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
Adapted to the Audio Task.
"""

from collections import OrderedDict
from dataclasses import dataclass
from email.mime import audio
from typing import Tuple, Union, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .timm_model import TimmModel
import logging
from .utils import freeze_batch_norm_2d

from .pann_model import create_pann_model
from .htsat import create_htsat_model
from transformers import BertModel, RobertaModel, BartModel
from transformers.tokenization_utils_base import BatchEncoding


class MLPLayers(nn.Module):
    def __init__(self, units=[512, 512, 512], nonlin=nn.ReLU(), dropout=0.1):
        super(MLPLayers, self).__init__()
        self.nonlin = nonlin
        self.dropout = dropout

        sequence = []
        for u0, u1 in zip(units[:-1], units[1:]):
            sequence.append(nn.Linear(u0, u1))
            sequence.append(self.nonlin)
            sequence.append(nn.Dropout(self.dropout))
        sequence = sequence[:-2]

        self.sequential = nn.Sequential(*sequence)

    def forward(self, X):
        X = self.sequential(X)
        return X


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        ("-1", nn.AvgPool2d(stride)),
                        (
                            "0",
                            nn.Conv2d(
                                inplanes,
                                planes * self.expansion,
                                1,
                                stride=1,
                                bias=False,
                            ),
                        ),
                        ("1", nn.BatchNorm2d(planes * self.expansion)),
                    ]
                )
            )

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(
        self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(
            2, 0, 1
        )  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, image_size=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.image_size = image_size

        # the 3-layer stem
        self.conv1 = nn.Conv2d(
            3, width // 2, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(
            width // 2, width // 2, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(image_size // 32, embed_dim, heads, output_dim)

        self.init_parameters()

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def init_parameters(self):
        if self.attnpool is not None:
            std = self.attnpool.c_proj.in_features**-0.5
            nn.init.normal_(self.attnpool.q_proj.weight, std=std)
            nn.init.normal_(self.attnpool.k_proj.weight, std=std)
            nn.init.normal_(self.attnpool.v_proj.weight, std=std)
            nn.init.normal_(self.attnpool.c_proj.weight, std=std)

        for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for name, param in resnet_block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert (
            unlocked_groups == 0
        ), "partial locking not currently supported for this model"
        for param in self.parameters():
            param.requires_grad = False
        if freeze_bn_stats:
            freeze_batch_norm_2d(self)

    def stem(self, x):
        for conv, bn in [
            (self.conv1, self.bn1),
            (self.conv2, self.bn2),
            (self.conv3, self.bn3),
        ]:
            x = self.relu(bn(conv(x)))
        x = self.avgpool(x)
        return x

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, act_layer: Callable = nn.GELU):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self, width: int, layers: int, heads: int, act_layer: Callable = nn.GELU
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(width, heads, act_layer=act_layer)
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            x = r(x, attn_mask=attn_mask)
        return x


class VisualTransformer(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        act_layer: Callable = nn.GELU,
    ):
        super().__init__()
        self.image_size = image_size
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((image_size // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = LayerNorm(width)

        self.text_branch = Transformer(width, layers, heads, act_layer=act_layer)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert (
            unlocked_groups == 0
        ), "partial locking not currently supported for this model"
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_branch(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


@dataclass
class CLAPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    timm_model_name: str = (
        None  # a valid model name overrides layers, width, patch_size
    )
    timm_model_pretrained: bool = (
        False  # use (imagenet) pretrained weights for named model
    )
    timm_pool: str = (
        "avg"  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    )
    timm_proj: str = (
        "linear"  # linear projection for timm model output ('linear', 'mlp', '')
    )


# Audio Config Class
@dataclass
class CLAPAudioCfp:
    model_type: str = "PANN"
    model_name: str = "Cnn14"
    sample_rate: int = 48000
    # Param
    audio_length: int = 1024
    window_size: int = 1024
    hop_size: int = 1024
    fmin: int = 50
    fmax: int = 14000
    class_num: int = 527
    mel_bins: int = 64
    clip_samples: int = 480000


@dataclass
class CLAPTextCfg:
    context_length: int
    vocab_size: int
    width: int
    heads: int
    layers: int
    model_type: str


class CLAP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        audio_cfg: CLAPAudioCfp,
        text_cfg: CLAPTextCfg,
        quick_gelu: bool = False,
        enable_fusion: bool = False,
        fusion_type: str = "None",
        joint_embed_shape: int = 512,
        mlp_act: str = "relu",
    ):
        super().__init__()
        if isinstance(audio_cfg, dict):
            audio_cfg = CLAPAudioCfp(**audio_cfg)
        if isinstance(text_cfg, dict):
            text_cfg = CLAPTextCfg(**text_cfg)

        self.audio_cfg = audio_cfg
        self.text_cfg = text_cfg
        self.enable_fusion = enable_fusion
        self.fusion_type = fusion_type
        self.joint_embed_shape = joint_embed_shape
        self.mlp_act = mlp_act

        self.context_length = text_cfg.context_length

        # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
        # memory efficient in recent PyTorch releases (>= 1.10).
        # NOTE: timm models always use native GELU regardless of quick_gelu flag.
        act_layer = QuickGELU if quick_gelu else nn.GELU

        if mlp_act == "relu":
            mlp_act_layer = nn.ReLU()
        elif mlp_act == "gelu":
            mlp_act_layer = nn.GELU()
        else:
            raise NotImplementedError

        # audio branch
        # audio branch parameters
        if audio_cfg.model_type == "PANN":
            self.audio_branch = create_pann_model(audio_cfg, enable_fusion, fusion_type)
        elif audio_cfg.model_type == "HTSAT":
            self.audio_branch = create_htsat_model(
                audio_cfg, enable_fusion, fusion_type
            )
        else:
            logging.error(f"Model config for {audio_cfg.model_type} not found")
            raise RuntimeError(f"Model config for {audio_cfg.model_type} not found.")

        # text branch
        # text branch parameters
        if text_cfg.model_type == "transformer":
            self.text_branch = Transformer(
                width=text_cfg.width,
                layers=text_cfg.layers,
                heads=text_cfg.heads,
                act_layer=act_layer,
            )
            self.vocab_size = text_cfg.vocab_size
            self.token_embedding = nn.Embedding(text_cfg.vocab_size, text_cfg.width)
            self.positional_embedding = nn.Parameter(
                torch.empty(self.context_length, text_cfg.width)
            )
            self.ln_final = LayerNorm(text_cfg.width)
            self.text_transform = MLPLayers(
                units=[
                    self.joint_embed_shape,
                    self.joint_embed_shape,
                    self.joint_embed_shape,
                ],
                dropout=0.1,
            )
            self.text_projection = nn.Sequential(
                nn.Linear(text_cfg.width, self.joint_embed_shape),
                mlp_act_layer,
                nn.Linear(self.joint_embed_shape, self.joint_embed_shape),
            )
        elif text_cfg.model_type == "bert":
            self.text_branch = BertModel.from_pretrained("bert-base-uncased")
            self.text_transform = MLPLayers(
                units=[
                    self.joint_embed_shape,
                    self.joint_embed_shape,
                    self.joint_embed_shape,
                ],
                dropout=0.1,
            )
            self.text_projection = nn.Sequential(
                nn.Linear(768, self.joint_embed_shape),
                mlp_act_layer,
                nn.Linear(self.joint_embed_shape, self.joint_embed_shape),
            )
        elif text_cfg.model_type == "roberta":
            self.text_branch = RobertaModel.from_pretrained("roberta-base")
            self.text_transform = MLPLayers(
                units=[
                    self.joint_embed_shape,
                    self.joint_embed_shape,
                    self.joint_embed_shape,
                ],
                dropout=0.1,
            )
            self.text_projection = nn.Sequential(
                nn.Linear(768, self.joint_embed_shape),
                mlp_act_layer,
                nn.Linear(self.joint_embed_shape, self.joint_embed_shape),
            )
        elif text_cfg.model_type == "bart":
            self.text_branch = BartModel.from_pretrained("facebook/bart-base")
            self.text_transform = MLPLayers(
                units=[
                    self.joint_embed_shape,
                    self.joint_embed_shape,
                    self.joint_embed_shape,
                ],
                dropout=0.1,
            )
            self.text_projection = nn.Sequential(
                nn.Linear(768, self.joint_embed_shape),
                mlp_act_layer,
                nn.Linear(self.joint_embed_shape, self.joint_embed_shape),
            )
        else:
            logging.error(f"Model config for {text_cfg.model_type} not found")
            raise RuntimeError(f"Model config for {text_cfg.model_type} not found.")
        self.text_branch_type = text_cfg.model_type
        # text branch parameters

        # audio branch parameters
        self.audio_transform = MLPLayers(
            units=[
                self.joint_embed_shape,
                self.joint_embed_shape,
                self.joint_embed_shape,
            ],
            dropout=0.1,
        )

        # below here is text branch parameters

        # ============================================================================================================
        self.audio_projection = nn.Sequential(
            nn.Linear(embed_dim, self.joint_embed_shape),
            mlp_act_layer,
            nn.Linear(self.joint_embed_shape, self.joint_embed_shape),
        )

        self.logit_scale_a = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_t = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.register_buffer("attn_mask", self.build_attention_mask(), persistent=False)

        self.init_text_branch_parameters()

    def init_text_branch_parameters(self):
        if self.text_branch_type == "transformer":
            nn.init.normal_(self.token_embedding.weight, std=0.02)
            nn.init.normal_(self.positional_embedding, std=0.01)
            proj_std = (self.text_branch.width**-0.5) * (
                (2 * self.text_branch.layers) ** -0.5
            )
            attn_std = self.text_branch.width**-0.5
            fc_std = (2 * self.text_branch.width) ** -0.5
            for block in self.text_branch.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        if self.text_branch_type == "bert" or self.text_branch_type == "roberta":
            width = self.text_branch.embeddings.word_embeddings.weight.shape[-1]
        elif self.text_branch_type == "bart":
            width = self.text_branch.shared.weight.shape[-1]
        else:
            width = self.text_branch.width
        nn.init.constant_(self.logit_scale_a, np.log(1 / 0.07))
        nn.init.constant_(self.logit_scale_t, np.log(1 / 0.07))

        # deprecated
        # if hasattr(self.visual, 'init_parameters'):
        # self.visual.init_parameters()

        # if self.text_projection is not None:
        #     nn.init.normal_(self.text_projection, std=width**-0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_audio(self, audio, device):
        return self.audio_branch(
            audio, mixup_lambda=None, device=device
        )  # mix lambda needs to add

    # def list_of_dict_of_tensor2dict_of_tensor(self, x, device):
    #     tmp = {}
    #     for k in x[0].keys():
    #         tmp[k] = []
    #         for i in range(len(x)):
    #             tmp[k].append(x[i][k][:77])
    #     for k in x[0].keys():
    #         tmp[k] = torch.tensor(tmp[k]).to(device=device, non_blocking=True)
    #     return tmp

    def encode_text(self, text, device):
        if self.text_branch_type == "transformer":
            text = text.to(device=device, non_blocking=True)
            x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

            x = x + self.positional_embedding
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.text_branch(x, attn_mask=self.attn_mask)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x)

            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = self.text_projection(x[torch.arange(x.shape[0]), text.argmax(dim=-1)])
        elif self.text_branch_type == "bert":
            # text = self.list_of_dict_of_tensor2dict_of_tensor(text, device)
            # text = BatchEncoding(text)
            x = self.text_branch(
                input_ids=text["input_ids"].to(device=device, non_blocking=True),
                attention_mask=text["attention_mask"].to(
                    device=device, non_blocking=True
                ),
                token_type_ids=text["token_type_ids"].to(
                    device=device, non_blocking=True
                ),
            )["pooler_output"]
            x = self.text_projection(x)
        elif self.text_branch_type == "roberta":
            x = self.text_branch(
                input_ids=text["input_ids"].to(device=device, non_blocking=True),
                attention_mask=text["attention_mask"].to(
                    device=device, non_blocking=True
                ),
            )["pooler_output"]
            x = self.text_projection(x)
        elif self.text_branch_type == "bart":
            x = torch.mean(
                self.text_branch(
                    input_ids=text["input_ids"].to(device=device, non_blocking=True),
                    attention_mask=text["attention_mask"].to(
                        device=device, non_blocking=True
                    ),
                )["encoder_last_hidden_state"],
                axis=1,
            )
            x = self.text_projection(x)
        else:
            logging.error(f"Model type {self.text_branch_type} not found")
            raise RuntimeError(f"Model type {self.text_branch_type} not found.")
        return x

    def forward(self, audio, text, device=None):
        """Forward audio and text into the CLAP

        Parameters
        ----------
        audio: torch.Tensor (batch_size, audio_length)
            the time-domain audio input / the batch of mel_spec and longer list.
        text: torch.Tensor () // need to add
            the text token input
        """
        if device is None:
            if audio is not None:
                device = audio.device
            elif text is not None:
                device = text.device
        if audio is None and text is None:
            # a hack to get the logit scale
            return self.logit_scale_a.exp(), self.logit_scale_t.exp()
        elif audio is None:
            return self.encode_text(text, device=device)
        elif text is None:
            return self.audio_projection(
                self.encode_audio(audio, device=device)["embedding"]
            )
        audio_features = self.audio_projection(
            self.encode_audio(audio, device=device)["embedding"]
        )
        audio_features = F.normalize(audio_features, dim=-1)

        text_features = self.encode_text(text, device=device)
        # print("text_features", text_features)
        # print("text_features.shape", text_features.shape)
        # print("text_features.type", type(text_features))
        text_features = F.normalize(text_features, dim=-1)

        audio_features_mlp = self.audio_transform(audio_features)
        text_features_mlp = self.text_transform(text_features)
        # Four outputs: audio features (basic & MLP), text features (basic & MLP)
        return (
            audio_features,
            text_features,
            audio_features_mlp,
            text_features_mlp,
            self.logit_scale_a.exp(),
            self.logit_scale_t.exp(),
        )

    def get_logit_scale(self):
        return self.logit_scale_a.exp(), self.logit_scale_t.exp()

    def get_text_embedding(self, data):
        """Get the text embedding from the model

        Parameters
        ----------
        data: torch.Tensor
            a tensor of text embedding

        Returns
        ----------
        text_embed: torch.Tensor
            a tensor of text_embeds (N, D)

        """
        device = next(self.parameters()).device
        for k in data:
            data[k] = data[k].to(device)
            if len(data[k].size()) < 2:
                data[k] = data[k].unsqueeze(0)
        text_embeds = self.encode_text(data, device=device)
        text_embeds = F.normalize(text_embeds, dim=-1)

        return text_embeds

    def get_audio_embedding(self, data):
        """Get the audio embedding from the model

        Parameters
        ----------
        data: a list of dict
            the audio input dict list from 'get_audio_feature' method

        Returns
        ----------
        audio_embed: torch.Tensor
            a tensor of audio_embeds (N, D)

        """
        device = next(self.parameters()).device
        input_dict = {}
        keys = data[0].keys()
        for k in keys:
            input_dict[k] = torch.cat([d[k].unsqueeze(0) for d in data], dim=0).to(
                device
            )

        audio_embeds = self.audio_projection(
            self.encode_audio(input_dict, device=device)["embedding"]
        )
        audio_embeds = F.normalize(audio_embeds, dim=-1)

        return audio_embeds

    def audio_infer(self, audio, hopsize=None, device=None):
        """Forward one audio and produce the audio embedding

        Parameters
        ----------
        audio:  (audio_length)
            the time-domain audio input, notice that it must be only one input
        hopsize: int
            the overlap hopsize as the sliding window

        Returns
        ----------
        output_dict: {
            key: [n, (embedding_shape)] if "HTS-AT"
            or
            key: [(embedding_shape)] if "PANN"
        }
            the list of key values of the audio branch

        """

        assert not self.training, "the inference mode must be run at eval stage"
        output_dict = {}
        # PANN
        if self.audio_cfg.model_type == "PANN":
            audio_input = audio.unsqueeze(dim=0)
            output_dict[key] = self.encode_audio(audio_input, device=device)[
                key
            ].squeeze(dim=0)
        elif self.audio_cfg.model_type == "HTSAT":
            # repeat
            audio_len = len(audio)
            k = self.audio_cfg.clip_samples // audio_len
            if k > 1:
                audio = audio.repeat(k)
                audio_len = len(audio)

            if hopsize is None:
                hopsize = min(hopsize, audio_len)

            if audio_len > self.audio_cfg.clip_samples:
                audio_input = [
                    audio[pos : pos + self.audio_cfg.clip_samples].clone()
                    for pos in range(
                        0, audio_len - self.audio_cfg.clip_samples, hopsize
                    )
                ]
                audio_input.append(audio[-self.audio_cfg.clip_samples :].clone())
                audio_input = torch.stack(audio_input)
                output_dict[key] = self.encode_audio(audio_input, device=device)[key]
            else:
                audio_input = audio.unsqueeze(dim=0)
                output_dict[key] = self.encode_audio(audio_input, device=device)[
                    key
                ].squeeze(dim=0)

        return output_dict


def convert_weights_to_fp16(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [
                *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                "in_proj_bias",
                "bias_k",
                "bias_v",
            ]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


# Ignore the state dict of the vision part
def build_model_from_openai_state_dict(
    state_dict: dict, model_cfg, enable_fusion: bool = False, fusion_type: str = "None"
):

    embed_dim = model_cfg["embed_dim"]
    audio_cfg = model_cfg["audio_cfg"]
    text_cfg = model_cfg["text_cfg"]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(
        set(
            k.split(".")[2]
            for k in state_dict
            if k.startswith(f"transformer.resblocks")
        )
    )

    audio_cfg = CLAPAudioCfp(**audio_cfg)
    text_cfg = CLAPTextCfg(**text_cfg)

    model = CLAP(
        embed_dim,
        audio_cfg=audio_cfg,
        text_cfg=text_cfg,
        quick_gelu=True,  # OpenAI models were trained with QuickGELU
        enable_fusion=enable_fusion,
        fusion_type=fusion_type,
    )
    state_dict["logit_scale_a"] = state_dict["logit_scale"]
    state_dict["logit_scale_t"] = state_dict["logit_scale"]
    pop_keys = list(state_dict.keys())[::]
    # pop the visual branch saved weights
    for key in pop_keys:
        if key.startswith("visual."):
            state_dict.pop(key, None)

    for key in ["logit_scale", "input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)

    # not use fp16
    # convert_weights_to_fp16(model)
    model.load_state_dict(state_dict, strict=False)
    return model.eval()


def trace_model(model, batch_size=256, device=torch.device("cpu")):
    model.eval()
    audio_length = model.audio_cfg.audio_length
    example_audio = torch.ones((batch_size, audio_length), device=device)
    example_text = torch.zeros(
        (batch_size, model.context_length), dtype=torch.int, device=device
    )
    model = torch.jit.trace_module(
        model,
        inputs=dict(
            forward=(example_audio, example_text),
            encode_text=(example_text,),
            encode_image=(example_audio,),
        ),
    )
    model.audio_cfg.audio_length = audio_length  # Question: what does this do?
    return model
