import numpy as np
import torch.nn.functional as F
from torch import nn
from .model import MLPLayers


class LinearProbe(nn.Module):
    def __init__(self, model, mlp, freeze, in_ch, out_ch, act=None):
        """
        Args:
            model: nn.Module
            mlp: bool, if True, then use the MLP layer as the linear probe module
            freeze: bool, if Ture, then freeze all the CLAP model's layers when training the linear probe
            in_ch: int, the output channel from CLAP model
            out_ch: int, the output channel from linear probe (class_num)
            act: torch.nn.functional, the activation function before the loss function
        """
        super().__init__()
        in_ch = 512
        self.clap_model = model
        self.clap_model.text_branch = None  # to save memory
        self.freeze = freeze
        if mlp:
            self.lp_layer = MLPLayers(units=[in_ch, in_ch * 2, out_ch])
        else:
            self.lp_layer = nn.Linear(in_ch, out_ch)

        if self.freeze:
            for param in self.clap_model.parameters():
                param.requires_grad = False

        if act == "None":
            self.act = None
        elif act == "relu":
            self.act = nn.ReLU()
        elif act == "elu":
            self.act = nn.ELU()
        elif act == "prelu":
            self.act = nn.PReLU(num_parameters=in_ch)
        elif act == "softmax":
            self.act = nn.Softmax(dim=-1)
        elif act == "sigmoid":
            self.act = nn.Sigmoid()

    def forward(self, x, mix_lambda=None, device=None):
        """
        Args:
            x: waveform, torch.tensor [batch, t_samples] / batch of mel_spec and longer list
            mix_lambda: torch.tensor [batch], the mixup lambda
        Returns:
            class_prob: torch.tensor [batch, class_num]

        """
        # batchnorm cancel grandient
        if self.freeze:
            self.clap_model.eval()

        x = self.clap_model.audio_projection(
            self.clap_model.audio_branch(x, mixup_lambda=mix_lambda, device=device)[
                "embedding"
            ]
        )
        out = self.lp_layer(x)
        if self.act is not None:
            out = self.act(out)
        return out
