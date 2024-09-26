import math
import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from einops import rearrange

from torch.utils.checkpoint import checkpoint as checkpoint_fn
from models.tts.UniCATS.CTXtxt2vec.build_model.modeling.transformers.espnet_nets.transformer.embedding import ScaledPositionalEncoding


class FullAttention(nn.Module):
    def __init__(self,
                 n_embd,  # the embed dim
                 n_head,  # the number of heads
                 attn_pdrop=0.1,  # attention dropout prob
                 resid_pdrop=0.1,  # residual attention dropout prob
                 causal=True,
                 ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
        self.causal = causal

    def forward(self, x, encoder_output, mask=None):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)

        if mask is None:
            mask = att.new(B, 1, T).zero_() == 1
        mask = mask.unsqueeze(1)  # (B, 1, 1, T)
        min_value = torch.finfo(att.dtype).min
        att = att.masked_fill(mask, min_value)

        att = F.softmax(att, dim=-1).masked_fill(mask, 0.0)  # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False)  # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att


class CrossAttention(nn.Module):
    def __init__(self,
                 n_embd,  # the embed dim
                 condition_embd,  # condition dim
                 n_head,  # the number of heads
                 seq_len=None,  # the max length of sequence
                 attn_pdrop=0.1,  # attention dropout prob
                 resid_pdrop=0.1,  # residual attention dropout prob
                 causal=True,
                 ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)

        self.n_head = n_head
        self.causal = causal

    def forward(self, x, encoder_output, mask=None):
        B, T, C = x.size()
        B, T_E, _ = encoder_output.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)
        if mask is None:
            mask = att.new(B, 1, T)
        mask = mask.unsqueeze(1)  # (B, 1, 1, T)
        min_value = torch.finfo(att.dtype).min
        att = att.masked_fill(mask, min_value)

        att = F.softmax(att, dim=-1).masked_fill(mask, 0.0)  # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False)  # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att


class GELU2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, num_steps, dim, rescale_steps=4000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x):
        x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd, diffusion_step, emb_type="adalayernorm_abs"):
        super().__init__()
        if "abs" in emb_type:
            self.emb = SinusoidalPosEmb(diffusion_step, n_embd)
        else:
            self.emb = nn.Embedding(diffusion_step, n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x, timestep):
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x


class AdaInsNorm(nn.Module):
    def __init__(self, n_embd, diffusion_step, emb_type="adainsnorm_abs"):
        super().__init__()
        if "abs" in emb_type:
            self.emb = SinusoidalPosEmb(diffusion_step, n_embd)
        else:
            self.emb = nn.Embedding(diffusion_step, n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)
        self.instancenorm = nn.InstanceNorm1d(n_embd)

    def forward(self, x, timestep):
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.instancenorm(x.transpose(-1, -2)).transpose(-1, -2) * (1 + scale) + shift
        return x


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self,
                 class_type='adalayernorm',
                 class_number=1000,
                 n_embd=1024,
                 n_head=16,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 mlp_hidden_times=4,
                 activate='GELU',
                 attn_type='full',
                 if_upsample=False,
                 upsample_type='bilinear',
                 upsample_pre_channel=0,
                 conv_attn_kernel_size=None,  # only need for dalle_conv attention
                 condition_dim=1024,
                 diffusion_step=100,
                 timestep_type='adalayernorm',
                 window_size=8,
                 mlp_type='fc',
                 ):
        super().__init__()
        self.if_upsample = if_upsample
        self.attn_type = attn_type

        if attn_type in ['selfcross', 'selfcondition', 'self']:
            if 'adalayernorm' in timestep_type:
                self.ln1 = AdaLayerNorm(n_embd, diffusion_step, timestep_type)
            else:
                print("timestep_type wrong")
        else:
            self.ln1 = nn.LayerNorm(n_embd)

        self.ln2 = nn.LayerNorm(n_embd)
        # self.if_selfcross = False
        if attn_type in ['self', 'selfcondition']:
            self.attn = FullAttention(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
            )
            if attn_type == 'selfcondition':
                if 'adalayernorm' in class_type:
                    self.ln2 = AdaLayerNorm(n_embd, class_number, class_type)
                else:
                    self.ln2 = AdaInsNorm(n_embd, class_number, class_type)
        elif attn_type == 'selfcross':
            self.attn1 = FullAttention(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
            )

            self.attn2 = nn.Linear(condition_dim, n_embd)

        else:
            print("attn_type error")
        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()
        if mlp_type == 'conv_mlp':
            self.mlp = Conv_MLP(n_embd, mlp_hidden_times, act, resid_pdrop)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(n_embd, mlp_hidden_times * n_embd),
                act,
                nn.Linear(mlp_hidden_times * n_embd, n_embd),
                nn.Dropout(resid_pdrop),
            )

    def forward_bak(self, x, encoder_output, timestep, mask=None):
        if self.attn_type == "selfcross":
            a, att = self.attn1(self.ln1(x, timestep), encoder_output, mask=mask)
            x = x + a
            a, att = self.attn2(self.ln1_1(x, timestep), encoder_output, mask=mask)
            x = x + a
        elif self.attn_type == "selfcondition":
            a, att = self.attn(self.ln1(x, timestep), encoder_output, mask=mask)
            x = x + a
            x = x + self.mlp(self.ln2(x, encoder_output.long()))  # only one really use encoder_output
            return x, att
        else:  # 'self'
            a, att = self.attn(self.ln1(x, timestep), encoder_output, mask=mask)
            x = x + a

        x = x + self.mlp(self.ln2(x))

        return x, att

    def forward(self, x, encoder_output, timestep, mask=None):
        if self.attn_type == "selfcross":
            a, att = self.attn1(self.ln1(x, timestep), encoder_output, mask=mask)
            x = x + a
            # a, att = self.attn2(self.ln1_1(x, timestep), encoder_output, mask=mask)
            a = self.attn2(encoder_output)
            x = x + a
        elif self.attn_type == "selfcondition":
            a, att = self.attn(self.ln1(x, timestep), encoder_output, mask=mask)
            x = x + a
            x = x + self.mlp(self.ln2(x, encoder_output.long()))  # only one really use encoder_output
            return x, att
        else:  # 'self'
            a, att = self.attn(self.ln1(x, timestep), encoder_output, mask=mask)
            x = x + a

        x = x + self.mlp(self.ln2(x))

        return x, att


class Conv_MLP(nn.Module):
    def __init__(self, n_embd, mlp_hidden_times, act, resid_pdrop):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=n_embd, out_channels=int(mlp_hidden_times * n_embd), kernel_size=3, stride=1, padding=1)
        self.act = act
        self.conv2 = nn.Conv2d(in_channels=int(mlp_hidden_times * n_embd), out_channels=n_embd, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, x):
        n = x.size()[1]
        x = rearrange(x, 'b (h w) c -> b c h w', h=int(math.sqrt(n)))
        x = self.conv2(self.act(self.conv1(x)))
        x = rearrange(x, 'b c h w -> b (h w) c')
        return self.dropout(x)


class DiffusionTransformer(nn.Module):
    def __init__(
            self,
            codebook_path,
            label2vqidx_path,
            num_cls,
            n_layer=14,
            n_embd=1024,
            n_head=16,
            attn_pdrop=0,
            resid_pdrop=0,
            mlp_hidden_times=4,
            block_activate=None,
            attn_type='selfcross',
            condition_dim=512,
            diffusion_step=1000,
            timestep_type='adalayernorm',
            mlp_type='fc',
            checkpoint=False,
    ):
        super().__init__()

        self.use_checkpoint = checkpoint

        vq_codebook = torch.tensor(np.load(codebook_path, allow_pickle=True))
        num_groups, _, entry_dim = vq_codebook.shape

        self.context_indicator = torch.nn.Embedding(
            num_embeddings=2, embedding_dim=512  # , padding_idx=0
        )
        content_emb = torch.zeros(num_cls, num_groups * entry_dim)
        with open(label2vqidx_path, 'r') as f:
            for line in f.readlines():
                line = line.strip().split()
                label = int(line[0])
                vqid = list(map(int, line[1:]))
                for i in range(num_groups):
                    content_emb[label, i * entry_dim:(i + 1) * entry_dim] = vq_codebook[i, vqid[i]]
        self.content_emb = torch.nn.Embedding.from_pretrained(content_emb, freeze=True)
        self.content_emb_proj = torch.nn.Linear(num_groups * entry_dim, n_embd)
        self.pe = ScaledPositionalEncoding(n_embd, 0.1)

        # transformer
        assert attn_type == 'selfcross'
        all_attn_type = [attn_type] * n_layer

        self.blocks = nn.Sequential(*[Block(
            n_embd=n_embd,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            mlp_hidden_times=mlp_hidden_times,
            activate=block_activate,
            attn_type=all_attn_type[n],
            condition_dim=condition_dim,
            diffusion_step=diffusion_step,
            timestep_type=timestep_type,
            mlp_type=mlp_type,
        ) for n in range(n_layer)])

        # final prediction head
        self.to_logits = nn.Sequential(
            nn.LayerNorm(n_embd),
            nn.Linear(n_embd, num_cls - 1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine == True:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def parameters(self, recurse=True, name=None):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # return super().parameters(recurse=True)
        if name is None or name == 'none':
            return super().parameters(recurse=recurse)
        else:
            # separate out all parameters to those that will and won't experience regularizing weight decay
            print("GPTLikeTransformer: get parameters by the overwrite method!")
            decay = set()
            no_decay = set()
            whitelist_weight_modules = (torch.nn.Linear,)
            blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
            for mn, m in self.named_modules():
                for pn, p in m.named_parameters():
                    fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                    if pn.endswith('bias'):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)
            # special case the position embedding parameter as not decayed
            module_name = ['condition_emb', 'content_emb']
            pos_emb_name = ['pos_emb', 'width_emb', 'height_emb', 'pad_emb', 'token_type_emb']
            for mn in module_name:
                if hasattr(self, mn) and getattr(self, mn) is not None:
                    for pn in pos_emb_name:
                        if hasattr(getattr(self, mn), pn):
                            if isinstance(getattr(getattr(self, mn), pn), torch.nn.Parameter):
                                no_decay.add('{}.{}'.format(mn, pn))

            # validate that we considered every parameter
            param_dict = {pn: p for pn, p in self.transformer.named_parameters()}  # if p.requires_grad}
            inter_params = decay & no_decay
            union_params = decay | no_decay
            assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
            assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                               % (str(param_dict.keys() - union_params),)

            # create the pytorch optimizer object
            optim_groups = [
                {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
                {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            ]
            return optim_groups

    def forward(
            self,
            x_t,
            context_indicator,
            cond_emb,
            t, mask=None
    ):
        emb = self.content_emb(x_t)
        emb += self.context_indicator(context_indicator)
        emb = self.content_emb_proj(emb)
        emb = self.pe(emb)

        for block_idx in range(len(self.blocks)):
            if self.use_checkpoint == False:
                emb, att_weight = self.blocks[block_idx](emb, cond_emb, t, mask)  # B x (Ld+Lt) x D, B x (Ld+Lt) x (Ld+Lt)
            else:
                emb, att_weight = checkpoint_fn(self.blocks[block_idx], emb, cond_emb, t, mask)
        logits = self.to_logits(emb)  # B x (Ld+Lt) x n
        out = rearrange(logits, 'b l c -> b c l')
        return out
