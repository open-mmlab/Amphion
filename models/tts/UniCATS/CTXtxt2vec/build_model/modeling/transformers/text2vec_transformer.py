import random
import torch
from torch import nn
import torch.nn.functional as F
from models.tts.UniCATS.CTXtxt2vec.build_model.modeling.transformers.espnet_nets.transformer.encoder import Encoder as TransformerEncoder
from models.tts.UniCATS.CTXtxt2vec.build_model.modeling.transformers.espnet_nets.transformer.embedding import ScaledPositionalEncoding
from models.tts.UniCATS.CTXtxt2vec.build_model.modeling.transformers.espnet_nets.nets_utils import make_pad_mask
from models.tts.UniCATS.CTXtxt2vec.build_model.modeling.transformers.espnet_nets.fastspeech.duration_predictor import DurationPredictor, DurationPredictorLoss
from models.tts.UniCATS.CTXtxt2vec.build_model.modeling.transformers.espnet_nets.fastspeech.length_regulator import LengthRegulator

from models.tts.UniCATS.CTXtxt2vec.build_model.utils.misc import instantiate_from_config
import numpy as np

from torch.cuda.amp import autocast

eps = 1e-8


def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


def sum_within_sent(x, sent_lens):
    result = torch.zeros_like(sent_lens).type(x.dtype)
    offset = 0
    for i in range(len(result)):
        sent_len = sent_lens[i].item()
        result[i] = torch.sum(x[offset:offset + sent_len])
        offset += sent_len
    return result


def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def log_onehot_to_index(log_x):
    return log_x.argmax(1)


def alpha_schedule(time_step, N=100, att_1=0.99999, att_T=0.000009, ctt_1=0.000009, ctt_T=0.89999):
    att = np.arange(0, time_step) / (time_step - 1) * (att_T - att_1) + att_1
    att = np.concatenate(([1], att))
    at = att[1:] / att[:-1]
    ctt = np.arange(0, time_step) / (time_step - 1) * (ctt_T - ctt_1) + ctt_1
    ctt = np.concatenate(([0], ctt))
    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1 - one_minus_ct
    bt = (1 - at - ct) / N
    att = np.concatenate((att[1:], [1]))
    ctt = np.concatenate((ctt[1:], [0]))
    btt = (1 - att - ctt) / N
    return at, bt, ct, att, btt, ctt


class Text2VecTransformer(nn.Module):
    def __init__(
            self,
            *,
            num_cls,
            transformer_config=None,
            diffusion_step=100,
            ctt_T=0.99999,
            alpha_init_type='cos',
            auxiliary_loss_weight=0,
            adaptive_auxiliary_loss=False,
            mask_weight=[1, 1],
    ):
        super().__init__()

        encoder_input_layer = torch.nn.Embedding(
            num_embeddings=80, embedding_dim=384, padding_idx=0
        )
        self.text_encoder = TransformerEncoder(idim=0,
                                               attention_dim=384,
                                               attention_heads=4,
                                               linear_units=1536,
                                               num_blocks=2,
                                               input_layer=encoder_input_layer,
                                               dropout_rate=0.1,
                                               positional_dropout_rate=0.1,
                                               attention_dropout_rate=0.1,
                                               pos_enc_class=ScaledPositionalEncoding,
                                               normalize_before=False,
                                               concat_after=False,
                                               positionwise_layer_type="linear",
                                               positionwise_conv_kernel_size=3)

        self.duration_predictor = DurationPredictor(
            idim=384,
            n_layers=2,
            n_chans=32,
            kernel_size=9,
            dropout_rate=0.1,
        )
        self.duration_criterion = DurationPredictorLoss()

        # define length regulator
        self.length_regulator = LengthRegulator()

        transformer_config['params']['diffusion_step'] = diffusion_step
        transformer_config['params']['num_cls'] = num_cls
        self.transformer = instantiate_from_config(transformer_config)
        self.amp = False

        self.num_classes = num_cls
        self.loss_type = 'vb_stochastic'
        self.num_timesteps = diffusion_step
        self.parametrization = 'x0'
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.adaptive_auxiliary_loss = adaptive_auxiliary_loss
        self.mask_weight = mask_weight

        if alpha_init_type == "alpha1":
            at, bt, ct, att, btt, ctt = alpha_schedule(self.num_timesteps, N=self.num_classes - 1, ctt_T=ctt_T)
        else:
            print("alpha_init_type is Wrong !! ")

        at = torch.tensor(at.astype('float64'))
        bt = torch.tensor(bt.astype('float64'))
        ct = torch.tensor(ct.astype('float64'))
        log_at = torch.log(at)
        log_bt = torch.log(bt)
        log_ct = torch.log(ct)
        att = torch.tensor(att.astype('float64'))
        btt = torch.tensor(btt.astype('float64'))
        ctt = torch.tensor(ctt.astype('float64'))
        log_cumprod_at = torch.log(att)
        log_cumprod_bt = torch.log(btt)
        log_cumprod_ct = torch.log(ctt)

        log_1_min_ct = log_1_min_a(log_ct)
        log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct)

        assert log_add_exp(log_ct, log_1_min_ct).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_ct, log_1_min_cumprod_ct).abs().sum().item() < 1.e-5

        self.diffusion_acc_list = [0] * self.num_timesteps
        self.diffusion_keep_list = [0] * self.num_timesteps
        # Convert to float32 and register buffers.
        self.register_buffer('log_at', log_at.float())
        self.register_buffer('log_bt', log_bt.float())
        self.register_buffer('log_ct', log_ct.float())
        self.register_buffer('log_cumprod_at', log_cumprod_at.float())
        self.register_buffer('log_cumprod_bt', log_cumprod_bt.float())
        self.register_buffer('log_cumprod_ct', log_cumprod_ct.float())
        self.register_buffer('log_1_min_ct', log_1_min_ct.float())
        self.register_buffer('log_1_min_cumprod_ct', log_1_min_cumprod_ct.float())

        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))

    def multinomial_kl(self, log_prob1, log_prob2):  # compute KL loss on log_prob
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep(self, log_x_t, t):  # q(xt|xt_1)
        log_at = extract(self.log_at, t, log_x_t.shape)  # at
        log_bt = extract(self.log_bt, t, log_x_t.shape)  # bt
        log_ct = extract(self.log_ct, t, log_x_t.shape)  # ct
        log_1_min_ct = extract(self.log_1_min_ct, t, log_x_t.shape)  # 1-ct

        log_probs = torch.cat(
            [
                log_add_exp(log_x_t[:, :-1, :] + log_at, log_bt),
                log_add_exp(log_x_t[:, -1:, :] + log_1_min_ct, log_ct)
            ],
            dim=1
        )

        return log_probs

    def q_pred(self, log_x_start, t):
        # q(xt|x0)
        # log_x_start can be onehot or not
        t = (t + (self.num_timesteps + 1)) % (self.num_timesteps + 1)
        log_cumprod_at = extract(self.log_cumprod_at, t, log_x_start.shape)  # at~
        log_cumprod_bt = extract(self.log_cumprod_bt, t, log_x_start.shape)  # bt~
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)  # ct~
        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct, t, log_x_start.shape)  # 1-ct~

        log_probs = torch.cat(
            [
                log_add_exp(log_x_start[:, :-1, :] + log_cumprod_at, log_cumprod_bt),
                log_add_exp(log_x_start[:, -1:, :] + log_1_min_cumprod_ct, log_cumprod_ct)
            ],
            dim=1
        )

        return log_probs

    def predict_start(self, log_x_t, context_indicator, cond_emb, t, mask=None):
        # p(x0|xt)
        x_t = log_onehot_to_index(log_x_t)
        if self.amp:
            with autocast():
                out = self.transformer(x_t, context_indicator, cond_emb, t, mask)
        else:
            out = self.transformer(x_t, context_indicator, cond_emb, t, mask)

        assert out.size(0) == x_t.size(0)
        assert out.size(1) == self.num_classes - 1
        assert out.size()[2:] == x_t.size()[1:]
        log_pred = F.log_softmax(out.double(), dim=1).float()
        batch_size = log_x_t.size()[0]

        zero_vector = torch.zeros(batch_size, 1, log_pred.size(-1)).type_as(log_x_t) - 70
        log_pred = torch.cat((log_pred, zero_vector), dim=1)
        log_pred = torch.clamp(log_pred, -70, 0)

        return log_pred

    def q_posterior(self, log_x_start, log_x_t, t):
        # p_theta(xt_1|xt) = sum(q(xt-1|xt,x0')*p(x0'))
        # notice that log_x_t is onehot
        assert t.min().item() >= 0 and t.max().item() < self.num_timesteps
        batch_size = log_x_start.size()[0]
        onehot_x_t = log_onehot_to_index(log_x_t)
        mask = (onehot_x_t == self.num_classes - 1).unsqueeze(1)
        log_one_vector = torch.zeros(batch_size, 1, 1).type_as(log_x_t)
        log_zero_vector = torch.log(log_one_vector + 1.0e-30).expand(-1, -1, log_x_t.size(-1))

        log_qt = self.q_pred(log_x_t, t)  # q(xt|x0)
        # log_qt = torch.cat((log_qt[:,:-1,:], log_zero_vector), dim=1)
        log_qt = log_qt[:, :-1, :]
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)  # ct~
        ct_cumprod_vector = log_cumprod_ct.expand(-1, self.num_classes - 1, -1)
        # ct_cumprod_vector = torch.cat((ct_cumprod_vector, log_one_vector), dim=1)
        log_qt = (~mask) * log_qt + mask * ct_cumprod_vector

        log_qt_one_timestep = self.q_pred_one_timestep(log_x_t, t)  # q(xt|xt_1)
        log_qt_one_timestep = torch.cat((log_qt_one_timestep[:, :-1, :], log_zero_vector), dim=1)
        log_ct = extract(self.log_ct, t, log_x_start.shape)  # ct
        ct_vector = log_ct.expand(-1, self.num_classes - 1, -1)
        ct_vector = torch.cat((ct_vector, log_one_vector), dim=1)
        log_qt_one_timestep = (~mask) * log_qt_one_timestep + mask * ct_vector

        # log_x_start = torch.cat((log_x_start, log_zero_vector), dim=1)
        # q = log_x_start - log_qt
        q = log_x_start[:, :-1, :] - log_qt
        q = torch.cat((q, log_zero_vector), dim=1)
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        q = q - q_log_sum_exp
        log_EV_xtmin_given_xt_given_xstart = self.q_pred(q, t - 1) + log_qt_one_timestep + q_log_sum_exp
        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)

    def p_pred(self, log_x, cond_emb, t, prefix=None, suffix=None):
        # if x0, first p(x0|xt), than sum(q(xt-1|xt,x0)*p(x0|xt))

        if prefix is not None:
            log_x[:, :, :prefix.size(-1)] = index_to_log_onehot(prefix, self.num_classes)
        if suffix is not None:
            log_x[:, :, -suffix.size(-1):] = index_to_log_onehot(suffix, self.num_classes)

        context_indicator = torch.zeros(log_x.size(0), log_x.size(-1)).long().to(log_x.device)
        if prefix is not None:
            context_indicator[:, :prefix.size(-1)] = 1
        if suffix is not None:
            context_indicator[:, -suffix.size(-1):] = 1

        log_x_recon = self.predict_start(log_x, context_indicator, cond_emb, t)

        if self.parametrization == 'x0' and t > 0:
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t)
        else:
            if prefix is not None:
                log_x_recon[:, :, :prefix.size(-1)] = index_to_log_onehot(prefix, self.num_classes)
            if suffix is not None:
                log_x_recon[:, :, -suffix.size(-1):] = index_to_log_onehot(suffix, self.num_classes)
            log_model_pred = log_x_recon

        return log_model_pred

    @torch.no_grad()
    def p_sample(self, log_x, cond_emb, t, prefix=None, suffix=None):
        # sample xt-1 for next step from xt, actually is p(xt-1|xt)

        # set context
        if prefix is not None:
            log_x[:, :, :prefix.size(-1)] = index_to_log_onehot(prefix, self.num_classes)
        if suffix is not None:
            log_x[:, :, -suffix.size(-1):] = index_to_log_onehot(suffix, self.num_classes)

        # build context indicator
        context_indicator = torch.zeros(log_x.size(0), log_x.size(-1)).long().to(log_x.device)
        if prefix is not None:
            context_indicator[:, :prefix.size(-1)] = 1
        if suffix is not None:
            context_indicator[:, -suffix.size(-1):] = 1

        # calculate x_0
        log_x_recon = self.predict_start(log_x, context_indicator, cond_emb, t)

        if t > 0:

            # calculate x_(t-1)
            model_log_prob = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t)
            out = self.log_sample_categorical(model_log_prob, sample=True)
        else:
            if prefix is not None:
                log_x_recon[:, :, :prefix.size(-1)] = index_to_log_onehot(prefix, self.num_classes)
            if suffix is not None:
                log_x_recon[:, :, -suffix.size(-1):] = index_to_log_onehot(suffix, self.num_classes)
            out = self.log_sample_categorical(log_x_recon, sample=False)

        return out

    def log_sample_categorical(self, logits, sample=True):  # use gumbel to sample onehot vector from log probability
        uniform = torch.rand_like(logits)
        if sample:
            gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        else:
            gumbel_noise = torch.zeros_like(uniform)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start, t):  # diffusion step, q(xt|x0) and sample xt
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)

        log_sample = self.log_sample_categorical(log_EV_qxt_x0)

        return log_sample

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def _train_loss(self, x, cond_emb, feat_len, mask, is_train=True):  # get the KL loss
        b, device = x.size(0), x.device

        assert self.loss_type == 'vb_stochastic'
        x_start = x
        t, pt = self.sample_time(b, device, 'importance')

        context_mask = torch.zeros_like(mask)
        for i in range(b):
            num_frames = feat_len[i].item()
            rand_int = random.random()
            if rand_int < 0.3:  # mask the first 3-5 seconds as prompt
                start = 0
                end = random.randint(200, 300)
                context_mask[i, start:end] = True
            elif rand_int < 0.9:
                context_mask_len = max(100, int(num_frames * random.randint(1, 8) / 10.0))
                start = random.randint(0, num_frames - context_mask_len)
                end = start + context_mask_len
                context_mask[i, :start] = True
                context_mask[i, end:] = True

        log_x_start = index_to_log_onehot(x_start, self.num_classes)
        log_xt = self.q_sample(log_x_start=log_x_start, t=t)
        log_xt = ~context_mask.unsqueeze(-2) * log_xt + context_mask.unsqueeze(-2) * log_x_start
        xt = log_onehot_to_index(log_xt)

        ############### go to p_theta function ###############
        log_x0_recon = self.predict_start(log_xt, context_mask.long(), cond_emb, t=t, mask=mask.unsqueeze(-2))  # P_theta(x0|xt)
        log_model_prob = self.q_posterior(log_x_start=log_x0_recon, log_x_t=log_xt, t=t)  # go through q(xt_1|xt,x0)

        # compute log_true_prob now
        log_true_prob = self.q_posterior(log_x_start=log_x_start, log_x_t=log_xt, t=t)

        mask = mask.bitwise_or(context_mask)
        x_start = x_start.masked_select(~mask).unsqueeze(0)
        xt = xt.masked_select(~mask).unsqueeze(0)
        log_x_start = log_x_start.transpose(-1, -2).masked_select(~mask.unsqueeze(-1)).view(-1, log_x_start.size(1)).transpose(-1, -2).unsqueeze(0)

        log_xt = log_xt.transpose(-1, -2).masked_select(~mask.unsqueeze(-1)).view(-1, log_xt.size(1)).transpose(-1, -2).unsqueeze(0)

        log_x0_recon = log_x0_recon.transpose(-1, -2).masked_select(~mask.unsqueeze(-1)).view(-1, log_x0_recon.size(1)).transpose(-1, -2).unsqueeze(0)

        log_model_prob = log_model_prob.transpose(-1, -2).masked_select(~mask.unsqueeze(-1)).view(-1, log_model_prob.size(1)).transpose(-1, -2).unsqueeze(0)

        log_true_prob = log_true_prob.transpose(-1, -2).masked_select(~mask.unsqueeze(-1)).view(-1, log_true_prob.size(1)).transpose(-1, -2).unsqueeze(0)

        ################## compute acc list ################
        x0_recon = log_onehot_to_index(log_x0_recon)
        x0_real = x_start
        xt_1_recon = log_onehot_to_index(log_model_prob)
        xt_recon = log_onehot_to_index(log_xt)
        offset = 0
        for index in range(t.size()[0]):
            this_t = t[index].item()
            sent_len = feat_len[index].item()
            same_rate = (x0_recon[:, offset:offset + sent_len] == x0_real[:, offset:offset + sent_len]).sum().cpu() / sent_len
            self.diffusion_acc_list[this_t] = same_rate.item() * 0.1 + self.diffusion_acc_list[this_t] * 0.9
            same_rate = (xt_1_recon[:, offset:offset + sent_len] == xt_recon[:, offset:offset + sent_len]).sum().cpu() / sent_len
            self.diffusion_keep_list[this_t] = same_rate.item() * 0.1 + self.diffusion_keep_list[this_t] * 0.9
            offset += sent_len

        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        mask_region = (xt == self.num_classes - 1).float()
        mask_weight = mask_region * self.mask_weight[0] + (1. - mask_region) * self.mask_weight[1]
        kl = kl * mask_weight
        kl = sum_within_sent(kl[0], feat_len)

        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        decoder_nll = sum_within_sent(decoder_nll[0], feat_len)

        mask = (t == torch.zeros_like(t)).float()
        kl_loss = mask * decoder_nll + (1. - mask) * kl

        Lt2 = kl_loss.pow(2)
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

        # Upweigh loss term of the kl
        # vb_loss = kl_loss / pt + kl_prior
        loss1 = kl_loss / pt
        vb_loss = loss1
        if self.auxiliary_loss_weight != 0 and is_train == True:
            kl_aux = self.multinomial_kl(log_x_start[:, :-1, :], log_x0_recon[:, :-1, :])
            kl_aux = kl_aux * mask_weight
            kl_aux = sum_within_sent(kl_aux[0], feat_len)
            kl_aux_loss = mask * decoder_nll + (1. - mask) * kl_aux
            if self.adaptive_auxiliary_loss == True:
                addition_loss_weight = (1 - t / self.num_timesteps) + 1.0
            else:
                addition_loss_weight = 1.0

            loss2 = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss / pt
            vb_loss += loss2

        return log_model_prob, vb_loss

    @property
    def device(self):
        return self.transformer.to_logits[-1].weight.device

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
            label, feat_len, text, text_len, duration,
            return_loss=False,
            return_logits=True,
            return_att_weight=False,
            is_train=True,
            **kwargs):
        if kwargs.get('autocast') == True:
            self.amp = True

        text_mask = make_pad_mask(text_len).to(text.device)
        feat_mask = make_pad_mask(feat_len).to(text.device)

        hs, _ = self.text_encoder(text, ~text_mask.unsqueeze(-2))  # (B, Tmax, adim)

        duration_hat = self.duration_predictor(hs, text_mask)
        cond_emb = self.length_regulator(hs, duration)  # (B, Lmax, adim)

        # now we get cond_emb and sample_image
        if is_train == True:
            log_model_prob, loss_vqdiff = self._train_loss(label, cond_emb, feat_len, feat_mask)
            loss_vqdiff = loss_vqdiff.sum() / torch.sum(feat_len)

        duration_hat = duration_hat.masked_select(~text_mask)
        duration = duration.masked_select(~text_mask)
        loss_dur = self.duration_criterion(duration_hat, duration)

        loss = loss_dur + loss_vqdiff

        # 4) get output, especially loss
        out = {}
        if return_logits:
            out['logits'] = torch.exp(log_model_prob)

        if return_loss:
            out['vqdiff loss'] = loss_vqdiff
            out['dur_loss'] = loss_dur
            out['loss'] = loss
        self.amp = False
        return out

    def sample(self, text, prefix=None, suffix=None, return_logits=False):

        device = self.log_at.device

        text_input = text
        if prefix is not None:
            text_input = torch.cat([prefix['text'], text_input], dim=-1)
        if suffix is not None:
            text_input = torch.cat([text_input, suffix['text']], dim=-1)

        with torch.no_grad():
            hs, _ = self.text_encoder(text_input, None)  # ~text_mask.unsqueeze(-2))  # (B, Tmax, adim)
            duration_hat = self.duration_predictor.inference(hs)

            gt_prefix_suffix_duration = 0
            predicted_prefix_suffix_duration = 0
            if prefix is not None:
                gt_prefix_suffix_duration += torch.sum(prefix['duration']).item()
                predicted_prefix_suffix_duration += torch.sum(duration_hat[:, :prefix['duration'].size(-1)]).item()
            if suffix is not None:
                gt_prefix_suffix_duration += torch.sum(suffix['duration']).item()
                predicted_prefix_suffix_duration += torch.sum(duration_hat[:, -suffix['duration'].size(-1):]).item()
            if gt_prefix_suffix_duration > 0:
                duration_ratio = gt_prefix_suffix_duration / predicted_prefix_suffix_duration
                duration_hat = (duration_hat.float() * duration_ratio).round().long()
            if prefix is not None:
                duration_hat[:, :prefix['duration'].size(-1)] = prefix['duration']
            if suffix is not None:
                duration_hat[:, -suffix['duration'].size(-1):] = suffix['duration']

            cond_emb = self.length_regulator(hs, duration_hat)  # (B, Lmax, adim)
            batch_size, seq_len, _ = cond_emb.shape

            t = torch.full((batch_size,), self.num_timesteps - 1, device=device, dtype=torch.long)
            x_start = torch.randint(0, self.num_classes - 1, size=(batch_size, seq_len), device=device)
            prefix_feat = None
            suffix_feat = None
            if prefix is not None:
                prefix_feat = prefix['feat']
            if suffix is not None:
                suffix_feat = suffix['feat']
            log_x_start = index_to_log_onehot(x_start, self.num_classes)
            log_z = self.q_sample(log_x_start=log_x_start, t=t)
            for diffusion_index in range(self.num_timesteps - 1, -1, -1):
                t = torch.full((batch_size,), diffusion_index, device=device, dtype=torch.long)
                log_z = self.p_sample(log_z, cond_emb, t, prefix_feat, suffix_feat)
                # log_z is log_onehot

        content_token = log_onehot_to_index(log_z)

        output = {'content_token': content_token}
        if return_logits:
            output['logits'] = torch.exp(log_z)
        return output
