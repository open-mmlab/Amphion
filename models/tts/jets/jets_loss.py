# Copyright (c) 2024 Amphion.
#
# This code is modified from https://github.com/imdanboy/jets/blob/main/espnet2/gan_tts/jets/loss.py
# Licensed under Apache License 2.0

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa

from models.vocoders.gan.discriminator.mpd import MultiScaleMultiPeriodDiscriminator
from models.tts.jets.alignments import make_non_pad_mask, make_pad_mask


class GeneratorAdversarialLoss(torch.nn.Module):
    """Generator adversarial loss module."""

    def __init__(self):
        super().__init__()

    def forward(self, outputs) -> torch.Tensor:
        if isinstance(outputs, (tuple, list)):
            adv_loss = 0.0
            for i, outputs_ in enumerate(outputs):
                if isinstance(outputs_, (tuple, list)):
                    # NOTE(kan-bayashi): case including feature maps
                    outputs_ = outputs_[-1]
                adv_loss += F.mse_loss(outputs_, outputs_.new_ones(outputs_.size()))
        else:
            adv_loss = F.mse_loss(outputs, outputs.new_ones(outputs.size()))

        return adv_loss


class FeatureMatchLoss(torch.nn.Module):
    """Feature matching loss module."""

    def __init__(
        self,
        average_by_layers: bool = False,
        average_by_discriminators: bool = False,
        include_final_outputs: bool = True,
    ):
        """Initialize FeatureMatchLoss module.

        Args:
            average_by_layers (bool): Whether to average the loss by the number
                of layers.
            average_by_discriminators (bool): Whether to average the loss by
                the number of discriminators.
            include_final_outputs (bool): Whether to include the final output of
                each discriminator for loss calculation.

        """
        super().__init__()
        self.average_by_layers = average_by_layers
        self.average_by_discriminators = average_by_discriminators
        self.include_final_outputs = include_final_outputs

    def forward(
        self,
        feats_hat: Union[List[List[torch.Tensor]], List[torch.Tensor]],
        feats: Union[List[List[torch.Tensor]], List[torch.Tensor]],
    ) -> torch.Tensor:
        """Calculate feature matching loss.

        Args:
            feats_hat (Union[List[List[Tensor]], List[Tensor]]): List of list of
                discriminator outputs or list of discriminator outputs calcuated
                from generator's outputs.
            feats (Union[List[List[Tensor]], List[Tensor]]): List of list of
                discriminator outputs or list of discriminator outputs calcuated
                from groundtruth..

        Returns:
            Tensor: Feature matching loss value.

        """
        feat_match_loss = 0.0
        for i, (feats_hat_, feats_) in enumerate(zip(feats_hat, feats)):
            feat_match_loss_ = 0.0
            if not self.include_final_outputs:
                feats_hat_ = feats_hat_[:-1]
                feats_ = feats_[:-1]
            for j, (feat_hat_, feat_) in enumerate(zip(feats_hat_, feats_)):
                feat_match_loss_ += F.l1_loss(feat_hat_, feat_.detach())
            if self.average_by_layers:
                feat_match_loss_ /= j + 1
            feat_match_loss += feat_match_loss_
        if self.average_by_discriminators:
            feat_match_loss /= i + 1

        return feat_match_loss


class DurationPredictorLoss(torch.nn.Module):
    """Loss function module for duration predictor.

    The loss value is Calculated in log domain to make it Gaussian.

    """

    def __init__(self, offset=1.0, reduction="mean"):
        """Initilize duration predictor loss module.

        Args:
            offset (float, optional): Offset value to avoid nan in log domain.
            reduction (str): Reduction type in loss calculation.

        """
        super().__init__()
        self.criterion = torch.nn.MSELoss(reduction=reduction)
        self.offset = offset

    def forward(self, outputs, targets):
        targets = torch.log(targets.float() + self.offset)
        loss = self.criterion(outputs, targets)

        return loss


class VarianceLoss(torch.nn.Module):
    def __init__(self):
        """Initialize JETS variance loss module."""
        super().__init__()

        # define criterions
        reduction = "mean"
        self.mse_criterion = torch.nn.MSELoss(reduction=reduction)
        self.duration_criterion = DurationPredictorLoss(reduction=reduction)

    def forward(
        self,
        d_outs: torch.Tensor,
        ds: torch.Tensor,
        p_outs: torch.Tensor,
        ps: torch.Tensor,
        e_outs: torch.Tensor,
        es: torch.Tensor,
        ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate forward propagation.

        Args:
            d_outs (LongTensor): Batch of outputs of duration predictor (B, T_text).
            ds (LongTensor): Batch of durations (B, T_text).
            p_outs (Tensor): Batch of outputs of pitch predictor (B, T_text, 1).
            ps (Tensor): Batch of target token-averaged pitch (B, T_text, 1).
            e_outs (Tensor): Batch of outputs of energy predictor (B, T_text, 1).
            es (Tensor): Batch of target token-averaged energy (B, T_text, 1).
            ilens (LongTensor): Batch of the lengths of each input (B,).

        Returns:
            Tensor: Duration predictor loss value.
            Tensor: Pitch predictor loss value.
            Tensor: Energy predictor loss value.

        """
        # apply mask to remove padded part
        duration_masks = make_non_pad_mask(ilens).to(ds.device)
        d_outs = d_outs.masked_select(duration_masks)
        ds = ds.masked_select(duration_masks)
        pitch_masks = make_non_pad_mask(ilens).to(ds.device)
        pitch_masks_ = make_non_pad_mask(ilens).unsqueeze(-1).to(ds.device)
        p_outs = p_outs.masked_select(pitch_masks)
        e_outs = e_outs.masked_select(pitch_masks)
        ps = ps.masked_select(pitch_masks_)
        es = es.masked_select(pitch_masks_)

        # calculate loss
        duration_loss = self.duration_criterion(d_outs, ds)
        pitch_loss = self.mse_criterion(p_outs, ps)
        energy_loss = self.mse_criterion(e_outs, es)

        return duration_loss, pitch_loss, energy_loss


class ForwardSumLoss(torch.nn.Module):
    """Forwardsum loss described at https://openreview.net/forum?id=0NQwnnwAORi"""

    def __init__(self):
        """Initialize forwardsum loss module."""
        super().__init__()

    def forward(
        self,
        log_p_attn: torch.Tensor,
        ilens: torch.Tensor,
        olens: torch.Tensor,
        blank_prob: float = np.e**-1,
    ) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            log_p_attn (Tensor): Batch of log probability of attention matrix
                (B, T_feats, T_text).
            ilens (Tensor): Batch of the lengths of each input (B,).
            olens (Tensor): Batch of the lengths of each target (B,).
            blank_prob (float): Blank symbol probability.

        Returns:
            Tensor: forwardsum loss value.

        """
        B = log_p_attn.size(0)

        # a row must be added to the attention matrix to account for
        #    blank token of CTC loss
        # (B,T_feats,T_text+1)
        log_p_attn_pd = F.pad(log_p_attn, (1, 0, 0, 0, 0, 0), value=np.log(blank_prob))

        loss = 0
        for bidx in range(B):
            # construct target sequnece.
            # Every text token is mapped to a unique sequnece number.
            target_seq = torch.arange(1, ilens[bidx] + 1).unsqueeze(0)
            cur_log_p_attn_pd = log_p_attn_pd[
                bidx, : olens[bidx], : ilens[bidx] + 1
            ].unsqueeze(
                1
            )  # (T_feats,1,T_text+1)
            cur_log_p_attn_pd = F.log_softmax(cur_log_p_attn_pd, dim=-1)
            loss += F.ctc_loss(
                log_probs=cur_log_p_attn_pd,
                targets=target_seq,
                input_lengths=olens[bidx : bidx + 1],
                target_lengths=ilens[bidx : bidx + 1],
                zero_infinity=True,
            )
        loss = loss / B
        return loss


class MelSpectrogramLoss(torch.nn.Module):
    """Mel-spectrogram loss."""

    def __init__(
        self,
        fs: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: Optional[int] = None,
        window: str = "hann",
        n_mels: int = 80,
        fmin: Optional[int] = 0,
        fmax: Optional[int] = None,
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        htk: bool = False,
    ):
        """Initialize Mel-spectrogram loss.

        Args:
            fs (int): Sampling rate.
            n_fft (int): FFT points.
            hop_length (int): Hop length.
            win_length (Optional[int]): Window length.
            window (str): Window type.
            n_mels (int): Number of Mel basis.
            fmin (Optional[int]): Minimum frequency for Mel.
            fmax (Optional[int]): Maximum frequency for Mel.
            center (bool): Whether to use center window.
            normalized (bool): Whether to use normalized one.
            onesided (bool): Whether to use oneseded one.

        """
        super().__init__()

        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft
        self.window = window
        self.n_mels = n_mels
        self.fmin = 0 if fmin is None else fmin
        self.fmax = fs / 2 if fmax is None else fmax
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        self.htk = htk

    def logmel(self, feat, ilens):
        mel_options = dict(
            sr=self.fs,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            htk=self.htk,
        )
        melmat = librosa.filters.mel(**mel_options)
        melmat = torch.from_numpy(melmat.T).float().to(feat.device)
        mel_feat = torch.matmul(feat, melmat)
        mel_feat = torch.clamp(mel_feat, min=1e-10)
        logmel_feat = mel_feat.log10()

        # Zero padding
        if ilens is not None:
            logmel_feat = logmel_feat.masked_fill(
                make_pad_mask(ilens, logmel_feat, 1), 0.0
            )
        else:
            ilens = feat.new_full(
                [feat.size(0)], fill_value=feat.size(1), dtype=torch.long
            )
        return logmel_feat

    def wav_to_mel(self, input, input_lengths=None):
        if self.window is not None:
            window_func = getattr(torch, f"{self.window}_window")
            window = window_func(
                self.win_length, dtype=input.dtype, device=input.device
            )

        stft_kwargs = dict(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            center=self.center,
            window=window,
            normalized=self.normalized,
            onesided=self.onesided,
            return_complex=True,
        )

        bs = input.size(0)
        if input.dim() == 3:
            multi_channel = True
            # input: (Batch, Nsample, Channels) -> (Batch * Channels, Nsample)
            input = input.transpose(1, 2).reshape(-1, input.size(1))
        else:
            multi_channel = False

        input_stft = torch.stft(input, **stft_kwargs)
        input_stft = torch.view_as_real(input_stft)
        input_stft = input_stft.transpose(1, 2)
        if multi_channel:
            input_stft = input_stft.view(
                bs, -1, input_stft.size(1), input_stft.size(2), 2
            ).transpose(1, 2)
        if input_lengths is not None:
            if self.center:
                pad = self.n_fft // 2
                input_lengths = input_lengths + 2 * pad

            feats_lens = (input_lengths - self.n_fft) // self.hop_length + 1
            input_stft.masked_fill_(make_pad_mask(feats_lens, input_stft, 1), 0.0)
        else:
            feats_lens = None
        input_power = input_stft[..., 0] ** 2 + input_stft[..., 1] ** 2
        input_amp = torch.sqrt(torch.clamp(input_power, min=1.0e-10))
        input_feats = self.logmel(input_amp, feats_lens)
        return input_feats, feats_lens

    def forward(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        mel_hat, _ = self.wav_to_mel(y_hat.squeeze(1))
        mel, _ = self.wav_to_mel(y.squeeze(1))
        mel_loss = F.l1_loss(mel_hat, mel)

        return mel_loss


class GeneratorLoss(nn.Module):
    """The total loss of the generator"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.mel_loss = MelSpectrogramLoss()
        self.generator_adv_loss = GeneratorAdversarialLoss()
        self.feat_match_loss = FeatureMatchLoss()
        self.var_loss = VarianceLoss()
        self.forwardsum_loss = ForwardSumLoss()

        self.lambda_adv = 1.0
        self.lambda_mel = 45.0
        self.lambda_feat_match = 2.0
        self.lambda_var = 1.0
        self.lambda_align = 2.0

    def forward(self, outputs_g, outputs_d, speech_):
        loss_g = {}

        # parse generator output
        (
            speech_hat_,
            bin_loss,
            log_p_attn,
            start_idxs,
            d_outs,
            ds,
            p_outs,
            ps,
            e_outs,
            es,
            text_lengths,
            feats_lengths,
        ) = outputs_g

        # parse discriminator output
        (p_hat, p) = outputs_d

        # calculate losses
        mel_loss = self.mel_loss(speech_hat_, speech_)
        adv_loss = self.generator_adv_loss(p_hat)
        feat_match_loss = self.feat_match_loss(p_hat, p)
        dur_loss, pitch_loss, energy_loss = self.var_loss(
            d_outs, ds, p_outs, ps, e_outs, es, text_lengths
        )
        forwardsum_loss = self.forwardsum_loss(log_p_attn, text_lengths, feats_lengths)

        # calculate total loss
        mel_loss = mel_loss * self.lambda_mel
        loss_g["mel_loss"] = mel_loss
        adv_loss = adv_loss * self.lambda_adv
        loss_g["adv_loss"] = adv_loss
        feat_match_loss = feat_match_loss * self.lambda_feat_match
        loss_g["feat_match_loss"] = feat_match_loss
        g_loss = mel_loss + adv_loss + feat_match_loss
        loss_g["g_loss"] = g_loss
        var_loss = (dur_loss + pitch_loss + energy_loss) * self.lambda_var
        loss_g["var_loss"] = var_loss
        align_loss = (forwardsum_loss + bin_loss) * self.lambda_align
        loss_g["align_loss"] = align_loss

        g_total_loss = g_loss + var_loss + align_loss

        loss_g["g_total_loss"] = g_total_loss

        return loss_g


class DiscriminatorAdversarialLoss(torch.nn.Module):
    """Discriminator adversarial loss module."""

    def __init__(
        self,
        average_by_discriminators: bool = True,
        loss_type: str = "mse",
    ):
        """Initialize DiscriminatorAversarialLoss module.

        Args:
            average_by_discriminators (bool): Whether to average the loss by
                the number of discriminators.
            loss_type (str): Loss type, "mse" or "hinge".

        """
        super().__init__()
        self.average_by_discriminators = average_by_discriminators
        assert loss_type in ["mse", "hinge"], f"{loss_type} is not supported."
        if loss_type == "mse":
            self.fake_criterion = self._mse_fake_loss
            self.real_criterion = self._mse_real_loss
        else:
            self.fake_criterion = self._hinge_fake_loss
            self.real_criterion = self._hinge_real_loss

    def forward(
        self,
        outputs_hat: Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor],
        outputs: Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calcualate discriminator adversarial loss.

        Args:
            outputs_hat (Union[List[List[Tensor]], List[Tensor], Tensor]): Discriminator
                outputs, list of discriminator outputs, or list of list of discriminator
                outputs calculated from generator.
            outputs (Union[List[List[Tensor]], List[Tensor], Tensor]): Discriminator
                outputs, list of discriminator outputs, or list of list of discriminator
                outputs calculated from groundtruth.

        Returns:
            Tensor: Discriminator real loss value.
            Tensor: Discriminator fake loss value.

        """
        if isinstance(outputs, (tuple, list)):
            real_loss = 0.0
            fake_loss = 0.0
            for i, (outputs_hat_, outputs_) in enumerate(zip(outputs_hat, outputs)):
                if isinstance(outputs_hat_, (tuple, list)):
                    # NOTE(kan-bayashi): case including feature maps
                    outputs_hat_ = outputs_hat_[-1]
                    outputs_ = outputs_[-1]
                real_loss += self.real_criterion(outputs_)
                fake_loss += self.fake_criterion(outputs_hat_)
            if self.average_by_discriminators:
                fake_loss /= i + 1
                real_loss /= i + 1
        else:
            real_loss = self.real_criterion(outputs)
            fake_loss = self.fake_criterion(outputs_hat)

        return real_loss, fake_loss

    def _mse_real_loss(self, x: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x, x.new_ones(x.size()))

    def _mse_fake_loss(self, x: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x, x.new_zeros(x.size()))

    def _hinge_real_loss(self, x: torch.Tensor) -> torch.Tensor:
        return -torch.mean(torch.min(x - 1, x.new_zeros(x.size())))

    def _hinge_fake_loss(self, x: torch.Tensor) -> torch.Tensor:
        return -torch.mean(torch.min(-x - 1, x.new_zeros(x.size())))


class DiscriminatorLoss(torch.nn.Module):
    """The total loss of the discriminator"""

    def __init__(self, cfg):
        super(DiscriminatorLoss, self).__init__()
        self.cfg = cfg
        self.discriminator = MultiScaleMultiPeriodDiscriminator()
        self.discriminator_adv_loss = DiscriminatorAdversarialLoss()

    def forward(self, speech_real, speech_generated):
        loss_d = {}

        real_loss, fake_loss = self.discriminator_adv_loss(
            speech_generated, speech_real
        )
        loss_d["loss_disc_all"] = real_loss + fake_loss

        return loss_d
