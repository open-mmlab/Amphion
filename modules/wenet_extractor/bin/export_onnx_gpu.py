# This module is from [WeNet](https://github.com/wenet-e2e/wenet).

# ## Citations

# ```bibtex
# @inproceedings{yao2021wenet,
#   title={WeNet: Production oriented Streaming and Non-streaming End-to-End Speech Recognition Toolkit},
#   author={Yao, Zhuoyuan and Wu, Di and Wang, Xiong and Zhang, Binbin and Yu, Fan and Yang, Chao and Peng, Zhendong and Chen, Xiaoyu and Xie, Lei and Lei, Xin},
#   booktitle={Proc. Interspeech},
#   year={2021},
#   address={Brno, Czech Republic },
#   organization={IEEE}
# }

# @article{zhang2022wenet,
#   title={WeNet 2.0: More Productive End-to-End Speech Recognition Toolkit},
#   author={Zhang, Binbin and Wu, Di and Peng, Zhendong and Song, Xingchen and Yao, Zhuoyuan and Lv, Hang and Xie, Lei and Yang, Chao and Pan, Fuping and Niu, Jianwei},
#   journal={arXiv preprint arXiv:2203.15455},
#   year={2022}
# }
#

from __future__ import print_function

import argparse
import os
import sys

import torch
import yaml
import logging

import torch.nn.functional as F
from wenet.utils.checkpoint import load_checkpoint
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.encoder import BaseEncoder
from wenet.utils.init_model import init_model
from wenet.utils.mask import make_pad_mask

try:
    import onnxruntime
except ImportError:
    print("Please install onnxruntime-gpu!")
    sys.exit(1)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class Encoder(torch.nn.Module):
    def __init__(self, encoder: BaseEncoder, ctc: CTC, beam_size: int = 10):
        super().__init__()
        self.encoder = encoder
        self.ctc = ctc
        self.beam_size = beam_size

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ):
        """Encoder
        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        Returns:
            encoder_out: B x T x F
            encoder_out_lens: B
            ctc_log_probs: B x T x V
            beam_log_probs: B x T x beam_size
            beam_log_probs_idx: B x T x beam_size
        """
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths, -1, -1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        ctc_log_probs = self.ctc.log_softmax(encoder_out)
        encoder_out_lens = encoder_out_lens.int()
        beam_log_probs, beam_log_probs_idx = torch.topk(
            ctc_log_probs, self.beam_size, dim=2
        )
        return (
            encoder_out,
            encoder_out_lens,
            ctc_log_probs,
            beam_log_probs,
            beam_log_probs_idx,
        )


class StreamingEncoder(torch.nn.Module):
    def __init__(self, model, required_cache_size, beam_size, transformer=False):
        super().__init__()
        self.ctc = model.ctc
        self.subsampling_rate = model.encoder.embed.subsampling_rate
        self.embed = model.encoder.embed
        self.global_cmvn = model.encoder.global_cmvn
        self.required_cache_size = required_cache_size
        self.beam_size = beam_size
        self.encoder = model.encoder
        self.transformer = transformer

    def forward(self, chunk_xs, chunk_lens, offset, att_cache, cnn_cache, cache_mask):
        """Streaming Encoder
        Args:
            xs (torch.Tensor): chunk input, with shape (b, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate + \
                        subsample.right_context + 1`
            offset (torch.Tensor): offset with shape (b, 1)
                        1 is retained for triton deployment
            required_cache_size (int): cache size required for next chunk
                compuation
                > 0: actual cache size
                <= 0: not allowed in streaming gpu encoder                   `
            att_cache (torch.Tensor): cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (b, elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.
            cnn_cache (torch.Tensor): cache tensor for cnn_module in conformer,
                (b, elayers, b, hidden-dim, cache_t2), where
                `cache_t2 == cnn.lorder - 1`
            cache_mask: (torch.Tensor): cache mask with shape (b, required_cache_size)
                 in a batch of request, each request may have different
                 history cache. Cache mask is used to indidate the effective
                 cache for each request
        Returns:
            torch.Tensor: log probabilities of ctc output and cutoff by beam size
                with shape (b, chunk_size, beam)
            torch.Tensor: index of top beam size probabilities for each timestep
                with shape (b, chunk_size, beam)
            torch.Tensor: output of current input xs,
                with shape (b, chunk_size, hidden-dim).
            torch.Tensor: new attention cache required for next chunk, with
                same shape (b, elayers, head, cache_t1, d_k * 2)
                as the original att_cache
            torch.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache.
            torch.Tensor: new cache mask, with same shape as the original
                cache mask
        """
        offset = offset.squeeze(1)
        T = chunk_xs.size(1)
        chunk_mask = ~make_pad_mask(chunk_lens, T).unsqueeze(1)
        # B X 1 X T
        chunk_mask = chunk_mask.to(chunk_xs.dtype)
        # transpose batch & num_layers dim
        att_cache = torch.transpose(att_cache, 0, 1)
        cnn_cache = torch.transpose(cnn_cache, 0, 1)

        # rewrite encoder.forward_chunk
        # <---------forward_chunk START--------->
        xs = self.global_cmvn(chunk_xs)
        # chunk mask is important for batch inferencing since
        # different sequence in a batch has different length
        xs, pos_emb, chunk_mask = self.embed(xs, chunk_mask, offset)
        cache_size = att_cache.size(3)  # required cache size
        masks = torch.cat((cache_mask, chunk_mask), dim=2)
        index = offset - cache_size

        pos_emb = self.embed.position_encoding(index, cache_size + xs.size(1))
        pos_emb = pos_emb.to(dtype=xs.dtype)

        next_cache_start = -self.required_cache_size
        r_cache_mask = masks[:, :, next_cache_start:]

        r_att_cache = []
        r_cnn_cache = []
        for i, layer in enumerate(self.encoder.encoders):
            xs, _, new_att_cache, new_cnn_cache = layer(
                xs, masks, pos_emb, att_cache=att_cache[i], cnn_cache=cnn_cache[i]
            )
            #   shape(new_att_cache) is (B, head, attention_key_size, d_k * 2),
            #   shape(new_cnn_cache) is (B, hidden-dim, cache_t2)
            r_att_cache.append(new_att_cache[:, :, next_cache_start:, :].unsqueeze(1))
            if not self.transformer:
                r_cnn_cache.append(new_cnn_cache.unsqueeze(1))
        if self.encoder.normalize_before:
            chunk_out = self.encoder.after_norm(xs)
        else:
            chunk_out = xs

        r_att_cache = torch.cat(r_att_cache, dim=1)  # concat on layers idx
        if not self.transformer:
            r_cnn_cache = torch.cat(r_cnn_cache, dim=1)  # concat on layers

        # <---------forward_chunk END--------->

        log_ctc_probs = self.ctc.log_softmax(chunk_out)
        log_probs, log_probs_idx = torch.topk(log_ctc_probs, self.beam_size, dim=2)
        log_probs = log_probs.to(chunk_xs.dtype)

        r_offset = offset + chunk_out.shape[1]
        # the below ops not supported in Tensorrt
        # chunk_out_lens = torch.div(chunk_lens, subsampling_rate,
        #                   rounding_mode='floor')
        chunk_out_lens = chunk_lens // self.subsampling_rate
        r_offset = r_offset.unsqueeze(1)

        return (
            log_probs,
            log_probs_idx,
            chunk_out,
            chunk_out_lens,
            r_offset,
            r_att_cache,
            r_cnn_cache,
            r_cache_mask,
        )


class StreamingSqueezeformerEncoder(torch.nn.Module):
    def __init__(self, model, required_cache_size, beam_size):
        super().__init__()
        self.ctc = model.ctc
        self.subsampling_rate = model.encoder.embed.subsampling_rate
        self.embed = model.encoder.embed
        self.global_cmvn = model.encoder.global_cmvn
        self.required_cache_size = required_cache_size
        self.beam_size = beam_size
        self.encoder = model.encoder
        self.reduce_idx = model.encoder.reduce_idx
        self.recover_idx = model.encoder.recover_idx
        if self.reduce_idx is None:
            self.time_reduce = None
        else:
            if self.recover_idx is None:
                self.time_reduce = "normal"  # no recovery at the end
            else:
                self.time_reduce = "recover"  # recovery at the end
                assert len(self.reduce_idx) == len(self.recover_idx)

    def calculate_downsampling_factor(self, i: int) -> int:
        if self.reduce_idx is None:
            return 1
        else:
            reduce_exp, recover_exp = 0, 0
            for exp, rd_idx in enumerate(self.reduce_idx):
                if i >= rd_idx:
                    reduce_exp = exp + 1
            if self.recover_idx is not None:
                for exp, rc_idx in enumerate(self.recover_idx):
                    if i >= rc_idx:
                        recover_exp = exp + 1
            return int(2 ** (reduce_exp - recover_exp))

    def forward(self, chunk_xs, chunk_lens, offset, att_cache, cnn_cache, cache_mask):
        """Streaming Encoder
        Args:
            xs (torch.Tensor): chunk input, with shape (b, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate + \
                        subsample.right_context + 1`
            offset (torch.Tensor): offset with shape (b, 1)
                        1 is retained for triton deployment
            required_cache_size (int): cache size required for next chunk
                compuation
                > 0: actual cache size
                <= 0: not allowed in streaming gpu encoder                   `
            att_cache (torch.Tensor): cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (b, elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.
            cnn_cache (torch.Tensor): cache tensor for cnn_module in conformer,
                (b, elayers, b, hidden-dim, cache_t2), where
                `cache_t2 == cnn.lorder - 1`
            cache_mask: (torch.Tensor): cache mask with shape (b, required_cache_size)
                 in a batch of request, each request may have different
                 history cache. Cache mask is used to indidate the effective
                 cache for each request
        Returns:
            torch.Tensor: log probabilities of ctc output and cutoff by beam size
                with shape (b, chunk_size, beam)
            torch.Tensor: index of top beam size probabilities for each timestep
                with shape (b, chunk_size, beam)
            torch.Tensor: output of current input xs,
                with shape (b, chunk_size, hidden-dim).
            torch.Tensor: new attention cache required for next chunk, with
                same shape (b, elayers, head, cache_t1, d_k * 2)
                as the original att_cache
            torch.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache.
            torch.Tensor: new cache mask, with same shape as the original
                cache mask
        """
        offset = offset.squeeze(1)
        T = chunk_xs.size(1)
        chunk_mask = ~make_pad_mask(chunk_lens, T).unsqueeze(1)
        # B X 1 X T
        chunk_mask = chunk_mask.to(chunk_xs.dtype)
        # transpose batch & num_layers dim
        att_cache = torch.transpose(att_cache, 0, 1)
        cnn_cache = torch.transpose(cnn_cache, 0, 1)

        # rewrite encoder.forward_chunk
        # <---------forward_chunk START--------->
        xs = self.global_cmvn(chunk_xs)
        # chunk mask is important for batch inferencing since
        # different sequence in a batch has different length
        xs, pos_emb, chunk_mask = self.embed(xs, chunk_mask, offset)
        elayers, cache_size = att_cache.size(0), att_cache.size(3)
        att_mask = torch.cat((cache_mask, chunk_mask), dim=2)
        index = offset - cache_size

        pos_emb = self.embed.position_encoding(index, cache_size + xs.size(1))
        pos_emb = pos_emb.to(dtype=xs.dtype)

        next_cache_start = -self.required_cache_size
        r_cache_mask = att_mask[:, :, next_cache_start:]

        r_att_cache = []
        r_cnn_cache = []
        mask_pad = torch.ones(1, xs.size(1), device=xs.device, dtype=torch.bool)
        mask_pad = mask_pad.unsqueeze(1)
        max_att_len: int = 0
        recover_activations: List[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []
        index = 0
        xs_lens = torch.tensor([xs.size(1)], device=xs.device, dtype=torch.int)
        xs = self.encoder.preln(xs)
        for i, layer in enumerate(self.encoder.encoders):
            if self.reduce_idx is not None:
                if self.time_reduce is not None and i in self.reduce_idx:
                    recover_activations.append((xs, att_mask, pos_emb, mask_pad))
                    xs, xs_lens, att_mask, mask_pad = self.encoder.time_reduction_layer(
                        xs, xs_lens, att_mask, mask_pad
                    )
                    pos_emb = pos_emb[:, ::2, :]
                    if self.encoder.pos_enc_layer_type == "rel_pos_repaired":
                        pos_emb = pos_emb[:, : xs.size(1) * 2 - 1, :]
                    index += 1

            if self.recover_idx is not None:
                if self.time_reduce == "recover" and i in self.recover_idx:
                    index -= 1
                    (
                        recover_tensor,
                        recover_att_mask,
                        recover_pos_emb,
                        recover_mask_pad,
                    ) = recover_activations[index]
                    # recover output length for ctc decode
                    xs = xs.unsqueeze(2).repeat(1, 1, 2, 1).flatten(1, 2)
                    xs = self.encoder.time_recover_layer(xs)
                    recoverd_t = recover_tensor.size(1)
                    xs = recover_tensor + xs[:, :recoverd_t, :].contiguous()
                    att_mask = recover_att_mask
                    pos_emb = recover_pos_emb
                    mask_pad = recover_mask_pad

            factor = self.calculate_downsampling_factor(i)

            xs, _, new_att_cache, new_cnn_cache = layer(
                xs,
                att_mask,
                pos_emb,
                att_cache=att_cache[i][:, :, ::factor, :][
                    :, :, : pos_emb.size(1) - xs.size(1), :
                ]
                if elayers > 0
                else att_cache[:, :, ::factor, :],
                cnn_cache=cnn_cache[i] if cnn_cache.size(0) > 0 else cnn_cache,
            )
            cached_att = new_att_cache[:, :, next_cache_start // factor :, :]
            cached_cnn = new_cnn_cache.unsqueeze(1)
            cached_att = (
                cached_att.unsqueeze(3).repeat(1, 1, 1, factor, 1).flatten(2, 3)
            )
            if i == 0:
                # record length for the first block as max length
                max_att_len = cached_att.size(2)
            r_att_cache.append(cached_att[:, :, :max_att_len, :].unsqueeze(1))
            r_cnn_cache.append(cached_cnn)

        chunk_out = xs
        r_att_cache = torch.cat(r_att_cache, dim=1)  # concat on layers idx
        r_cnn_cache = torch.cat(r_cnn_cache, dim=1)  # concat on layers

        # <---------forward_chunk END--------->

        log_ctc_probs = self.ctc.log_softmax(chunk_out)
        log_probs, log_probs_idx = torch.topk(log_ctc_probs, self.beam_size, dim=2)
        log_probs = log_probs.to(chunk_xs.dtype)

        r_offset = offset + chunk_out.shape[1]
        # the below ops not supported in Tensorrt
        # chunk_out_lens = torch.div(chunk_lens, subsampling_rate,
        #                   rounding_mode='floor')
        chunk_out_lens = chunk_lens // self.subsampling_rate
        r_offset = r_offset.unsqueeze(1)

        return (
            log_probs,
            log_probs_idx,
            chunk_out,
            chunk_out_lens,
            r_offset,
            r_att_cache,
            r_cnn_cache,
            r_cache_mask,
        )


class StreamingEfficientConformerEncoder(torch.nn.Module):
    def __init__(self, model, required_cache_size, beam_size):
        super().__init__()
        self.ctc = model.ctc
        self.subsampling_rate = model.encoder.embed.subsampling_rate
        self.embed = model.encoder.embed
        self.global_cmvn = model.encoder.global_cmvn
        self.required_cache_size = required_cache_size
        self.beam_size = beam_size
        self.encoder = model.encoder

        # Efficient Conformer
        self.stride_layer_idx = model.encoder.stride_layer_idx
        self.stride = model.encoder.stride
        self.num_blocks = model.encoder.num_blocks
        self.cnn_module_kernel = model.encoder.cnn_module_kernel

    def calculate_downsampling_factor(self, i: int) -> int:
        factor = 1
        for idx, stride_idx in enumerate(self.stride_layer_idx):
            if i > stride_idx:
                factor *= self.stride[idx]
        return factor

    def forward(self, chunk_xs, chunk_lens, offset, att_cache, cnn_cache, cache_mask):
        """Streaming Encoder
        Args:
            chunk_xs (torch.Tensor): chunk input, with shape (b, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate + \
                        subsample.right_context + 1`
            chunk_lens (torch.Tensor):
            offset (torch.Tensor): offset with shape (b, 1)
                        1 is retained for triton deployment
            att_cache (torch.Tensor): cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (b, elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.
            cnn_cache (torch.Tensor): cache tensor for cnn_module in conformer,
                (b, elayers, hidden-dim, cache_t2), where
                `cache_t2 == cnn.lorder - 1`
            cache_mask: (torch.Tensor): cache mask with shape (b, required_cache_size)
                 in a batch of request, each request may have different
                 history cache. Cache mask is used to indidate the effective
                 cache for each request
        Returns:
            torch.Tensor: log probabilities of ctc output and cutoff by beam size
                with shape (b, chunk_size, beam)
            torch.Tensor: index of top beam size probabilities for each timestep
                with shape (b, chunk_size, beam)
            torch.Tensor: output of current input xs,
                with shape (b, chunk_size, hidden-dim).
            torch.Tensor: new attention cache required for next chunk, with
                same shape (b, elayers, head, cache_t1, d_k * 2)
                as the original att_cache
            torch.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache.
            torch.Tensor: new cache mask, with same shape as the original
                cache mask
        """
        offset = offset.squeeze(1)  # (b, )
        offset *= self.calculate_downsampling_factor(self.num_blocks + 1)

        T = chunk_xs.size(1)
        chunk_mask = ~make_pad_mask(chunk_lens, T).unsqueeze(1)  # (b, 1, T)
        # B X 1 X T
        chunk_mask = chunk_mask.to(chunk_xs.dtype)
        # transpose batch & num_layers dim
        #   Shape(att_cache): (elayers, b, head, cache_t1, d_k * 2)
        #   Shape(cnn_cache): (elayers, b, outsize, cnn_kernel)
        att_cache = torch.transpose(att_cache, 0, 1)
        cnn_cache = torch.transpose(cnn_cache, 0, 1)

        # rewrite encoder.forward_chunk
        # <---------forward_chunk START--------->
        xs = self.global_cmvn(chunk_xs)
        # chunk mask is important for batch inferencing since
        # different sequence in a batch has different length
        xs, pos_emb, chunk_mask = self.embed(xs, chunk_mask, offset)
        cache_size = att_cache.size(3)  # required cache size
        masks = torch.cat((cache_mask, chunk_mask), dim=2)
        att_mask = torch.cat((cache_mask, chunk_mask), dim=2)
        index = offset - cache_size

        pos_emb = self.embed.position_encoding(index, cache_size + xs.size(1))
        pos_emb = pos_emb.to(dtype=xs.dtype)

        next_cache_start = -self.required_cache_size
        r_cache_mask = masks[:, :, next_cache_start:]

        r_att_cache = []
        r_cnn_cache = []
        mask_pad = chunk_mask.to(torch.bool)
        max_att_len, max_cnn_len = 0, 0  # for repeat_interleave of new_att_cache
        for i, layer in enumerate(self.encoder.encoders):
            factor = self.calculate_downsampling_factor(i)
            # NOTE(xcsong): Before layer.forward
            #   shape(att_cache[i:i + 1]) is (b, head, cache_t1, d_k * 2),
            #   shape(cnn_cache[i])       is (b=1, hidden-dim, cache_t2)
            # shape(new_att_cache) = [ batch, head, time2, outdim//head * 2 ]
            att_cache_trunc = 0
            if xs.size(1) + att_cache.size(3) / factor > pos_emb.size(1):
                # The time step is not divisible by the downsampling multiple
                # We propose to double the chunk_size.
                att_cache_trunc = (
                    xs.size(1) + att_cache.size(3) // factor - pos_emb.size(1) + 1
                )
            xs, _, new_att_cache, new_cnn_cache = layer(
                xs,
                att_mask,
                pos_emb,
                mask_pad=mask_pad,
                att_cache=att_cache[i][:, :, ::factor, :][:, :, att_cache_trunc:, :],
                cnn_cache=cnn_cache[i, :, :, :] if cnn_cache.size(0) > 0 else cnn_cache,
            )

            if i in self.stride_layer_idx:
                # compute time dimension for next block
                efficient_index = self.stride_layer_idx.index(i)
                att_mask = att_mask[
                    :, :: self.stride[efficient_index], :: self.stride[efficient_index]
                ]
                mask_pad = mask_pad[
                    :, :: self.stride[efficient_index], :: self.stride[efficient_index]
                ]
                pos_emb = pos_emb[:, :: self.stride[efficient_index], :]

            # shape(new_att_cache) = [batch, head, time2, outdim]
            new_att_cache = new_att_cache[:, :, next_cache_start // factor :, :]
            # shape(new_cnn_cache) = [batch, 1, outdim, cache_t2]
            new_cnn_cache = new_cnn_cache.unsqueeze(1)  # shape(1):layerID

            # use repeat_interleave to new_att_cache
            # new_att_cache = new_att_cache.repeat_interleave(repeats=factor, dim=2)
            new_att_cache = (
                new_att_cache.unsqueeze(3).repeat(1, 1, 1, factor, 1).flatten(2, 3)
            )
            # padding new_cnn_cache to cnn.lorder for casual convolution
            new_cnn_cache = F.pad(
                new_cnn_cache, (self.cnn_module_kernel - 1 - new_cnn_cache.size(3), 0)
            )

            if i == 0:
                # record length for the first block as max length
                max_att_len = new_att_cache.size(2)
                max_cnn_len = new_cnn_cache.size(3)

            # update real shape of att_cache and cnn_cache
            r_att_cache.append(new_att_cache[:, :, -max_att_len:, :].unsqueeze(1))
            r_cnn_cache.append(new_cnn_cache[:, :, :, -max_cnn_len:])

        if self.encoder.normalize_before:
            chunk_out = self.encoder.after_norm(xs)
        else:
            chunk_out = xs

        # shape of r_att_cache: (b, elayers, head, time2, outdim)
        r_att_cache = torch.cat(r_att_cache, dim=1)  # concat on layers idx
        # shape of r_cnn_cache: (b, elayers, outdim, cache_t2)
        r_cnn_cache = torch.cat(r_cnn_cache, dim=1)  # concat on layers

        # <---------forward_chunk END--------->

        log_ctc_probs = self.ctc.log_softmax(chunk_out)
        log_probs, log_probs_idx = torch.topk(log_ctc_probs, self.beam_size, dim=2)
        log_probs = log_probs.to(chunk_xs.dtype)

        r_offset = offset + chunk_out.shape[1]
        # the below ops not supported in Tensorrt
        # chunk_out_lens = torch.div(chunk_lens, subsampling_rate,
        #                   rounding_mode='floor')
        chunk_out_lens = (
            chunk_lens
            // self.subsampling_rate
            // self.calculate_downsampling_factor(self.num_blocks + 1)
        )
        chunk_out_lens += 1
        r_offset = r_offset.unsqueeze(1)

        return (
            log_probs,
            log_probs_idx,
            chunk_out,
            chunk_out_lens,
            r_offset,
            r_att_cache,
            r_cnn_cache,
            r_cache_mask,
        )


class Decoder(torch.nn.Module):
    def __init__(
        self,
        decoder: TransformerDecoder,
        ctc_weight: float = 0.5,
        reverse_weight: float = 0.0,
        beam_size: int = 10,
        decoder_fastertransformer: bool = False,
    ):
        super().__init__()
        self.decoder = decoder
        self.ctc_weight = ctc_weight
        self.reverse_weight = reverse_weight
        self.beam_size = beam_size
        self.decoder_fastertransformer = decoder_fastertransformer

    def forward(
        self,
        encoder_out: torch.Tensor,
        encoder_lens: torch.Tensor,
        hyps_pad_sos_eos: torch.Tensor,
        hyps_lens_sos: torch.Tensor,
        r_hyps_pad_sos_eos: torch.Tensor,
        ctc_score: torch.Tensor,
    ):
        """Encoder
        Args:
            encoder_out: B x T x F
            encoder_lens: B
            hyps_pad_sos_eos: B x beam x (T2+1),
                        hyps with sos & eos and padded by ignore id
            hyps_lens_sos: B x beam, length for each hyp with sos
            r_hyps_pad_sos_eos: B x beam x (T2+1),
                    reversed hyps with sos & eos and padded by ignore id
            ctc_score: B x beam, ctc score for each hyp
        Returns:
            decoder_out: B x beam x T2 x V
            r_decoder_out: B x beam x T2 x V
            best_index: B
        """
        B, T, F = encoder_out.shape
        bz = self.beam_size
        B2 = B * bz
        encoder_out = encoder_out.repeat(1, bz, 1).view(B2, T, F)
        encoder_mask = ~make_pad_mask(encoder_lens, T).unsqueeze(1)
        encoder_mask = encoder_mask.repeat(1, bz, 1).view(B2, 1, T)
        T2 = hyps_pad_sos_eos.shape[2] - 1
        hyps_pad = hyps_pad_sos_eos.view(B2, T2 + 1)
        hyps_lens = hyps_lens_sos.view(
            B2,
        )
        hyps_pad_sos = hyps_pad[:, :-1].contiguous()
        hyps_pad_eos = hyps_pad[:, 1:].contiguous()

        r_hyps_pad = r_hyps_pad_sos_eos.view(B2, T2 + 1)
        r_hyps_pad_sos = r_hyps_pad[:, :-1].contiguous()
        r_hyps_pad_eos = r_hyps_pad[:, 1:].contiguous()

        decoder_out, r_decoder_out, _ = self.decoder(
            encoder_out,
            encoder_mask,
            hyps_pad_sos,
            hyps_lens,
            r_hyps_pad_sos,
            self.reverse_weight,
        )
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        V = decoder_out.shape[-1]
        decoder_out = decoder_out.view(B2, T2, V)
        mask = ~make_pad_mask(hyps_lens, T2)  # B2 x T2
        # mask index, remove ignore id
        index = torch.unsqueeze(hyps_pad_eos * mask, 2)
        score = decoder_out.gather(2, index).squeeze(2)  # B2 X T2
        # mask padded part
        score = score * mask
        decoder_out = decoder_out.view(B, bz, T2, V)
        if self.reverse_weight > 0:
            r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
            r_decoder_out = r_decoder_out.view(B2, T2, V)
            index = torch.unsqueeze(r_hyps_pad_eos * mask, 2)
            r_score = r_decoder_out.gather(2, index).squeeze(2)
            r_score = r_score * mask
            score = score * (1 - self.reverse_weight) + self.reverse_weight * r_score
            r_decoder_out = r_decoder_out.view(B, bz, T2, V)
        score = torch.sum(score, axis=1)  # B2
        score = torch.reshape(score, (B, bz)) + self.ctc_weight * ctc_score
        best_index = torch.argmax(score, dim=1)
        if self.decoder_fastertransformer:
            return decoder_out, best_index
        else:
            return best_index


def to_numpy(tensors):
    out = []
    if type(tensors) == torch.tensor:
        tensors = [tensors]
    for tensor in tensors:
        if tensor.requires_grad:
            tensor = tensor.detach().cpu().numpy()
        else:
            tensor = tensor.cpu().numpy()
        out.append(tensor)
    return out


def test(xlist, blist, rtol=1e-3, atol=1e-5, tolerate_small_mismatch=True):
    for a, b in zip(xlist, blist):
        try:
            torch.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
        except AssertionError as error:
            if tolerate_small_mismatch:
                print(error)
            else:
                raise


def export_offline_encoder(model, configs, args, logger, encoder_onnx_path):
    bz = 32
    seq_len = 100
    beam_size = args.beam_size
    feature_size = configs["input_dim"]

    speech = torch.randn(bz, seq_len, feature_size, dtype=torch.float32)
    speech_lens = torch.randint(low=10, high=seq_len, size=(bz,), dtype=torch.int32)
    encoder = Encoder(model.encoder, model.ctc, beam_size)
    encoder.eval()

    torch.onnx.export(
        encoder,
        (speech, speech_lens),
        encoder_onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["speech", "speech_lengths"],
        output_names=[
            "encoder_out",
            "encoder_out_lens",
            "ctc_log_probs",
            "beam_log_probs",
            "beam_log_probs_idx",
        ],
        dynamic_axes={
            "speech": {0: "B", 1: "T"},
            "speech_lengths": {0: "B"},
            "encoder_out": {0: "B", 1: "T_OUT"},
            "encoder_out_lens": {0: "B"},
            "ctc_log_probs": {0: "B", 1: "T_OUT"},
            "beam_log_probs": {0: "B", 1: "T_OUT"},
            "beam_log_probs_idx": {0: "B", 1: "T_OUT"},
        },
        verbose=False,
    )

    with torch.no_grad():
        o0, o1, o2, o3, o4 = encoder(speech, speech_lens)

    providers = ["CUDAExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(encoder_onnx_path, providers=providers)
    ort_inputs = {"speech": to_numpy(speech), "speech_lengths": to_numpy(speech_lens)}
    ort_outs = ort_session.run(None, ort_inputs)

    # check encoder output
    test(to_numpy([o0, o1, o2, o3, o4]), ort_outs)
    logger.info("export offline onnx encoder succeed!")
    onnx_config = {
        "beam_size": args.beam_size,
        "reverse_weight": args.reverse_weight,
        "ctc_weight": args.ctc_weight,
        "fp16": args.fp16,
    }
    return onnx_config


def export_online_encoder(model, configs, args, logger, encoder_onnx_path):
    decoding_chunk_size = args.decoding_chunk_size
    subsampling = model.encoder.embed.subsampling_rate
    context = model.encoder.embed.right_context + 1
    decoding_window = (decoding_chunk_size - 1) * subsampling + context
    batch_size = 32
    audio_len = decoding_window
    feature_size = configs["input_dim"]
    output_size = configs["encoder_conf"]["output_size"]
    num_layers = configs["encoder_conf"]["num_blocks"]
    # in transformer the cnn module will not be available
    transformer = False
    cnn_module_kernel = configs["encoder_conf"].get("cnn_module_kernel", 1) - 1
    if not cnn_module_kernel:
        transformer = True
    num_decoding_left_chunks = args.num_decoding_left_chunks
    required_cache_size = decoding_chunk_size * num_decoding_left_chunks
    if configs["encoder"] == "squeezeformer":
        encoder = StreamingSqueezeformerEncoder(
            model, required_cache_size, args.beam_size
        )
    elif configs["encoder"] == "efficientConformer":
        encoder = StreamingEfficientConformerEncoder(
            model, required_cache_size, args.beam_size
        )
    else:
        encoder = StreamingEncoder(
            model, required_cache_size, args.beam_size, transformer
        )
    encoder.eval()

    # begin to export encoder
    chunk_xs = torch.randn(batch_size, audio_len, feature_size, dtype=torch.float32)
    chunk_lens = torch.ones(batch_size, dtype=torch.int32) * audio_len

    offset = torch.arange(0, batch_size).unsqueeze(1)
    #  (elayers, b, head, cache_t1, d_k * 2)
    head = configs["encoder_conf"]["attention_heads"]
    d_k = configs["encoder_conf"]["output_size"] // head
    att_cache = torch.randn(
        batch_size, num_layers, head, required_cache_size, d_k * 2, dtype=torch.float32
    )
    cnn_cache = torch.randn(
        batch_size, num_layers, output_size, cnn_module_kernel, dtype=torch.float32
    )

    cache_mask = torch.ones(batch_size, 1, required_cache_size, dtype=torch.float32)
    input_names = [
        "chunk_xs",
        "chunk_lens",
        "offset",
        "att_cache",
        "cnn_cache",
        "cache_mask",
    ]
    output_names = [
        "log_probs",
        "log_probs_idx",
        "chunk_out",
        "chunk_out_lens",
        "r_offset",
        "r_att_cache",
        "r_cnn_cache",
        "r_cache_mask",
    ]
    input_tensors = (chunk_xs, chunk_lens, offset, att_cache, cnn_cache, cache_mask)
    if transformer:
        output_names.pop(6)

    all_names = input_names + output_names
    dynamic_axes = {}
    for name in all_names:
        # only the first dimension is dynamic
        # all other dimension is fixed
        dynamic_axes[name] = {0: "B"}

    torch.onnx.export(
        encoder,
        input_tensors,
        encoder_onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=False,
    )

    with torch.no_grad():
        torch_outs = encoder(
            chunk_xs, chunk_lens, offset, att_cache, cnn_cache, cache_mask
        )
    if transformer:
        torch_outs = list(torch_outs).pop(6)
    ort_session = onnxruntime.InferenceSession(
        encoder_onnx_path, providers=["CUDAExecutionProvider"]
    )
    ort_inputs = {}

    input_tensors = to_numpy(input_tensors)
    for idx, name in enumerate(input_names):
        ort_inputs[name] = input_tensors[idx]
    if transformer:
        del ort_inputs["cnn_cache"]
    ort_outs = ort_session.run(None, ort_inputs)
    test(to_numpy(torch_outs), ort_outs, rtol=1e-03, atol=1e-05)
    logger.info("export to onnx streaming encoder succeed!")
    onnx_config = {
        "subsampling_rate": subsampling,
        "context": context,
        "decoding_chunk_size": decoding_chunk_size,
        "num_decoding_left_chunks": num_decoding_left_chunks,
        "beam_size": args.beam_size,
        "fp16": args.fp16,
        "feat_size": feature_size,
        "decoding_window": decoding_window,
        "cnn_module_kernel_cache": cnn_module_kernel,
    }
    return onnx_config


def export_rescoring_decoder(
    model, configs, args, logger, decoder_onnx_path, decoder_fastertransformer
):
    bz, seq_len = 32, 100
    beam_size = args.beam_size
    decoder = Decoder(
        model.decoder,
        model.ctc_weight,
        model.reverse_weight,
        beam_size,
        decoder_fastertransformer,
    )
    decoder.eval()

    hyps_pad_sos_eos = torch.randint(low=3, high=1000, size=(bz, beam_size, seq_len))
    hyps_lens_sos = torch.randint(
        low=3, high=seq_len, size=(bz, beam_size), dtype=torch.int32
    )
    r_hyps_pad_sos_eos = torch.randint(low=3, high=1000, size=(bz, beam_size, seq_len))

    output_size = configs["encoder_conf"]["output_size"]
    encoder_out = torch.randn(bz, seq_len, output_size, dtype=torch.float32)
    encoder_out_lens = torch.randint(low=3, high=seq_len, size=(bz,), dtype=torch.int32)
    ctc_score = torch.randn(bz, beam_size, dtype=torch.float32)

    input_names = [
        "encoder_out",
        "encoder_out_lens",
        "hyps_pad_sos_eos",
        "hyps_lens_sos",
        "r_hyps_pad_sos_eos",
        "ctc_score",
    ]
    output_names = ["best_index"]
    if decoder_fastertransformer:
        output_names.insert(0, "decoder_out")

    torch.onnx.export(
        decoder,
        (
            encoder_out,
            encoder_out_lens,
            hyps_pad_sos_eos,
            hyps_lens_sos,
            r_hyps_pad_sos_eos,
            ctc_score,
        ),
        decoder_onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "encoder_out": {0: "B", 1: "T"},
            "encoder_out_lens": {0: "B"},
            "hyps_pad_sos_eos": {0: "B", 2: "T2"},
            "hyps_lens_sos": {0: "B"},
            "r_hyps_pad_sos_eos": {0: "B", 2: "T2"},
            "ctc_score": {0: "B"},
            "best_index": {0: "B"},
        },
        verbose=False,
    )
    with torch.no_grad():
        o0 = decoder(
            encoder_out,
            encoder_out_lens,
            hyps_pad_sos_eos,
            hyps_lens_sos,
            r_hyps_pad_sos_eos,
            ctc_score,
        )
    providers = ["CUDAExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(decoder_onnx_path, providers=providers)

    input_tensors = [
        encoder_out,
        encoder_out_lens,
        hyps_pad_sos_eos,
        hyps_lens_sos,
        r_hyps_pad_sos_eos,
        ctc_score,
    ]
    ort_inputs = {}
    input_tensors = to_numpy(input_tensors)
    for idx, name in enumerate(input_names):
        ort_inputs[name] = input_tensors[idx]

    # if model.reverse weight == 0,
    # the r_hyps_pad will be removed
    # from the onnx decoder since it doen't play any role
    if model.reverse_weight == 0:
        del ort_inputs["r_hyps_pad_sos_eos"]
    ort_outs = ort_session.run(None, ort_inputs)

    # check decoder output
    if decoder_fastertransformer:
        test(to_numpy(o0), ort_outs, rtol=1e-03, atol=1e-05)
    else:
        test(to_numpy([o0]), ort_outs, rtol=1e-03, atol=1e-05)
    logger.info("export to onnx decoder succeed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="export x86_gpu model")
    parser.add_argument("--config", required=True, help="config file")
    parser.add_argument("--checkpoint", required=True, help="checkpoint model")
    parser.add_argument(
        "--cmvn_file",
        required=False,
        default="",
        type=str,
        help="global_cmvn file, default path is in config file",
    )
    parser.add_argument(
        "--reverse_weight",
        default=-1.0,
        type=float,
        required=False,
        help="reverse weight for bitransformer," + "default value is in config file",
    )
    parser.add_argument(
        "--ctc_weight",
        default=-1.0,
        type=float,
        required=False,
        help="ctc weight, default value is in config file",
    )
    parser.add_argument(
        "--beam_size",
        default=10,
        type=int,
        required=False,
        help="beam size would be ctc output size",
    )
    parser.add_argument(
        "--output_onnx_dir",
        default="onnx_model",
        help="output onnx encoder and decoder directory",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="whether to export fp16 model, default false",
    )
    # arguments for streaming encoder
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="whether to export streaming encoder, default false",
    )
    parser.add_argument(
        "--decoding_chunk_size",
        default=16,
        type=int,
        required=False,
        help="the decoding chunk size, <=0 is not supported",
    )
    parser.add_argument(
        "--num_decoding_left_chunks",
        default=5,
        type=int,
        required=False,
        help="number of left chunks, <= 0 is not supported",
    )
    parser.add_argument(
        "--decoder_fastertransformer",
        action="store_true",
        help="return decoder_out and best_index for ft",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_printoptions(precision=10)

    with open(args.config, "r") as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if args.cmvn_file and os.path.exists(args.cmvn_file):
        configs["cmvn_file"] = args.cmvn_file
    if args.reverse_weight != -1.0 and "reverse_weight" in configs["model_conf"]:
        configs["model_conf"]["reverse_weight"] = args.reverse_weight
        print("Update reverse weight to", args.reverse_weight)
    if args.ctc_weight != -1:
        print("Update ctc weight to ", args.ctc_weight)
        configs["model_conf"]["ctc_weight"] = args.ctc_weight
    configs["encoder_conf"]["use_dynamic_chunk"] = False

    model = init_model(configs)
    load_checkpoint(model, args.checkpoint)
    model.eval()

    if not os.path.exists(args.output_onnx_dir):
        os.mkdir(args.output_onnx_dir)
    encoder_onnx_path = os.path.join(args.output_onnx_dir, "encoder.onnx")
    export_enc_func = None
    if args.streaming:
        assert args.decoding_chunk_size > 0
        assert args.num_decoding_left_chunks > 0
        export_enc_func = export_online_encoder
    else:
        export_enc_func = export_offline_encoder

    onnx_config = export_enc_func(model, configs, args, logger, encoder_onnx_path)

    decoder_onnx_path = os.path.join(args.output_onnx_dir, "decoder.onnx")
    export_rescoring_decoder(
        model, configs, args, logger, decoder_onnx_path, args.decoder_fastertransformer
    )

    if args.fp16:
        try:
            import onnxmltools
            from onnxmltools.utils.float16_converter import convert_float_to_float16
        except ImportError:
            print("Please install onnxmltools!")
            sys.exit(1)
        encoder_onnx_model = onnxmltools.utils.load_model(encoder_onnx_path)
        encoder_onnx_model = convert_float_to_float16(encoder_onnx_model)
        encoder_onnx_path = os.path.join(args.output_onnx_dir, "encoder_fp16.onnx")
        onnxmltools.utils.save_model(encoder_onnx_model, encoder_onnx_path)
        decoder_onnx_model = onnxmltools.utils.load_model(decoder_onnx_path)
        decoder_onnx_model = convert_float_to_float16(decoder_onnx_model)
        decoder_onnx_path = os.path.join(args.output_onnx_dir, "decoder_fp16.onnx")
        onnxmltools.utils.save_model(decoder_onnx_model, decoder_onnx_path)
    # dump configurations

    config_dir = os.path.join(args.output_onnx_dir, "config.yaml")
    with open(config_dir, "w") as out:
        yaml.dump(onnx_config, out)
