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

from typing import Dict, Optional, Tuple

import torch

from modules.wenet_extractor.cif.predictor import MAELoss
from modules.wenet_extractor.paraformer.search.beam_search import Hypothesis
from modules.wenet_extractor.transformer.asr_model import ASRModel
from modules.wenet_extractor.transformer.ctc import CTC
from modules.wenet_extractor.transformer.decoder import TransformerDecoder
from modules.wenet_extractor.transformer.encoder import TransformerEncoder
from modules.wenet_extractor.utils.common import IGNORE_ID, add_sos_eos, th_accuracy
from modules.wenet_extractor.utils.mask import make_pad_mask


class Paraformer(ASRModel):
    """Paraformer: Fast and Accurate Parallel Transformer for
    Non-autoregressive End-to-End Speech Recognition
    see https://arxiv.org/pdf/2206.08317.pdf
    """

    def __init__(
        self,
        vocab_size: int,
        encoder: TransformerEncoder,
        decoder: TransformerDecoder,
        ctc: CTC,
        predictor,
        ctc_weight: float = 0.5,
        predictor_weight: float = 1.0,
        predictor_bias: int = 0,
        ignore_id: int = IGNORE_ID,
        reverse_weight: float = 0.0,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
    ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert 0.0 <= predictor_weight <= 1.0, predictor_weight

        super().__init__(
            vocab_size,
            encoder,
            decoder,
            ctc,
            ctc_weight,
            ignore_id,
            reverse_weight,
            lsm_weight,
            length_normalized_loss,
        )
        self.predictor = predictor
        self.predictor_weight = predictor_weight
        self.predictor_bias = predictor_bias
        self.criterion_pre = MAELoss(normalize_length=length_normalized_loss)

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        # 1. Encoder
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        # 2a. Attention-decoder branch
        if self.ctc_weight != 1.0:
            loss_att, acc_att, loss_pre = self._calc_att_loss(
                encoder_out, encoder_mask, text, text_lengths
            )
        else:
            # loss_att = None
            # loss_pre = None
            loss_att: torch.Tensor = torch.tensor(0)
            loss_pre: torch.Tensor = torch.tensor(0)

        # 2b. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc = self.ctc(encoder_out, encoder_out_lens, text, text_lengths)
        else:
            loss_ctc = None

        if loss_ctc is None:
            loss = loss_att + self.predictor_weight * loss_pre
        # elif loss_att is None:
        elif loss_att == torch.tensor(0):
            loss = loss_ctc
        else:
            loss = (
                self.ctc_weight * loss_ctc
                + (1 - self.ctc_weight) * loss_att
                + self.predictor_weight * loss_pre
            )
        return {
            "loss": loss,
            "loss_att": loss_att,
            "loss_ctc": loss_ctc,
            "loss_pre": loss_pre,
        }

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, torch.Tensor]:
        if self.predictor_bias == 1:
            _, ys_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
            ys_pad_lens = ys_pad_lens + self.predictor_bias
        pre_acoustic_embeds, pre_token_length, _, pre_peak_index = self.predictor(
            encoder_out, ys_pad, encoder_mask, ignore_id=self.ignore_id
        )
        # 1. Forward decoder
        decoder_out, _, _ = self.decoder(
            encoder_out, encoder_mask, pre_acoustic_embeds, ys_pad_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_pad,
            ignore_label=self.ignore_id,
        )
        loss_pre: torch.Tensor = self.criterion_pre(
            ys_pad_lens.type_as(pre_token_length), pre_token_length
        )

        return loss_att, acc_att, loss_pre

    def calc_predictor(self, encoder_out, encoder_mask):
        encoder_mask = (
            ~make_pad_mask(encoder_mask, max_len=encoder_out.size(1))[:, None, :]
        ).to(encoder_out.device)
        pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = self.predictor(
            encoder_out, None, encoder_mask, ignore_id=self.ignore_id
        )
        return pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index

    def cal_decoder_with_predictor(
        self, encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens
    ):
        decoder_out, _, _ = self.decoder(
            encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens
        )
        decoder_out = torch.log_softmax(decoder_out, dim=-1)
        return decoder_out, ys_pad_lens

    def recognize(self):
        raise NotImplementedError

    def paraformer_greedy_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply beam search on attention decoder

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion

        Returns:
            torch.Tensor: decoding result, (batch, max_result_len)
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        device = speech.device
        batch_size = speech.shape[0]

        # Let's assume B = batch_size and N = beam_size
        # 1. Encoder
        encoder_out, encoder_mask = self._forward_encoder(
            speech,
            speech_lengths,
            decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming,
        )  # (B, maxlen, encoder_dim)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        # 2. Predictor
        predictor_outs = self.calc_predictor(encoder_out, encoder_mask)
        pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = (
            predictor_outs[0],
            predictor_outs[1],
            predictor_outs[2],
            predictor_outs[3],
        )
        pre_token_length = pre_token_length.round().long()
        if torch.max(pre_token_length) < 1:
            return torch.tensor([]), torch.tensor([])
        # 2. Decoder forward
        decoder_outs = self.cal_decoder_with_predictor(
            encoder_out, encoder_out_lens, pre_acoustic_embeds, pre_token_length
        )
        decoder_out, ys_pad_lens = decoder_outs[0], decoder_outs[1]
        hyps = []
        b, n, d = decoder_out.size()
        for i in range(b):
            x = encoder_out[i, : encoder_out_lens[i], :]
            am_scores = decoder_out[i, : pre_token_length[i], :]
            yseq = am_scores.argmax(dim=-1)
            score = am_scores.max(dim=-1)[0]
            score = torch.sum(score, dim=-1)
            # pad with mask tokens to ensure compatibility with sos/eos tokens
            yseq = torch.tensor(
                [self.sos] + yseq.tolist() + [self.eos], device=yseq.device
            )
            nbest_hyps = [Hypothesis(yseq=yseq, score=score)]

            for hyp in nbest_hyps:
                assert isinstance(hyp, (Hypothesis)), type(hyp)

                # remove sos/eos and get hyps
                last_pos = -1
                if isinstance(hyp.yseq, list):
                    token_int = hyp.yseq[1:last_pos]
                else:
                    token_int = hyp.yseq[1:last_pos].tolist()

                # remove blank symbol id and unk id, which is assumed to be 0
                # and 1
                token_int = list(filter(lambda x: x != 0 and x != 1, token_int))
                hyps.append(token_int)
        return hyps

    def paraformer_beam_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_search: torch.nn.Module = None,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply beam search on attention decoder

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_lengths (torch.Tensor): (batch, )
            beam_search (torch.nn.Moudle): beam search module
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion

        Returns:
            torch.Tensor: decoding result, (batch, max_result_len)
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        device = speech.device
        batch_size = speech.shape[0]

        # Let's assume B = batch_size and N = beam_size
        # 1. Encoder
        encoder_out, encoder_mask = self._forward_encoder(
            speech,
            speech_lengths,
            decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming,
        )  # (B, maxlen, encoder_dim)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        # 2. Predictor
        predictor_outs = self.calc_predictor(encoder_out, encoder_mask)
        pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = (
            predictor_outs[0],
            predictor_outs[1],
            predictor_outs[2],
            predictor_outs[3],
        )
        pre_token_length = pre_token_length.round().long()
        if torch.max(pre_token_length) < 1:
            return torch.tensor([]), torch.tensor([])
        # 2. Decoder forward
        decoder_outs = self.cal_decoder_with_predictor(
            encoder_out, encoder_out_lens, pre_acoustic_embeds, pre_token_length
        )
        decoder_out, ys_pad_lens = decoder_outs[0], decoder_outs[1]
        hyps = []
        b, n, d = decoder_out.size()
        for i in range(b):
            x = encoder_out[i, : encoder_out_lens[i], :]
            am_scores = decoder_out[i, : pre_token_length[i], :]
            if beam_search is not None:
                nbest_hyps = beam_search(x=x, am_scores=am_scores)
                nbest_hyps = nbest_hyps[:1]
            else:
                yseq = am_scores.argmax(dim=-1)
                score = am_scores.max(dim=-1)[0]
                score = torch.sum(score, dim=-1)
                # pad with mask tokens to ensure compatibility with sos/eos
                # tokens
                yseq = torch.tensor(
                    [self.sos] + yseq.tolist() + [self.eos], device=yseq.device
                )
                nbest_hyps = [Hypothesis(yseq=yseq, score=score)]

            for hyp in nbest_hyps:
                assert isinstance(hyp, (Hypothesis)), type(hyp)

                # remove sos/eos and get hyps
                last_pos = -1
                if isinstance(hyp.yseq, list):
                    token_int = hyp.yseq[1:last_pos]
                else:
                    token_int = hyp.yseq[1:last_pos].tolist()

                # remove blank symbol id and unk id, which is assumed to be 0
                # and 1
                token_int = list(filter(lambda x: x != 0 and x != 1, token_int))
                hyps.append(token_int)
        return hyps
