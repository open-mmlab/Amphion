# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchaudio
import numpy as np
import time
from .valle_ar_trainer import ValleARTrainer, make_pad_mask


class ValleNARTrainer(ValleARTrainer):
    def __init__(self, args=None, cfg=None):
        super().__init__(args, cfg)
        print("simple NAR")
        self.top1_accuracies = {
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
        }
        self.top5_accuracies = {
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
        }
        self.top10_accuracies = {
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
        }

    def _build_model(self):
        from .valle_nar import ValleNAR

        return ValleNAR(**self.cfg.model)

    def _train_step(self, batch):
        # inference codec
        """Returns: dict('speech', 'speech_len', 'phone_ids', 'phone_lens')
        speech: [B, T]
        speech_len: [B]
        phone_ids: [B, T]
        phone_lens: [B]
        """
        device = self.accelerator.device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        with torch.no_grad():
            if self.cfg.use_speechtokenizer:
                # Extract discrete codes from SpeechTokenizer
                # 16k
                vq_id = self.codec_encoder.encode(
                    batch["speech"].unsqueeze(1)
                )  # [B,T] -> (n_q, B, T)
                # RVQ_1 = codes[:1, :, :] # Contain content info, can be considered as semantic tokens
                # RVQ_supplement = codes[1:, :, :] # Contain timbre info, complete info lost by the first quantizer
                # Concatenating semantic tokens (RVQ_1) and supplementary timbre tokens and then decoding
                # wav = self.codec_encoder.decode(vq_id)
                # torchaudio.save('a.wav', wav[0].cpu(), 16000)

                # # Decoding from RVQ-i:j tokens from the ith quantizers to the jth quantizers
                # wav = model.decode(codes[i: (j + 1)], st=i)
            else:
                # using encodec, 24k
                vq_id = self.codec_encoder.encode(batch["speech"].unsqueeze(1))
                vq_id = torch.cat([encoded[0] for encoded in vq_id], dim=-1).transpose(
                    0, 1
                )

            # recovered_audio = self.codec_decoder(vq_emb, vq=False)
            # torchaudio.save('a.wav', recovered_audio[0], 16000)
            # vq_id: [8, B, T//320]
            batch["speech"] = vq_id
        batch["speech_len"] = batch["speech_len"] // 320  # our codec downsamples 320x
        assert batch["speech_len"].max() <= batch["speech"].shape[-1]

        phone_mask = 1 - make_pad_mask(
            batch["phone_lens"], max_len=batch["phone_ids"].size(1), left_pad=False
        ).to(torch.long)
        speech_mask = 1 - make_pad_mask(
            batch["speech_len"], max_len=batch["speech"].size(-1)
        ).to(torch.long)

        np.random.seed(int(time.time()) - 5 * self.accelerator.process_index)

        if hasattr(self.cfg.train, "dropout"):
            dropout = self.cfg.train.dropout
        else:
            dropout = 0.0

        out = self.model(
            phone_ids=batch["phone_ids"],
            phone_mask=phone_mask,
            target_ids=batch["speech"],
            target_mask=speech_mask,
            dropout=dropout,
        )
        loss = out.loss

        self.accelerator.log(
            {f"Train/NAR L{out.target_quantization_layer} Top1 acc": out.top1_acc},
            step=self.step,
        )
        self.accelerator.log(
            {f"Train/NAR L{out.target_quantization_layer} Top5 acc": out.top5_acc},
            step=self.step,
        )
        self.accelerator.log(
            {f"Train/NAR L{out.target_quantization_layer} Top10 acc": out.top10_acc},
            step=self.step,
        )

        # if hasattr(out, 'top1_acc'):
        #     idx = out.target_quantization_layer
        #     self.top1_accuracies[idx].append(out.top1_acc)
        #     self.top5_accuracies[idx].append(out.top5_acc)
        #     self.top10_accuracies[idx].append(out.top10_acc)
        #     if len(self.top1_accuracies[idx]) >= 160:
        #         breakpoint()
        # if self.accelerator.is_main_process:
        #     print(loss)
        return loss

    def _test_step(self, batch):
        # inference codec
        """Returns: dict('speech', 'speech_len', 'phone_ids', 'phone_lens')
        speech: [B, T]
        speech_len: [B]
        phone_ids: [B, T]
        phone_lens: [B]
        """
        import torchaudio

        device = self.accelerator.device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        with torch.no_grad():
            if self.cfg.use_speechtokenizer:
                # Extract discrete codes from SpeechTokenizer
                # 16k
                vq_id = self.codec_encoder.encode(
                    batch["speech"].unsqueeze(1)
                )  # [B,1,T] -> (n_q, B, T)
                # Concatenating semantic tokens (RVQ_1) and supplementary timbre tokens and then decoding
                # wav = self.codec_encoder.decode(vq_id)
                # torchaudio.save('a.wav', wav[0].cpu(), 16000)

            else:
                vq_id = self.codec_encoder.encode(batch["speech"].unsqueeze(1))
                vq_id = torch.cat([encoded[0] for encoded in vq_id], dim=-1).transpose(
                    0, 1
                )
            # recovered_audio = self.codec_encoder.decode([(vq_id.transpose(0,1), None)])
            # recovered_audio = self.codec_decoder(vq_emb, vq=False)
            # torchaudio.save('a.wav', recovered_audio[0], 16000)
            # vq_id: [8, B, T//200]

            # vq_emb = self.codec_decoder.quantizer.vq2emb(vq=vq_id[:1], n_quantizers=1)
            # recovered_audio = self.codec_decoder(vq_emb, vq=False)
            # recovered_audio.shape: torch.Size([1, 1, 50200])

            batch["speech"] = vq_id

            # save gt
            if self.cfg.use_speechtokenizer:
                recovered_audio = self.codec_encoder.decode(vq_id)
            else:
                recovered_audio = self.codec_encoder.decode(
                    [(vq_id.transpose(0, 1), None)]
                )
            torchaudio.save("gt.wav", recovered_audio[0].cpu(), 16000)
            self.model.eval()
            out_vq_ids = self.model.sample_hf(
                phone_ids=batch["phone_ids"][:1],
                prompt_ids=batch["speech"][:, :1, :150],
                first_stage_ids=batch["speech"][0, :1, 150:],
            )
            # breakpoint()
            # out_vq_ids = torch.cat([batch['speech'][:, :225], out_vq_ids], dim=1)

            # reconstruct form tokens
            if self.cfg.use_speechtokenizer:
                recovered_audio = self.codec_encoder.decode(out_vq_ids)
            else:
                recovered_audio = self.codec_encoder.decode(
                    [(out_vq_ids.transpose(0, 1)[:1], None)]
                )
            torchaudio.save("a.wav", recovered_audio[0].cpu(), 16000)
            breakpoint()
