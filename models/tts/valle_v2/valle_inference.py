# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchaudio


class ValleInference(torch.nn.Module):
    def __init__(
        self,
        use_vocos=False,
        use_speechtokenizer=True,
        ar_path=None,
        nar_path=None,
        speechtokenizer_path=None,
        device="cuda",
    ):
        super().__init__()

        self.device = device

        # prepare pretrained VALLE AR model
        from .valle_ar import ValleAR

        self.ar_model = ValleAR(
            phone_vocab_size=300,
            target_vocab_size=1024,
            pad_token_id=1324,
            bos_target_id=1325,
            eos_target_id=1326,
            bos_phone_id=1327,
            eos_phone_id=1328,
            bos_prompt_id=1329,
            eos_prompt_id=1330,
            num_hidden_layers=16,
        )
        # change the following path to your trained model path
        assert ar_path is not None
        self.ar_model.load_state_dict(torch.load(ar_path, map_location="cpu"))
        self.ar_model.eval().to(self.device)

        # prepare pretrained VALLE NAR model
        from .valle_nar import ValleNAR

        self.nar_model = ValleNAR(
            phone_vocab_size=300,
            target_vocab_size=1024,
            pad_token_id=1324,
            bos_target_id=1325,
            eos_target_id=1326,
            bos_phone_id=1327,
            eos_phone_id=1328,
            bos_prompt_id=1329,
            eos_prompt_id=1330,
            num_hidden_layers=16,
        )
        assert nar_path is not None
        self.nar_model.load_state_dict(torch.load(nar_path, map_location="cpu"))
        self.nar_model.eval().to(self.device)

        # prepare codec encoder
        assert not (
            use_speechtokenizer and use_vocos
        ), "Only one of use_speechtokenizer and use_vocos can be True"
        self.use_speechtokenizer = use_speechtokenizer
        if use_speechtokenizer:
            from models.codec.speechtokenizer.model import SpeechTokenizer

            # download from https://huggingface.co/fnlp/SpeechTokenizer/tree/main/speechtokenizer_hubert_avg
            config_path = speechtokenizer_path + "/config.json"
            ckpt_path = speechtokenizer_path + "/SpeechTokenizer.pt"
            self.codec_encoder = SpeechTokenizer.load_from_checkpoint(
                config_path, ckpt_path
            )
            self.codec_encoder.eval()
            self.codec_encoder.to(device)
            print(f"Loaded SpeechTokenizer from {config_path} and {ckpt_path}")
        else:
            # use Encodec
            from encodec import EncodecModel

            self.codec_encoder = EncodecModel.encodec_model_24khz()
            self.codec_encoder.set_target_bandwidth(6.0)
            self.codec_encoder.to(self.device)
            if use_vocos:
                from vocos import Vocos

                self.codec_decoder = Vocos.from_pretrained(
                    "charactr/vocos-encodec-24khz"
                )
                self.codec_decoder.to(self.device)
                print("Loaded Vocos")
            print("Loaded EncodecModel")

        self.use_vocos = use_vocos

    def decode(self, vq_ids):
        """vq_ids.shape: [8, B, T],
        returns: [B, 1, T]"""
        if self.use_speechtokenizer:
            # infer speechtokenizer
            return self.codec_encoder.decode(vq_ids)  # [B, 1, T]
        else:
            if not self.use_vocos:
                # vocos decoder
                return self.codec_encoder.decode([(vq_ids.transpose(0, 1), None)])
            else:
                # encodec decoder
                features = self.codec_decoder.codes_to_features(vq_ids.squeeze(1))
                bandwidth_id = torch.tensor([2], device=vq_ids.device)
                return self.codec_decoder.decode(
                    features, bandwidth_id=bandwidth_id
                ).unsqueeze(0)

    def forward(self, batch, chunk_configs: list, return_prompt=False, prompt_len=None):
        """batch: dict(
            speech: [B, T]
            phone_ids: [B, T]
        )
        returns: [B, 1, T] audio
        """
        if prompt_len is None:
            prompt_len = 100000  # no prompt length limiting
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)
        with torch.no_grad():
            if self.use_speechtokenizer:
                vq_id = self.codec_encoder.encode(
                    batch["speech"].unsqueeze(1)
                )  # [B,1,T] -> (n_q, B, T)
            else:
                vq_id = self.codec_encoder.encode(batch["speech"].unsqueeze(1))
                vq_id = torch.cat([encoded[0] for encoded in vq_id], dim=-1).transpose(
                    0, 1
                )

            # typically we only require one config in the chunk,
            # but we can also use multiple configs to, for example, use different sampling temperature at different positions
            for chunk in chunk_configs:
                ar_vq_ids = self.ar_model.sample_hf(
                    batch["phone_ids"],
                    vq_id[0, :, :prompt_len],
                    top_p=chunk["top_p"],
                    top_k=chunk["top_k"],
                    temperature=chunk["temperature"],
                    num_beams=chunk["num_beams"],
                    repeat_penalty=chunk["repeat_penalty"],
                    max_length=chunk["max_length"],
                )
                # recovered_audio_ar = self.decode(ar_vq_ids.unsqueeze(0))
                # torchaudio.save('recovered_audio_ar.wav', recovered_audio_ar[0].cpu(), 24000)

                nar_vq_ids = self.nar_model.sample_hf(
                    phone_ids=batch["phone_ids"],
                    prompt_ids=vq_id[:, :, :prompt_len],
                    first_stage_ids=ar_vq_ids,
                    # first_stage_ids=vq_id[0, :, prompt_len:],
                )

                if return_prompt:
                    nar_vq_ids = torch.cat(
                        [vq_id[..., :prompt_len], nar_vq_ids], dim=-1
                    )

                recovered_audio = self.decode(nar_vq_ids)
                return recovered_audio  # [B, 1, T]
