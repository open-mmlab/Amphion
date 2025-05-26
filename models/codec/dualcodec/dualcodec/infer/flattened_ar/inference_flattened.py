# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from cv2 import repeat
import torch
from einops import rearrange
from .flatten_patterns import offset_codes, deoffset_codes


class Inference:
    def __init__(
        self,
        model,
        tokenizer_obj,
        dualcodec_inference_obj,
        device="cuda",
        normalize=False,
        half=False,
        split_paragraph=True,
        offset_sizes=[16384, 4096, 4096, 4096],
        **kwargs,
    ) -> None:
        self.model = model
        import safetensors.torch

        self.model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer_obj
        self.dualcodec_inference_obj = dualcodec_inference_obj
        self.device = device
        self.normalize = normalize
        self.offset_sizes = offset_sizes

        self.model = self.model.half()

        self.split_paragraph = split_paragraph

    @torch.no_grad()
    def inference(
        self,
        speech_24k,
        prompt_speech,
        prompt_text,
        prompt_language,
        target_text,
        target_language,
        use_prompt_text=True,
        temp=1.0,
        top_k=1000,
        top_p=0.85,
        repeat_penalty=1.1,
    ):
        """
            Generate text given speech and text prompts.

        Args:
            prompt_speech (str or Tensor): Speech file path or a tensor with shape (n_samples,).
            prompt_text (str): Text prompt.
            prompt_language (str): Language of the prompt.
            target_text (str): Target text to be completed.
            target_language (str): Language of the target text.
            use_prompt_text (bool, optional): Whether to use the prompt text as input. Defaults to True.
            temp (float, optional): Temperature parameter for the distribution. Defaults to 1.0.
            top_k (int, optional): Number of tokens to keep before applying `top_p`. Defaults to 1000.
            top_p (float, optional): Probability threshold to use for filtering tokens. Defaults to 0.85.

        Returns:
            str: Completed text.
        """
        self.model.eval()
        prompt_text = prompt_text.strip()
        # prompt_text = prompt_text.replace('.',',')
        # prompt_text = prompt_text.replace('。','，')
        target_text = target_text.replace("\n", "")
        target_text = target_text.replace("\t", "")
        return_values_0 = []
        return_values_1 = []

        prompt_len_tmp = len(self.tokenizer.encode(prompt_text)) // 2

        if self.split_paragraph:
            if prompt_language == "zh":
                from dualcodec.utils.frontend_utils import split_paragraph

                texts = split_paragraph(
                    target_text,
                    None,
                    "zh",
                    token_max_n=60 - prompt_len_tmp,
                    token_min_n=40 - prompt_len_tmp,
                    merge_len=20,
                    comma_split=False,
                )
            elif prompt_language == "ja":
                from dualcodec.utils.frontend_utils import split_paragraph

                texts = split_paragraph(
                    target_text,
                    None,
                    "zh",
                    token_max_n=70,
                    token_min_n=60,
                    merge_len=20,
                    comma_split=False,
                )
            elif prompt_language == "en":
                from dualcodec.utils.frontend_utils import split_paragraph

                texts = split_paragraph(
                    target_text,
                    self.tokenizer.encode,
                    "en",
                    token_max_n=70 - prompt_len_tmp,
                    token_min_n=60 - prompt_len_tmp,
                    merge_len=20,
                    comma_split=True,
                )
            else:
                texts = [target_text]
        if prompt_language == "en":
            texts = [prompt_text + " " + t for t in texts]
        else:
            texts = [prompt_text + t for t in texts]
        print(texts)

        all_codes = []

        for text in texts:

            if self.normalize:
                from dualcodec.dataset.processor import normalize

                text = list(
                    normalize(
                        [
                            {
                                "language": prompt_language,
                                "text": text,
                            }
                        ],
                        en_punct=True,
                        use_kana=False,
                    )
                )[0]["text"]
            print(text)

            prompt_text_tokens = torch.tensor(
                [
                    [self.tokenizer.to_language_token(prompt_language)]
                    + self.tokenizer.encode(text)
                ],
                dtype=torch.int32,
                device=self.device,
            )
            prompt_text_len = torch.tensor(
                [prompt_text_tokens.shape[-1]], device=self.device
            )

            # target_text_tokens = torch.tensor(
            #     [tokenizer.encode(target_text)], dtype=torch.int32
            # )
            # target_text_len = torch.tensor([target_text_tokens.shape[-1]])

            text_token = prompt_text_tokens

            # prompt semantic codes
            # semantic_code, _ = self._extract_semantic_code(input_features, attention_mask)
            semantic_codes, acoustic_codes = self.dualcodec_inference_obj.encode(
                prompt_speech, n_quantizers=4
            )
            semantic_codes = rearrange(semantic_codes, "b t -> b t 1")
            num_codec_layers = 4
            semantic_code = torch.cat([semantic_codes, acoustic_codes], dim=-1)[
                ..., :num_codec_layers
            ]

            semantic_code = offset_codes(semantic_code, self.offset_sizes)
            semantic_code = rearrange(semantic_code, "b t q -> b (t q)")

            ret_semantic_code = semantic_code.clone().detach()

            out = self.model.inference(
                text=text_token,
                text_len=prompt_text_len,
                prompt_text=None,
                prompt_text_len=None,
                prompt_speech_token=semantic_code,
                prompt_speech_token_len=torch.tensor([semantic_code.shape[-1]]),
                top_k=top_k,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                temperature=temp,
            )
            out = deoffset_codes(out, self.offset_sizes)

            all_codes.append(out)

        all_codes = torch.cat(all_codes, dim=1)  # FIXME not tested
        out = self.dualcodec_inference_obj.decode(all_codes)
        return out
