# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


class PromptPreparer:
    def prepare_prompts(self, y, y_lens, codes, nar_stage, y_prompts_codes):
        if self.prefix_mode == 0:
            y_emb, prefix_len = self._handle_prefix_mode_0(y, codes, nar_stage)
        elif self.prefix_mode == 1:
            y_emb, prefix_len = self._handle_prefix_mode_1(y, y_lens, codes, nar_stage)
        elif self.prefix_mode in [2, 4]:
            y_emb, prefix_len = self._handle_prefix_mode_2_4(
                y, y_lens, codes, nar_stage, y_prompts_codes
            )
        else:
            raise ValueError("Invalid prefix mode")

        return y_emb, prefix_len

    def _handle_prefix_mode_0(self, y, codes, nar_stage):
        prefix_len = 0
        y_emb = self.nar_audio_embeddings[0](y)
        for j in range(1, nar_stage):
            y_emb = y_emb + self.nar_audio_embeddings[j](codes[..., j])
        return y_emb, 0

    def _handle_prefix_mode_1(self, y, y_lens, codes, nar_stage):
        int_low = (0.25 * y_lens.min()).type(torch.int64).item()
        prefix_len = torch.randint(int_low, int_low * 2, size=()).item()
        prefix_len = min(prefix_len, 225)

        y_prompts = self.nar_audio_embeddings[0](y[:, :prefix_len])
        y_emb = self.nar_audio_embeddings[0](y[:, prefix_len:])
        for j in range(1, self.num_quantizers):
            y_prompts += self.nar_audio_embeddings[j](codes[:, :prefix_len, j])
            if j < nar_stage:
                y_emb += self.nar_audio_embeddings[j](codes[:, prefix_len:, j])
        y_emb = torch.concat([y_prompts, y_emb], axis=1)
        return y_emb, prefix_len

    def _handle_prefix_mode_2_4(self, y, y_lens, codes, nar_stage, y_prompts_codes):
        if self.prefix_mode == 2:
            prefix_len = min(225, int(0.25 * y_lens.min().item()))

            y_prompts_codes = []
            for b in range(codes.shape[0]):
                start = self.rng.randint(0, y_lens[b].item() - prefix_len)
                y_prompts_codes.append(
                    torch.clone(codes[b, start : start + prefix_len])
                )
                codes[b, start : start + prefix_len, nar_stage] = self.audio_token_num
            y_prompts_codes = torch.stack(y_prompts_codes, dim=0)
        else:
            prefix_len = y_prompts_codes.shape[1]

        y_prompts = self.nar_audio_embeddings[0](y_prompts_codes[..., 0])
        y_emb = self.nar_audio_embeddings[0](y)
        for j in range(1, self.num_quantizers):
            y_prompts += self.nar_audio_embeddings[j](y_prompts_codes[..., j])
            if j < nar_stage:
                y_emb += self.nar_audio_embeddings[j](codes[..., j])
        y_emb = torch.concat([y_prompts, y_emb], axis=1)

        return y_emb, prefix_len
