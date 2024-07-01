# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.nn.utils.rnn import pad_sequence


class VALLECollator:
    def __init__(self, cfg=None):
        self.cfg = cfg

    def __call__(self, batch):
        """Returns: dict('speech', 'speech_len', 'phone_ids', 'phone_lens')
        speech: [B, T]
        speech_len: [B]
        phone_ids: [B, T]
        phone_lens: [B]
        """
        assert len(batch) != 0, "batch is empty before None checking"
        batch = [b for b in batch if b is not None]
        assert len(batch) != 0, "batch is empty after None checking"
        packed_batch_features = {}

        # Function to handle tensor copying
        def process_tensor(data, dtype=torch.float32):
            if isinstance(data, torch.Tensor):
                return data.detach()
            else:
                return torch.tensor(data, dtype=dtype)

        # Process 'speech' data
        speeches = [process_tensor(b["speech"]) for b in batch]
        packed_batch_features["speech_len"] = torch.tensor(
            [len(s) for s in speeches], dtype=torch.long
        )
        packed_batch_features["speech"] = pad_sequence(
            speeches, batch_first=True, padding_value=0
        )

        # right-padding 'phone' data
        phones = [process_tensor(b["phone"], dtype=torch.long) for b in batch]
        packed_batch_features["phone_lens"] = torch.tensor(
            [len(phone) for phone in phones], dtype=torch.long
        )
        packed_batch_features["phone_ids"] = pad_sequence(
            phones, batch_first=True, padding_value=0
        )

        # # Process 'phone' data, with left padding
        # phones = [process_tensor(b['phone'], dtype=torch.long).flip(0) for b in batch] # first reverse the whole sequence
        # packed_batch_features['phone_lens'] = torch.tensor([len(phone) for phone in phones], dtype=torch.long)
        # packed_batch_features['phone_ids'] = pad_sequence(phones, batch_first=True, padding_value=0) # do the right padding
        # packed_batch_features['phone_ids'] = packed_batch_features['phone_ids'].flip(1) # flip back to original order (left padding)

        return packed_batch_features
