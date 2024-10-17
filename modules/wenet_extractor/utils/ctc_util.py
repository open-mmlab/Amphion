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

import numpy as np
import torch


def insert_blank(label, blank_id=0):
    """Insert blank token between every two label token."""
    label = np.expand_dims(label, 1)
    blanks = np.zeros((label.shape[0], 1), dtype=np.int64) + blank_id
    label = np.concatenate([blanks, label], axis=1)
    label = label.reshape(-1)
    label = np.append(label, label[0])
    return label


def forced_align(ctc_probs: torch.Tensor, y: torch.Tensor, blank_id=0) -> list:
    """ctc forced alignment.

    Args:
        torch.Tensor ctc_probs: hidden state sequence, 2d tensor (T, D)
        torch.Tensor y: id sequence tensor 1d tensor (L)
        int blank_id: blank symbol index
    Returns:
        torch.Tensor: alignment result
    """
    y_insert_blank = insert_blank(y, blank_id)

    log_alpha = torch.zeros((ctc_probs.size(0), len(y_insert_blank)))
    log_alpha = log_alpha - float("inf")  # log of zero
    state_path = (
        torch.zeros((ctc_probs.size(0), len(y_insert_blank)), dtype=torch.int16) - 1
    )  # state path

    # init start state
    log_alpha[0, 0] = ctc_probs[0][y_insert_blank[0]]
    log_alpha[0, 1] = ctc_probs[0][y_insert_blank[1]]

    for t in range(1, ctc_probs.size(0)):
        for s in range(len(y_insert_blank)):
            if (
                y_insert_blank[s] == blank_id
                or s < 2
                or y_insert_blank[s] == y_insert_blank[s - 2]
            ):
                candidates = torch.tensor(
                    [log_alpha[t - 1, s], log_alpha[t - 1, s - 1]]
                )
                prev_state = [s, s - 1]
            else:
                candidates = torch.tensor(
                    [
                        log_alpha[t - 1, s],
                        log_alpha[t - 1, s - 1],
                        log_alpha[t - 1, s - 2],
                    ]
                )
                prev_state = [s, s - 1, s - 2]
            log_alpha[t, s] = torch.max(candidates) + ctc_probs[t][y_insert_blank[s]]
            state_path[t, s] = prev_state[torch.argmax(candidates)]

    state_seq = -1 * torch.ones((ctc_probs.size(0), 1), dtype=torch.int16)

    candidates = torch.tensor(
        [log_alpha[-1, len(y_insert_blank) - 1], log_alpha[-1, len(y_insert_blank) - 2]]
    )
    final_state = [len(y_insert_blank) - 1, len(y_insert_blank) - 2]
    state_seq[-1] = final_state[torch.argmax(candidates)]
    for t in range(ctc_probs.size(0) - 2, -1, -1):
        state_seq[t] = state_path[t + 1, state_seq[t + 1, 0]]

    output_alignment = []
    for t in range(0, ctc_probs.size(0)):
        output_alignment.append(y_insert_blank[state_seq[t, 0]])

    return output_alignment
