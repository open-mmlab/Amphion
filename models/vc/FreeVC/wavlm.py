import os

from huggingface_hub import try_to_load_from_cache, snapshot_download
import torch
from transformers import WavLMModel

REPO_ID = "microsoft/wavlm-large"


def rename_state_key(state_dict, key, new_key):
    state_dict[new_key] = state_dict.pop(key)


def load_wavlm():
    # https://github.com/huggingface/transformers/issues/30469
    bin_name = "pytorch_model.bin"
    bin_path = try_to_load_from_cache(repo_id=REPO_ID, filename=bin_name)
    if bin_path is None:
        download_wavlm()
        bin_path = try_to_load_from_cache(repo_id=REPO_ID, filename=bin_name)
        assert bin_path is not None

    # https://github.com/pytorch/pytorch/issues/102999
    # https://pytorch.org/docs/stable/generated/torch.nn.utils.weight_norm.html
    state_dict = torch.load(bin_path)
    rename_state_key(
        state_dict,
        "encoder.pos_conv_embed.conv.weight_g",
        "encoder.pos_conv_embed.conv.parametrizations.weight.original0",
    )
    rename_state_key(
        state_dict,
        "encoder.pos_conv_embed.conv.weight_v",
        "encoder.pos_conv_embed.conv.parametrizations.weight.original1",
    )

    model = WavLMModel.from_pretrained(os.path.dirname(bin_path), state_dict=state_dict)
    assert isinstance(model, WavLMModel)
    return model


def download_wavlm():
    snapshot_download(repo_id=REPO_ID, repo_type="model", resume_download=True)


if __name__ == "__main__":
    download_wavlm()
    print(load_wavlm())
