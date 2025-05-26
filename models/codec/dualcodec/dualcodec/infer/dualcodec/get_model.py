# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

model_id_to_fname = {
    "12hz_v1": "dualcodec_12hz_16384_4096.safetensors",
    "25hz_v1": "dualcodec_25hz_16384_1024.safetensors",
}
model_id_to_cfgname = {
    "12hz_v1": "dualcodec_12hz_16384_4096_8vq.yaml",
    "25hz_v1": "dualcodec_25hz_16384_1024_12vq.yaml",
}

from cached_path import cached_path


def get_model(model_id="12hz_v1", pretrained_model_path="hf://amphion/dualcodec"):
    import os

    # import importlib.resources as pkg_resources
    # conf_dir = pkg_resources.files("dualcodec") / "conf/model"
    pretrained_model_path = cached_path(pretrained_model_path)

    import hydra
    from hydra import initialize

    with initialize(version_base="1.3", config_path="../../conf/model"):
        cfg = hydra.compose(config_name=model_id_to_cfgname[model_id], overrides=[])
        model = hydra.utils.instantiate(cfg.model)

    if pretrained_model_path is None:
        import warnings

        warnings.warn(
            "pretrained_model_path is not given, model will be loaded without weights"
        )
    else:
        model_fname = os.path.join(pretrained_model_path, model_id_to_fname[model_id])
        print("Loading model from", model_fname)
        import safetensors
        import safetensors.torch

        safetensors.torch.load_model(model, model_fname)
        print("Model loaded")
    model.eval()
    return model
