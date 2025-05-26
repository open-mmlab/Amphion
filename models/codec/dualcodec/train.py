# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Launch training scripts
"""
from omegaconf import DictConfig, OmegaConf
from typing import Optional
import hydra


def train(cfg):
    if hasattr(cfg.trainer, "trainer"):
        trainer = hydra.utils.instantiate(cfg.trainer.trainer)
    else:
        trainer = hydra.utils.instantiate(cfg.trainer)
    trainer._build_dataloader(hydra.utils.instantiate(cfg.data.dataloader))
    trainer.train_loop()


@hydra.main(
    version_base="1.3",
    config_path="./dualcodec/conf",
    config_name="dualcodec_train.yaml",
)
def main(cfg: DictConfig) -> Optional[float]:
    # train the model
    train(cfg)


if __name__ == "__main__":
    main(None)
