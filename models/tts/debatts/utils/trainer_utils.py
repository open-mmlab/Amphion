# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


def check_nan(logger, loss, y_pred, y_gt):
    if torch.any(torch.isnan(loss)):
        logger.info("out has nan: ", torch.any(torch.isnan(y_pred)))
        logger.info("y_gt has nan: ", torch.any(torch.isnan(y_gt)))
        logger.info("out: ", y_pred)
        logger.info("y_gt: ", y_gt)
        logger.info("loss = {:.4f}\n".format(loss.item()))
        exit()
