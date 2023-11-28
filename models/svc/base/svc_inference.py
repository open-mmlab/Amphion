# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from models.base.new_inference import BaseInference
from models.svc.base.svc_dataset import SVCTestCollator, SVCTestDataset


class SVCInference(BaseInference):
    def __init__(self, args=None, cfg=None, infer_type="from_dataset"):
        BaseInference.__init__(self, args, cfg, infer_type)

    def _build_test_dataset(self):
        return SVCTestDataset, SVCTestCollator
