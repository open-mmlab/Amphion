# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This code is modified from https://github.com/descriptinc/descript-audio-codec/blob/main/dac/__init__.py

__version__ = "1.0.0"

# preserved here for legacy reasons
__model_version__ = "latest"

# import audiotools

# audiotools.ml.BaseModel.INTERN += ["dac.**"]
# audiotools.ml.BaseModel.EXTERN += ["einops"]


from . import nn
from . import model
from .model import DAC
from .model import DACFile
