# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class HyperParams:
    """The class to store hyperparameters. The key is case-insensitive.

    Args:
        *args: a list of dict or HyperParams.
        **kwargs: a list of key-value pairs.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HyperParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()
