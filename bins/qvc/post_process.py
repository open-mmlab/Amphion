# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil

def main():
    target_folder = 'temp'
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)

if __name__ == "__main__":
    main()