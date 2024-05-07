# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import argparse
#from utils.util import load_config

def inf_preprocess(file1, file2):
    #cfg = load_config(args.config)
    source_file = file1
    target_folder1 = 'temp/temp1/temp2/song1'
    if not os.path.exists(target_folder1):
        os.makedirs(target_folder1)
    for i in range(1, 5):
        new_file_name = f'A{i}.wav'
        new_file_path = os.path.join(target_folder1, new_file_name)
        shutil.copy(source_file, new_file_path)
        print(f'Copied {source_file} to {new_file_path}')
    target_folder2 = 'temp/temp0'
    source_file = file2
    if not os.path.exists(target_folder2):
        os.makedirs(target_folder2)
    new_file_name = f'B.wav'
    new_file_path = os.path.join(target_folder2, new_file_name)
    shutil.copy(source_file, new_file_path)
    print(f'Copied {source_file} to {new_file_path}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infsource",
        type=str,
        default="source_audio",
        help="Source audio file or directory. If a JSON file is given, "
        "inference from dataset is applied. If a directory is given, "
        "inference from all wav/flac/mp3 audio files in the directory is applied. "
        "Default: inference from all wav/flac/mp3 audio files in ./source_audio",
    )
    args = parser.parse_args()
    #cfg = load_config(args.config)
    source_file = args.infsource
    target_folder = 'temp/temp1/temp2/song1'
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for i in range(1, 5):
        new_file_name = f'A{i}.wav'
        new_file_path = os.path.join(target_folder, new_file_name)
        shutil.copy(source_file, new_file_path)
        print(f'Copied {source_file} to {new_file_path}')
        
if __name__ == "__main__":
    main()