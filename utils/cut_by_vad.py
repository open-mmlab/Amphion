# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""This code is modified from https://github.com/facebookresearch/libri-light/blob/main/data_preparation/cut_by_vad.py"""
import pathlib
import soundfile as sf
import numpy as np
import json
import multiprocessing
import tqdm


def save(seq, fname, index, extension):
    """save audio sequences to file"""
    output = np.hstack(seq)
    file_name = fname.parent / (fname.stem + f"_{index:04}{extension}")
    fname.parent.mkdir(exist_ok=True, parents=True)
    sf.write(file_name, output, samplerate=16000)


def cut_sequence(path, vad, path_out, target_len_sec, out_extension):
    """cut audio sequences based on VAD"""
    data, samplerate = sf.read(path)

    assert len(data.shape) == 1
    assert samplerate == 16000

    to_stitch = []
    length_accumulated = 0.0

    i = 0
    # Iterate over VAD segments
    for start, end in vad:
        start_index = int(start * samplerate)
        end_index = int(end * samplerate)
        slice = data[start_index:end_index]

        # Save slices that exceed the target length or if there's already accumulated audio
        if (
            length_accumulated + (end - start) > target_len_sec
            and length_accumulated > 0
        ):
            save(to_stitch, path_out, i, out_extension)
            to_stitch = []
            i += 1
            length_accumulated = 0

        # Add the current slice to the list to be stitched
        to_stitch.append(slice)
        length_accumulated += end - start

    # Save any remaining slices
    if to_stitch:
        save(to_stitch, path_out, i, out_extension)


def cut_book(task):
    """process each book in the dataset"""
    path_book, root_out, target_len_sec, extension = task

    speaker = pathlib.Path(path_book.parent.name)

    for i, meta_file_path in enumerate(path_book.glob("*.json")):
        with open(meta_file_path, "r") as f:
            meta = json.loads(f.read())
        book_id = meta["book_meta"]["id"]
        vad = meta["voice_activity"]

        sound_file = meta_file_path.parent / (meta_file_path.stem + ".flac")

        path_out = root_out / speaker / book_id / (meta_file_path.stem)
        cut_sequence(sound_file, vad, path_out, target_len_sec, extension)


def cut_segments(
    input_dir, output_dir, target_len_sec=30, n_process=32, out_extension=".wav"
):
    """Main function to cut segments from audio files"""

    pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)
    list_dir = pathlib.Path(input_dir).glob("*/*")
    list_dir = [x for x in list_dir if x.is_dir()]

    print(f"{len(list_dir)} directories detected")
    print(f"Launching {n_process} processes")

    # Create tasks for multiprocessing
    tasks = [
        (path_book, output_dir, target_len_sec, out_extension) for path_book in list_dir
    ]

    # Process tasks in parallel using multiprocessing
    with multiprocessing.Pool(processes=n_process) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(cut_book, tasks), total=len(tasks)):
            pass


if __name__ == "__main__":
    input_dir = "/path/to/input_dir"
    output_dir = "/path/to/output_dir"
    target_len_sec = 10
    n_process = 16
    cut_segments(input_dir, output_dir, target_len_sec, n_process)
