"""
Selectively download data from Emilia dataset to a specified destination.

It supports download resume in case of interruption.

Example usage for downloading all JA data from Emilia-YODAS:

python3 data_downloader.py \
  --output_data_path "/mnt/Emilia-YODAS/data" \
  --emilia_token hf_xxx \
  --data_path_pattern "datasets/amphion/Emilia-Dataset/Emilia-YODAS/JA/*.tar"
"""

import argparse
import datetime
import os
from pathlib import Path

from huggingface_hub import HfApi, HfFileSystem


def download_dataset(output_data_path, emilia_token, data_path_pattern):
    os.environ["HF_DATASETS_CACHE"] = output_data_path
    os.environ["HF_HOME"] = output_data_path

    fs = HfFileSystem(token=emilia_token)
    tar_files = fs.glob(data_path_pattern)
    api = HfApi(token=emilia_token)

    print("Number of files to download:", len(tar_files))
    print("")
    for file in tar_files:
        filename = str(Path(file).relative_to("datasets/amphion/Emilia-Dataset"))
        print(
            datetime.datetime.now(),
            "downloaded file:",
            api.hf_hub_download(
                repo_id="amphion/Emilia-Dataset",
                filename=filename,
                repo_type="dataset",
                cache_dir=output_data_path,
                local_dir=output_data_path,
            ),
        )
    print("")
    print("downloading dataset complete")


def main():
    parser = argparse.ArgumentParser(
        description="Selectively download data from Emilia dataset."
    )

    parser.add_argument(
        "--output_data_path", required=True, type=str, help="Path of the output data"
    )
    parser.add_argument(
        "--emilia_token",
        required=True,
        type=str,
        help="Emilia token for authentication",
    )
    parser.add_argument(
        "--data_path_pattern", required=True, type=str, help="Data path pattern"
    )
    args = parser.parse_args()

    download_dataset(args.output_data_path, args.emilia_token, args.data_path_pattern)


if __name__ == "__main__":
    main()
