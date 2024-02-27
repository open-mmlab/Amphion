# Mount dataset in Docker container

When using Docker to run Amphion, mount the dataset to the container first is needed. It is recommend to mounte dataset to `/mnt/<dataset_name>` in the container, where `<dataset_name>` is the name of the dataset.

When configuring the dataset in `exp_config.json`, you should use the path `/mnt/<dataset_name>` as the dataset path instead of the actual path on your host machine. Otherwise, the dataset will not be found in the container.

## Mount Example

```bash
docker run --runtime=nvidia --gpus all -it -v .:/app -v <dataset_path1>:/mnt/<dataset_name1> -v <dataset_path2>:/mnt/<dataset_name2> amphion
```

For example, if you want to use the `LJSpeech` dataset, you can mount the dataset to `/mnt/LJSpeech` in the container.

```bash
docker run --runtime=nvidia --gpus all -it -v .:/app -v /home/username/datasets/LJSpeech:/mnt/LJSpeech amphion
```

If you want to use multiple datasets, you can mount them to different directories in the container by adding more `-v` options.
