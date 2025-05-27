from setuptools import setup

setup(
    name="dualcodec",
    packages=["dualcodec"],
    install_requires=[
        "transformers>=4.30.0",
        "descript-audiotools>=0.7.2",
        "huggingface_hub[cli]",
        "easydict",
        "torch",
        "torchaudio",
        "hydra-core",
        "einops",
        "safetensors",
        "cached_path",
    ],
    python_requires=">=3.9",
)
