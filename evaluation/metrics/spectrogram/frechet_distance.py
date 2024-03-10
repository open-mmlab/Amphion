# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from frechet_audio_distance import FrechetAudioDistance


def extract_fad(
    audio_dir1,
    audio_dir2,
    **kwargs,
):
    """Extract Frechet Audio Distance for two given audio folders.
    audio_dir1: path to the ground truth audio folder.
    audio_dir2: path to the predicted audio folder.
    mode: "vggish", "pann", "clap" for different models.
    """
    frechet = FrechetAudioDistance(
        model_name="vggish",
        use_pca=False,
        use_activation=False,
        verbose=False,
    )

    fad_score = frechet.score(audio_dir1, audio_dir2)

    return fad_score
