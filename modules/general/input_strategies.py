# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# This code is modified from
# https://github.com/lifeiteng/vall-e/blob/9c69096d603ce13174fb5cb025f185e2e9b36ac7/valle/data/input_strategies.py
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Type

from lhotse import CutSet
from lhotse.dataset.collation import collate_features
from lhotse.dataset.input_strategies import (
    ExecutorType,
    PrecomputedFeatures,
    _get_executor,
)
from lhotse.utils import fastcopy


class PromptedFeatures:
    def __init__(self, prompts, features):
        self.prompts = prompts
        self.features = features

    def to(self, device):
        return PromptedFeatures(self.prompts.to(device), self.features.to(device))

    def sum(self):
        return self.features.sum()

    @property
    def ndim(self):
        return self.features.ndim

    @property
    def data(self):
        return (self.prompts, self.features)


class PromptedPrecomputedFeatures(PrecomputedFeatures):
    def __init__(
        self,
        dataset: str,
        cuts: CutSet,
        num_workers: int = 0,
        executor_type: Type[ExecutorType] = ThreadPoolExecutor,
    ) -> None:
        super().__init__(num_workers, executor_type)
        self.utt2neighbors = self._create_utt2neighbors(dataset, cuts)

    def __call__(self, cuts: CutSet) -> Tuple[PromptedFeatures, PromptedFeatures]:
        features, features_lens = self._collate_features(cuts)
        prompts, prompts_lens = self._collate_prompts(cuts)
        return PromptedFeatures(prompts, features), PromptedFeatures(
            prompts_lens, features_lens
        )

    def _create_utt2neighbors(self, dataset, cuts):
        utt2neighbors = defaultdict(lambda: [])
        utt2cut = {cut.id: cut for cut in cuts}
        if dataset.lower() == "libritts":
            self._process_libritts_dataset(utt2neighbors, utt2cut, cuts)
        elif dataset.lower() == "ljspeech":
            self._process_ljspeech_dataset(utt2neighbors, utt2cut, cuts)
        else:
            raise ValueError("Unsupported dataset")
        return utt2neighbors

    def _process_libritts_dataset(self, utt2neighbors, utt2cut, cuts):
        speaker2utts = defaultdict(lambda: [])
        for cut in cuts:
            speaker = cut.supervisions[0].speaker
            speaker2utts[speaker].append(cut.id)

        for spk, uttids in speaker2utts.items():
            sorted_uttids = sorted(uttids)
            if len(sorted_uttids) == 1:
                utt2neighbors[sorted_uttids[0]].append(utt2cut[sorted_uttids[0]])
                continue

            utt2prevutt = dict(
                zip(sorted_uttids, [sorted_uttids[1]] + sorted_uttids[:-1])
            )
            utt2postutt = dict(zip(sorted_uttids[:-1], sorted_uttids[1:]))
            for utt in sorted_uttids:
                if utt in utt2prevutt:
                    utt2neighbors[utt].append(utt2cut[utt2prevutt[utt]])
                if utt in utt2postutt:
                    utt2neighbors[utt].append(utt2cut[utt2postutt[utt]])

    def _process_ljspeech_dataset(self, utt2neighbors, utt2cut, cuts):
        uttids = [cut.id for cut in cuts]
        if len(uttids) == 1:
            utt2neighbors[uttids[0]].append(utt2cut[uttids[0]])
            return

        utt2prevutt = dict(zip(uttids, [uttids[1]] + uttids[:-1]))
        utt2postutt = dict(zip(uttids[:-1], uttids[1:]))
        for utt in uttids:
            prevutt, postutt = utt2prevutt.get(utt), utt2postutt.get(utt)
            if prevutt and utt[:5] == prevutt[:5]:
                utt2neighbors[utt].append(utt2cut[prevutt])
            if postutt and utt[:5] == postutt[:5]:
                utt2neighbors[utt].append(utt2cut[postutt])

    def _collate_features(self, cuts):
        return collate_features(
            cuts,
            executor=_get_executor(self.num_workers, executor_type=self._executor_type),
        )

    def _collate_prompts(self, cuts):
        prompts_cuts = []
        for k, cut in enumerate(cuts):
            prompts_cut = random.choice(self.utt2neighbors[cut.id])
            prompts_cuts.append(fastcopy(prompts_cut, id=f"{cut.id}-{str(k)}"))

        mini_duration = min([cut.duration for cut in prompts_cuts] + [3.0])
        prompts_cuts = CutSet(
            cuts={k: cut for k, cut in enumerate(prompts_cuts)}
        ).truncate(max_duration=mini_duration, offset_type="random", preserve_id=False)

        return collate_features(
            prompts_cuts,
            executor=_get_executor(self.num_workers, executor_type=self._executor_type),
        )
