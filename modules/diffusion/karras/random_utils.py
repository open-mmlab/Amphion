# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch as th


def get_generator(generator, num_samples=0, seed=0):
    if generator == "dummy":
        return DummyGenerator()
    elif generator == "determ":
        return DeterministicGenerator(num_samples, seed)
    elif generator == "determ-indiv":
        return DeterministicIndividualGenerator(num_samples, seed)
    else:
        raise NotImplementedError


class DummyGenerator:
    def randn(self, *args, **kwargs):
        return th.randn(*args, **kwargs)

    def randint(self, *args, **kwargs):
        return th.randint(*args, **kwargs)

    def randn_like(self, *args, **kwargs):
        return th.randn_like(*args, **kwargs)


class DeterministicGenerator:
    """
    RNG to deterministically sample num_samples samples that does not depend on batch_size or mpi_machines
    Uses a single rng and samples num_samples sized randomness and subsamples the current indices
    """

    def __init__(self, num_samples, seed=0):
        print("Warning: Distributed not initialised, using single rank")
        self.rank = 0
        self.world_size = 1
        self.num_samples = num_samples
        self.done_samples = 0
        self.seed = seed
        self.rng_cpu = th.Generator()
        if th.cuda.is_available():
            self.rng_cuda = th.Generator(dist_util.dev())
        self.set_seed(seed)

    def get_global_size_and_indices(self, size):
        global_size = (self.num_samples, *size[1:])
        indices = th.arange(
            self.done_samples + self.rank,
            self.done_samples + self.world_size * int(size[0]),
            self.world_size,
        )
        indices = th.clamp(indices, 0, self.num_samples - 1)
        assert (
            len(indices) == size[0]
        ), f"rank={self.rank}, ws={self.world_size}, l={len(indices)}, bs={size[0]}"
        return global_size, indices

    def get_generator(self, device):
        return self.rng_cpu if th.device(device).type == "cpu" else self.rng_cuda

    def randn(self, *size, dtype=th.float, device="cpu"):
        global_size, indices = self.get_global_size_and_indices(size)
        generator = self.get_generator(device)
        return th.randn(*global_size, generator=generator, dtype=dtype, device=device)[
            indices
        ]

    def randint(self, low, high, size, dtype=th.long, device="cpu"):
        global_size, indices = self.get_global_size_and_indices(size)
        generator = self.get_generator(device)
        return th.randint(
            low, high, generator=generator, size=global_size, dtype=dtype, device=device
        )[indices]

    def randn_like(self, tensor):
        size, dtype, device = tensor.size(), tensor.dtype, tensor.device
        return self.randn(*size, dtype=dtype, device=device)

    def set_done_samples(self, done_samples):
        self.done_samples = done_samples
        self.set_seed(self.seed)

    def get_seed(self):
        return self.seed

    def set_seed(self, seed):
        self.rng_cpu.manual_seed(seed)
        if th.cuda.is_available():
            self.rng_cuda.manual_seed(seed)


class DeterministicIndividualGenerator:
    """
    RNG to deterministically sample num_samples samples that does not depend on batch_size or mpi_machines
    Uses a separate rng for each sample to reduce memoery usage
    """

    def __init__(self, num_samples, seed=0):
        print("Warning: Distributed not initialised, using single rank")
        self.rank = 0
        self.world_size = 1
        self.num_samples = num_samples
        self.done_samples = 0
        self.seed = seed
        self.rng_cpu = [th.Generator() for _ in range(num_samples)]
        if th.cuda.is_available():
            self.rng_cuda = [th.Generator(dist_util.dev()) for _ in range(num_samples)]
        self.set_seed(seed)

    def get_size_and_indices(self, size):
        indices = th.arange(
            self.done_samples + self.rank,
            self.done_samples + self.world_size * int(size[0]),
            self.world_size,
        )
        indices = th.clamp(indices, 0, self.num_samples - 1)
        assert (
            len(indices) == size[0]
        ), f"rank={self.rank}, ws={self.world_size}, l={len(indices)}, bs={size[0]}"
        return (1, *size[1:]), indices

    def get_generator(self, device):
        return self.rng_cpu if th.device(device).type == "cpu" else self.rng_cuda

    def randn(self, *size, dtype=th.float, device="cpu"):
        size, indices = self.get_size_and_indices(size)
        generator = self.get_generator(device)
        return th.cat(
            [
                th.randn(*size, generator=generator[i], dtype=dtype, device=device)
                for i in indices
            ],
            dim=0,
        )

    def randint(self, low, high, size, dtype=th.long, device="cpu"):
        size, indices = self.get_size_and_indices(size)
        generator = self.get_generator(device)
        return th.cat(
            [
                th.randint(
                    low,
                    high,
                    generator=generator[i],
                    size=size,
                    dtype=dtype,
                    device=device,
                )
                for i in indices
            ],
            dim=0,
        )

    def randn_like(self, tensor):
        size, dtype, device = tensor.size(), tensor.dtype, tensor.device
        return self.randn(*size, dtype=dtype, device=device)

    def set_done_samples(self, done_samples):
        self.done_samples = done_samples

    def get_seed(self):
        return self.seed

    def set_seed(self, seed):
        [
            rng_cpu.manual_seed(i + self.num_samples * seed)
            for i, rng_cpu in enumerate(self.rng_cpu)
        ]
        if th.cuda.is_available():
            [
                rng_cuda.manual_seed(i + self.num_samples * seed)
                for i, rng_cuda in enumerate(self.rng_cuda)
            ]
