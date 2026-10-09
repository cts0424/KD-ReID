"""PK sampler: each batch holds P identities x K instances, as batch-hard triplet loss needs."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator

from torch.utils.data import Sampler


class RandomIdentitySampler(Sampler[int]):
    def __init__(self, items: list[tuple[str, int, int]], batch_size: int, num_instances: int):
        if batch_size % num_instances != 0:
            raise ValueError("batch_size must be divisible by num_instances")
        self.num_instances = num_instances
        self.pids_per_batch = batch_size // num_instances
        self.index_by_pid: dict[int, list[int]] = defaultdict(list)
        for idx, (_, pid, _) in enumerate(items):
            self.index_by_pid[pid].append(idx)
        self.pids = list(self.index_by_pid)
        # Estimated epoch length (rounded down to whole PK groups)
        self._length = sum(
            max(len(v), num_instances) // num_instances * num_instances
            for v in self.index_by_pid.values()
        )

    def _chunks(self) -> dict[int, list[list[int]]]:
        out: dict[int, list[list[int]]] = {}
        for pid, idxs in self.index_by_pid.items():
            idxs = list(idxs)
            if len(idxs) < self.num_instances:
                idxs = random.choices(idxs, k=self.num_instances)
            random.shuffle(idxs)
            k = self.num_instances
            out[pid] = [idxs[i : i + k] for i in range(0, len(idxs) - k + 1, k)]
        return out

    def __iter__(self) -> Iterator[int]:
        chunks = self._chunks()
        avail = [p for p in self.pids if chunks[p]]
        result: list[int] = []
        while len(avail) >= self.pids_per_batch:
            for pid in random.sample(avail, self.pids_per_batch):
                result.extend(chunks[pid].pop(0))
                if not chunks[pid]:
                    avail.remove(pid)
        self._length = len(result)
        return iter(result)

    def __len__(self) -> int:
        return self._length
