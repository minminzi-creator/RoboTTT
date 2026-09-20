# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for deterministic, sampler-owned RoboTTT sequence windows."""

from types import MethodType

from torch.utils.data.distributed import DistributedSampler

from rlinf.data.datasets.gr00t_n1d7.sequence_dataset import (
    Gr00tN1d7SequenceDataset,
)


class _FakeLoader:
    def __init__(self, lengths):
        self.episode_lengths = lengths

    def __getitem__(self, episode_id):
        return episode_id


def _make_dataset():
    dataset = Gr00tN1d7SequenceDataset.__new__(Gr00tN1d7SequenceDataset)
    dataset.num_timesteps = 2
    dataset.timestep_stride = 2
    dataset.action_horizon = 3
    dataset.loader = _FakeLoader([6, 7])
    dataset._index = dataset._build_index(dataset.loader, dataset.num_timesteps)
    dataset.samples_per_epoch = len(dataset._index)
    dataset.human_loader = None
    dataset._human_rows = {}

    def process_step(self, episode_id, step_index, *, mask_loss):
        return {
            "episode_id": episode_id,
            "step_index": step_index,
            "mask_loss": mask_loss,
        }

    dataset._process_step = MethodType(process_step, dataset)
    return dataset


def test_index_enumerates_every_valid_window():
    dataset = _make_dataset()

    assert dataset._index == [
        (0, 0, 6),
        (0, 1, 6),
        (1, 0, 7),
        (1, 1, 7),
        (1, 2, 7),
    ]


def test_getitem_uses_sampler_index_without_private_rng():
    dataset = _make_dataset()

    first = dataset[0]
    third = dataset[2]

    assert [(step["episode_id"], step["step_index"]) for step in first["steps"]] == [
        (0, 0),
        (0, 2),
    ]
    assert [(step["episode_id"], step["step_index"]) for step in third["steps"]] == [
        (1, 0),
        (1, 2),
    ]


def test_distributed_sampler_partitions_window_indices():
    dataset = _make_dataset()
    rank0 = set(
        DistributedSampler(
            dataset,
            num_replicas=2,
            rank=0,
            shuffle=False,
            drop_last=True,
        )
    )
    rank1 = set(
        DistributedSampler(
            dataset,
            num_replicas=2,
            rank=1,
            shuffle=False,
            drop_last=True,
        )
    )

    assert rank0.isdisjoint(rank1)
    assert rank0 | rank1 == set(range(4))
