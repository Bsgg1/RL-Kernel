# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Unit tests for deterministic TP/CP rank layout."""

import pytest

from rl_engine.kernels.distributed.attention_mesh import (
    rank_to_tp_cp,
    tp_cp_group_ranks,
)


def test_tp2_cp2_group_layout():
    tp_groups, cp_groups = tp_cp_group_ranks(tp_world_size=2, cp_world_size=2)

    assert tp_groups == ((0, 1), (2, 3))
    assert cp_groups == ((0, 2), (1, 3))
    assert [rank_to_tp_cp(rank, 2, 2) for rank in range(4)] == [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
    ]


@pytest.mark.parametrize("tp_world_size", (1, 2, 4))
@pytest.mark.parametrize("cp_world_size", (1, 2, 4))
def test_every_rank_belongs_to_one_tp_group_and_one_cp_group(
    tp_world_size, cp_world_size
):
    tp_groups, cp_groups = tp_cp_group_ranks(tp_world_size, cp_world_size)
    world_size = tp_world_size * cp_world_size

    for rank in range(world_size):
        assert sum(rank in group for group in tp_groups) == 1
        assert sum(rank in group for group in cp_groups) == 1
