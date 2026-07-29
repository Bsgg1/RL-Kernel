# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Deterministic two-dimensional TP/CP process-group construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch.distributed as dist


def tp_cp_group_ranks(
    tp_world_size: int,
    cp_world_size: int,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Return every TP group followed by every CP group in creation order.

    Global ranks use ``rank = cp_rank * tp_world_size + tp_rank``.  Returning
    all groups in a stable order is important because every distributed process
    must call ``dist.new_group`` in the same global order.
    """

    _validate_parallel_sizes(tp_world_size, cp_world_size)
    tp_groups = tuple(
        tuple(cp_rank * tp_world_size + tp_rank for tp_rank in range(tp_world_size))
        for cp_rank in range(cp_world_size)
    )
    cp_groups = tuple(
        tuple(cp_rank * tp_world_size + tp_rank for cp_rank in range(cp_world_size))
        for tp_rank in range(tp_world_size)
    )
    return tp_groups, cp_groups


def rank_to_tp_cp(
    rank: int,
    tp_world_size: int,
    cp_world_size: int,
) -> tuple[int, int]:
    """Map one global rank to ``(tp_rank, cp_rank)`` coordinates."""

    _validate_parallel_sizes(tp_world_size, cp_world_size)
    world_size = tp_world_size * cp_world_size
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or not 0 <= rank < world_size
    ):
        raise ValueError(f"rank must be in [0, {world_size}); got {rank!r}")
    return rank % tp_world_size, rank // tp_world_size


@dataclass(frozen=True)
class AttentionParallelMesh:
    """The TP/CP coordinates and process groups owned by the current rank."""

    rank: int
    world_size: int
    tp_rank: int
    cp_rank: int
    tp_world_size: int
    cp_world_size: int
    tp_group_ranks: tuple[int, ...]
    cp_group_ranks: tuple[int, ...]
    tp_group: dist.ProcessGroup
    cp_group: dist.ProcessGroup


def create_attention_parallel_mesh(
    tp_world_size: int,
    cp_world_size: int,
) -> AttentionParallelMesh:
    """Collectively create the TP and CP groups for an initialized world."""

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed must be initialized before creating the mesh"
        )
    _validate_parallel_sizes(tp_world_size, cp_world_size)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    expected_world_size = tp_world_size * cp_world_size
    if world_size != expected_world_size:
        raise RuntimeError(
            "distributed world size must equal TP * CP; "
            f"got world_size={world_size}, TP={tp_world_size}, CP={cp_world_size}"
        )

    tp_rank, cp_rank = rank_to_tp_cp(rank, tp_world_size, cp_world_size)
    tp_groups, cp_groups = tp_cp_group_ranks(tp_world_size, cp_world_size)
    local_tp_group = None
    local_cp_group = None
    for group_ranks in (*tp_groups, *cp_groups):
        group = dist.new_group(ranks=list(group_ranks))
        if group_ranks == tp_groups[cp_rank]:
            local_tp_group = group
        if group_ranks == cp_groups[tp_rank]:
            local_cp_group = group

    if local_tp_group is None or local_cp_group is None:
        raise RuntimeError("failed to resolve local TP/CP process groups")
    return AttentionParallelMesh(
        rank=rank,
        world_size=world_size,
        tp_rank=tp_rank,
        cp_rank=cp_rank,
        tp_world_size=tp_world_size,
        cp_world_size=cp_world_size,
        tp_group_ranks=tp_groups[cp_rank],
        cp_group_ranks=cp_groups[tp_rank],
        tp_group=local_tp_group,
        cp_group=local_cp_group,
    )


def _validate_parallel_sizes(tp_world_size: int, cp_world_size: int) -> None:
    for name, value in (
        ("tp_world_size", tp_world_size),
        ("cp_world_size", cp_world_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer; got {value!r}")


__all__ = [
    "AttentionParallelMesh",
    "create_attention_parallel_mesh",
    "rank_to_tp_cp",
    "tp_cp_group_ranks",
]
