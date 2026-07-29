# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Correctness-first building blocks for deterministic CP attention.

This module contains no distributed communication.  It computes the normalized
attention output and attention-domain LSE for one logical KV block, then merges
multiple block results in stable ``global_block_index`` order.  A later
distributed wrapper is responsible for exchanging KV blocks across a CP group.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Real
from typing import Iterable, Sequence

import torch
import torch.distributed as dist

from rl_engine.kernels.attention_contract import (
    AttentionContract,
    AttentionDType,
    AttentionMode,
)


@dataclass(frozen=True)
class AttentionPartial:
    """Normalized attention state produced for one logical KV block."""

    global_block_index: int
    out: torch.Tensor
    lse: torch.Tensor

    def __post_init__(self) -> None:
        if (
            isinstance(self.global_block_index, bool)
            or not isinstance(self.global_block_index, int)
            or self.global_block_index < 0
        ):
            raise ValueError("global_block_index must be a non-negative integer")
        if not isinstance(self.out, torch.Tensor) or not isinstance(
            self.lse, torch.Tensor
        ):
            raise TypeError("out and lse must be torch.Tensor instances")
        if self.out.ndim != 4:
            raise ValueError(
                f"out must have shape [B, Hq, Sq, D]; got {tuple(self.out.shape)}"
            )
        if self.lse.shape != self.out.shape[:-1]:
            raise ValueError(
                "lse must have shape [B, Hq, Sq] matching out; "
                f"got out={tuple(self.out.shape)} and lse={tuple(self.lse.shape)}"
            )
        if self.out.device != self.lse.device:
            raise ValueError("out and lse must be on the same device")
        if not self.out.is_floating_point() or not self.lse.is_floating_point():
            raise TypeError("out and lse must have floating-point dtypes")


class DeterministicCPAttentionOp:
    """Contract-bound prefill wrapper around the distributed CP reference."""

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        contract: AttentionContract,
        cp_group: dist.ProcessGroup,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(q, k, v, contract=contract, cp_group=cp_group)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        contract: AttentionContract,
        cp_group: dist.ProcessGroup,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(contract, AttentionContract):
            raise TypeError("contract must be an AttentionContract")
        if contract.mode is not AttentionMode.PREFILL:
            raise ValueError(
                "the deterministic CP reference currently supports prefill only"
            )
        if contract.sharding.packed_sequence_offsets is not None:
            raise ValueError(
                "the deterministic CP reference does not support packed varlen input"
            )
        if contract.kv_cache is not None:
            raise ValueError(
                "the deterministic CP prefill reference does not accept KV-cache input"
            )
        if any(offset != 0 for offset in contract.causal_offsets or ()):
            raise ValueError(
                "the first deterministic CP reference requires causal_offsets=0"
            )
        if len(contract.sharding.global_block_indices) != 1:
            raise ValueError(
                "each rank must own exactly one CP block in the first reference"
            )

        expected_dtype = {
            AttentionDType.BF16: torch.bfloat16,
            AttentionDType.FP32: torch.float32,
        }.get(contract.dtype)
        if expected_dtype is None or q.dtype is not expected_dtype:
            raise ValueError(
                f"contract dtype={contract.dtype.value} does not match q dtype={q.dtype}"
            )
        sharding = contract.sharding
        expected_q_shape = (
            contract.batch_size,
            sharding.local_q_heads,
            contract.query_sequence_length,
            contract.head_dim,
        )
        expected_kv_shape = (
            contract.batch_size,
            sharding.local_kv_heads,
            sharding.local_sequence_length,
            contract.head_dim,
        )
        if tuple(q.shape) != expected_q_shape:
            raise ValueError(
                f"q shape must match contract {expected_q_shape}; got {tuple(q.shape)}"
            )
        if tuple(k.shape) != expected_kv_shape or tuple(v.shape) != expected_kv_shape:
            raise ValueError(
                "k/v shapes must match contract "
                f"{expected_kv_shape}; got k={tuple(k.shape)}, v={tuple(v.shape)}"
            )
        if dist.get_world_size(group=cp_group) != sharding.cp_world_size:
            raise ValueError(
                "cp_group world size does not match contract cp_world_size"
            )

        block_start = sharding.global_block_token_starts[0]
        q_positions = torch.arange(
            block_start,
            block_start + q.shape[2],
            device=q.device,
        )
        k_positions = torch.arange(
            block_start,
            block_start + k.shape[2],
            device=k.device,
        )
        out, lse = distributed_cp_attention(
            q,
            k,
            v,
            global_block_index=sharding.global_block_indices[0],
            q_positions=q_positions,
            k_positions=k_positions,
            cp_group=cp_group,
            causal=contract.causal,
        )
        return out.to(q.dtype), lse


def partial_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    global_block_index: int,
    q_positions: torch.Tensor | Sequence[int],
    k_positions: torch.Tensor | Sequence[int],
    causal: bool = True,
    scale: float | None = None,
) -> AttentionPartial:
    """Compute FP32 ``(out, lse)`` for one KV block.

    Inputs use the WS1 reference layout ``[B, H, S, D]``.  Position arrays are
    global logical token positions shared by every batch entry in this first
    prefill milestone.  Fully masked rows produce ``out=0`` and ``lse=-inf``.
    """

    q_positions_t, k_positions_t = _validate_partial_inputs(
        q,
        k,
        v,
        q_positions=q_positions,
        k_positions=k_positions,
        causal=causal,
        scale=scale,
    )
    head_dim = q.shape[-1]
    resolved_scale = float(scale) if scale is not None else 1.0 / math.sqrt(head_dim)

    with _strict_fp32_math(q.device.type):
        qf = q.float()
        kf = k.float()
        vf = v.float()

        if qf.shape[1] != kf.shape[1]:
            group_size = qf.shape[1] // kf.shape[1]
            kf = kf.repeat_interleave(group_size, dim=1)
            vf = vf.repeat_interleave(group_size, dim=1)

        scores = torch.matmul(qf, kf.transpose(-1, -2)) * resolved_scale
        if causal:
            causal_mask = k_positions_t.unsqueeze(0) > q_positions_t.unsqueeze(1)
            scores = scores.masked_fill(causal_mask[None, None, :, :], float("-inf"))

        row_has_visible_key = torch.isfinite(scores).any(dim=-1)
        safe_scores = torch.where(
            row_has_visible_key.unsqueeze(-1),
            scores,
            torch.zeros_like(scores),
        )
        lse = torch.logsumexp(safe_scores, dim=-1)
        probabilities = torch.exp(safe_scores - lse.unsqueeze(-1))
        probabilities = torch.where(
            row_has_visible_key.unsqueeze(-1),
            probabilities,
            torch.zeros_like(probabilities),
        )
        out = torch.matmul(probabilities, vf)
        lse = torch.where(
            row_has_visible_key,
            lse,
            torch.full_like(lse, float("-inf")),
        )

    return AttentionPartial(global_block_index=global_block_index, out=out, lse=lse)


def merge_attention_partials(
    partials: Iterable[AttentionPartial],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge normalized block states in fixed logical block order.

    The returned output and LSE remain FP32.  The distributed operator performs
    the contract's final-write downcast after every CP block has been merged.
    """

    ordered = tuple(partials)
    if not ordered:
        raise ValueError("partials must contain at least one AttentionPartial")
    if any(not isinstance(partial, AttentionPartial) for partial in ordered):
        raise TypeError("partials must contain only AttentionPartial instances")

    ordered = tuple(sorted(ordered, key=lambda partial: partial.global_block_index))
    block_indices = tuple(partial.global_block_index for partial in ordered)
    if len(set(block_indices)) != len(block_indices):
        raise ValueError(
            f"global_block_index values must be unique; got {block_indices}"
        )

    reference_out = ordered[0].out
    reference_lse = ordered[0].lse
    for partial in ordered[1:]:
        if (
            partial.out.shape != reference_out.shape
            or partial.lse.shape != reference_lse.shape
        ):
            raise ValueError(
                "all attention partials must have identical out and lse shapes"
            )
        if partial.out.device != reference_out.device:
            raise ValueError("all attention partials must be on the same device")

    out_acc = reference_out.float()
    lse_acc = reference_lse.float()
    for partial in ordered[1:]:
        block_out = partial.out.float()
        block_lse = partial.lse.float()
        merged_lse = torch.logaddexp(lse_acc, block_lse)

        merged_is_finite = torch.isfinite(merged_lse)
        acc_is_finite = torch.isfinite(lse_acc) & merged_is_finite
        block_is_finite = torch.isfinite(block_lse) & merged_is_finite
        safe_merged_lse = torch.where(
            merged_is_finite,
            merged_lse,
            torch.zeros_like(merged_lse),
        )
        safe_acc_lse = torch.where(acc_is_finite, lse_acc, torch.zeros_like(lse_acc))
        safe_block_lse = torch.where(
            block_is_finite, block_lse, torch.zeros_like(block_lse)
        )
        acc_weight = torch.where(
            acc_is_finite,
            torch.exp(safe_acc_lse - safe_merged_lse),
            torch.zeros_like(merged_lse),
        )
        block_weight = torch.where(
            block_is_finite,
            torch.exp(safe_block_lse - safe_merged_lse),
            torch.zeros_like(merged_lse),
        )

        out_acc = out_acc * acc_weight.unsqueeze(
            -1
        ) + block_out * block_weight.unsqueeze(-1)
        lse_acc = merged_lse

    return out_acc, lse_acc


def distributed_cp_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    global_block_index: int,
    q_positions: torch.Tensor | Sequence[int],
    k_positions: torch.Tensor | Sequence[int],
    cp_group: dist.ProcessGroup,
    causal: bool = True,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather equal-sized KV blocks in one CP group and merge their states.

    This correctness path intentionally gathers materialized K/V blocks rather
    than using a collective as a numerical reducer.  Gather order may follow
    process-group rank order; the explicit block ids restore logical merge order.
    """

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized for CP attention")
    if cp_group is None:
        raise ValueError("cp_group must be an explicit TP-column process group")
    _validate_partial_inputs(
        q,
        k,
        v,
        q_positions=q_positions,
        k_positions=k_positions,
        causal=causal,
        scale=scale,
    )

    cp_world_size = dist.get_world_size(group=cp_group)
    local_shape = torch.tensor(k.shape, device=k.device, dtype=torch.int64)
    gathered_shapes = [torch.empty_like(local_shape) for _ in range(cp_world_size)]
    dist.all_gather(gathered_shapes, local_shape, group=cp_group)
    shape_rows = tuple(
        tuple(int(value) for value in shape.tolist()) for shape in gathered_shapes
    )
    if any(shape != shape_rows[0] for shape in shape_rows[1:]):
        raise ValueError(
            "the first distributed CP reference requires equal K/V block shapes; "
            f"got {shape_rows}"
        )

    gathered_k = [torch.empty_like(k) for _ in range(cp_world_size)]
    gathered_v = [torch.empty_like(v) for _ in range(cp_world_size)]
    dist.all_gather(gathered_k, k.contiguous(), group=cp_group)
    dist.all_gather(gathered_v, v.contiguous(), group=cp_group)

    local_block = torch.tensor([global_block_index], device=k.device, dtype=torch.int64)
    gathered_blocks = [torch.empty_like(local_block) for _ in range(cp_world_size)]
    dist.all_gather(gathered_blocks, local_block, group=cp_group)

    local_k_positions = torch.as_tensor(k_positions, device=k.device, dtype=torch.int64)
    gathered_k_positions = [
        torch.empty_like(local_k_positions) for _ in range(cp_world_size)
    ]
    dist.all_gather(gathered_k_positions, local_k_positions, group=cp_group)

    partials = tuple(
        partial_attention(
            q,
            block_k,
            block_v,
            global_block_index=int(block_index.item()),
            q_positions=q_positions,
            k_positions=block_positions,
            causal=causal,
            scale=scale,
        )
        for block_k, block_v, block_index, block_positions in zip(
            gathered_k,
            gathered_v,
            gathered_blocks,
            gathered_k_positions,
            strict=True,
        )
    )
    return merge_attention_partials(partials)


def _validate_partial_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_positions: torch.Tensor | Sequence[int],
    k_positions: torch.Tensor | Sequence[int],
    causal: bool,
    scale: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not all(isinstance(tensor, torch.Tensor) for tensor in (q, k, v)):
        raise TypeError("q, k, and v must be torch.Tensor instances")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must use [B, H, S, D] layout")
    if k.shape != v.shape:
        raise ValueError(
            f"k and v must have identical shapes; got {k.shape} and {v.shape}"
        )
    if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k/v must have matching batch size and head dimension")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError(
            f"Hq={q.shape[1]} must be divisible by Hkv={k.shape[1]} for GQA"
        )
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if (
        not q.is_floating_point()
        or not k.is_floating_point()
        or not v.is_floating_point()
    ):
        raise TypeError("q, k, and v must have floating-point dtypes")
    if not isinstance(causal, bool):
        raise TypeError("causal must be a bool")
    if scale is not None and (
        isinstance(scale, bool)
        or not isinstance(scale, Real)
        or not math.isfinite(float(scale))
    ):
        raise ValueError("scale must be a finite real number or None")

    q_positions_t = torch.as_tensor(q_positions, device=q.device, dtype=torch.long)
    k_positions_t = torch.as_tensor(k_positions, device=q.device, dtype=torch.long)
    if q_positions_t.ndim != 1 or q_positions_t.numel() != q.shape[2]:
        raise ValueError(
            f"q_positions must have shape [{q.shape[2]}]; got {tuple(q_positions_t.shape)}"
        )
    if k_positions_t.ndim != 1 or k_positions_t.numel() != k.shape[2]:
        raise ValueError(
            f"k_positions must have shape [{k.shape[2]}]; got {tuple(k_positions_t.shape)}"
        )
    return q_positions_t, k_positions_t


@contextmanager
def _strict_fp32_math(device_type: str):
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.autocast(device_type=device_type, enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


__all__ = [
    "AttentionPartial",
    "DeterministicCPAttentionOp",
    "distributed_cp_attention",
    "merge_attention_partials",
    "partial_attention",
]
