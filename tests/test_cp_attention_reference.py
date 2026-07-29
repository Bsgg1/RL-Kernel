# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Focused tests for the deterministic CP attention math building blocks."""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from rl_engine.kernels.ops.pytorch.attention.cp_attention import (
    AttentionPartial,
    merge_attention_partials,
    partial_attention,
)


def _qkv(*, seed: int = 0, sq: int = 4, skv: int = 6):
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(1, 4, sq, 8, generator=generator)
    k = torch.randn(1, 2, skv, 8, generator=generator)
    v = torch.randn(1, 2, skv, 8, generator=generator)
    return q, k, v


def _full_reference(q, k, v, *, q_positions, k_positions):
    k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(
        q.shape[-1]
    )
    mask = torch.as_tensor(k_positions).unsqueeze(0) > torch.as_tensor(
        q_positions
    ).unsqueeze(1)
    scores = scores.masked_fill(mask[None, None, :, :], float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    out = torch.softmax(scores, dim=-1) @ v.float()
    return out, lse


def test_partial_attention_matches_one_block_reference():
    q, k, v = _qkv(seed=1, sq=4, skv=4)
    positions = tuple(range(4))

    partial = partial_attention(
        q,
        k,
        v,
        global_block_index=0,
        q_positions=positions,
        k_positions=positions,
    )
    expected_out, expected_lse = _full_reference(
        q,
        k,
        v,
        q_positions=positions,
        k_positions=positions,
    )

    assert partial.out.dtype is torch.float32
    assert partial.lse.dtype is torch.float32
    torch.testing.assert_close(partial.out, expected_out, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(partial.lse, expected_lse, atol=1e-6, rtol=1e-6)


def test_future_only_block_produces_zero_output_and_negative_infinite_lse():
    q, k, v = _qkv(seed=2, sq=2, skv=2)
    partial = partial_attention(
        q,
        k,
        v,
        global_block_index=1,
        q_positions=(0, 1),
        k_positions=(2, 3),
    )

    assert torch.equal(partial.out, torch.zeros_like(partial.out))
    assert torch.isneginf(partial.lse).all()


def test_two_block_merge_matches_full_attention_and_sorts_logical_order():
    q, k, v = _qkv(seed=3, sq=3, skv=6)
    q_positions = (3, 4, 5)
    first = partial_attention(
        q,
        k[:, :, :3],
        v[:, :, :3],
        global_block_index=0,
        q_positions=q_positions,
        k_positions=(0, 1, 2),
    )
    second = partial_attention(
        q,
        k[:, :, 3:],
        v[:, :, 3:],
        global_block_index=1,
        q_positions=q_positions,
        k_positions=(3, 4, 5),
    )

    actual_out, actual_lse = merge_attention_partials((second, first))
    expected_out, expected_lse = _full_reference(
        q,
        k,
        v,
        q_positions=q_positions,
        k_positions=tuple(range(6)),
    )

    torch.testing.assert_close(actual_out, expected_out, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual_lse, expected_lse, atol=1e-6, rtol=1e-6)


def test_merge_handles_all_masked_accumulator_without_nan():
    out = torch.zeros(1, 1, 2, 3)
    masked = AttentionPartial(
        global_block_index=0,
        out=out,
        lse=torch.full((1, 1, 2), float("-inf")),
    )
    visible = AttentionPartial(
        global_block_index=1,
        out=torch.ones_like(out),
        lse=torch.zeros(1, 1, 2),
    )

    merged_out, merged_lse = merge_attention_partials((masked, visible))

    assert torch.equal(merged_out, visible.out)
    assert torch.equal(merged_lse, visible.lse)
    assert not torch.isnan(merged_out).any()


def test_merge_rejects_duplicate_global_block_indices():
    partial = AttentionPartial(
        global_block_index=0,
        out=torch.zeros(1, 1, 1, 1),
        lse=torch.zeros(1, 1, 1),
    )

    with pytest.raises(ValueError, match="must be unique"):
        merge_attention_partials((partial, partial))


@pytest.mark.parametrize("tp_world_size", (1, 2))
@pytest.mark.parametrize("cp_world_size", (1, 2, 4))
def test_qwen3_tp_cp_shards_reconstruct_full_bf16_attention(
    tp_world_size: int,
    cp_world_size: int,
):
    """Every TP/CP rank slice matches the corresponding full-attention slice."""

    batch, sequence, q_heads, kv_heads, head_dim = 2, 16, 32, 8, 128
    generator = torch.Generator().manual_seed(11)
    q_global = torch.randn(
        batch, q_heads, sequence, head_dim, generator=generator, dtype=torch.bfloat16
    )
    k_global = torch.randn(
        batch, kv_heads, sequence, head_dim, generator=generator, dtype=torch.bfloat16
    )
    v_global = torch.randn(
        batch, kv_heads, sequence, head_dim, generator=generator, dtype=torch.bfloat16
    )
    position_blocks = torch.tensor_split(torch.arange(sequence), cp_world_size)
    q_sequence_blocks = torch.tensor_split(q_global, cp_world_size, dim=2)
    k_sequence_blocks = torch.tensor_split(k_global, cp_world_size, dim=2)
    v_sequence_blocks = torch.tensor_split(v_global, cp_world_size, dim=2)
    expected_global_out, expected_global_lse = _full_reference(
        q_global,
        k_global,
        v_global,
        q_positions=tuple(range(sequence)),
        k_positions=tuple(range(sequence)),
    )

    outputs_by_cp: list[torch.Tensor] = []
    lse_by_cp: list[torch.Tensor] = []
    for cp_rank in range(cp_world_size):
        outputs_by_tp: list[torch.Tensor] = []
        lse_by_tp: list[torch.Tensor] = []
        for tp_rank in range(tp_world_size):
            q_head_start = tp_rank * (q_heads // tp_world_size)
            q_head_end = (tp_rank + 1) * (q_heads // tp_world_size)
            kv_head_start = tp_rank * (kv_heads // tp_world_size)
            kv_head_end = (tp_rank + 1) * (kv_heads // tp_world_size)
            q_local = q_sequence_blocks[cp_rank][:, q_head_start:q_head_end]
            q_positions = position_blocks[cp_rank]

            partials = []
            for block_index in range(cp_world_size):
                partials.append(
                    partial_attention(
                        q_local,
                        k_sequence_blocks[block_index][:, kv_head_start:kv_head_end],
                        v_sequence_blocks[block_index][:, kv_head_start:kv_head_end],
                        global_block_index=block_index,
                        q_positions=q_positions,
                        k_positions=position_blocks[block_index],
                    )
                )

            local_out, local_lse = merge_attention_partials(reversed(partials))
            expected_out = expected_global_out[
                :, q_head_start:q_head_end, q_positions[0] : q_positions[-1] + 1
            ]
            expected_lse = expected_global_lse[
                :, q_head_start:q_head_end, q_positions[0] : q_positions[-1] + 1
            ]
            torch.testing.assert_close(local_out, expected_out, atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(local_lse, expected_lse, atol=2e-6, rtol=2e-6)
            assert local_out.dtype is torch.float32
            assert local_lse.dtype is torch.float32
            assert not torch.isnan(local_out).any()
            assert not torch.isnan(local_lse).any()
            outputs_by_tp.append(local_out)
            lse_by_tp.append(local_lse)

        outputs_by_cp.append(torch.cat(outputs_by_tp, dim=1))
        lse_by_cp.append(torch.cat(lse_by_tp, dim=1))

    reconstructed_out = torch.cat(outputs_by_cp, dim=2)
    reconstructed_lse = torch.cat(lse_by_cp, dim=2)
    torch.testing.assert_close(
        reconstructed_out, expected_global_out, atol=2e-6, rtol=2e-6
    )
    torch.testing.assert_close(
        reconstructed_lse, expected_global_lse, atol=2e-6, rtol=2e-6
    )


def test_uneven_cp4_blocks_match_full_attention():
    q, k, v = _qkv(seed=21, sq=10, skv=10)
    positions = torch.arange(10)
    q_positions = positions[7:]
    q_local = q[:, :, 7:]
    k_blocks = torch.tensor_split(k, 4, dim=2)
    v_blocks = torch.tensor_split(v, 4, dim=2)
    position_blocks = torch.tensor_split(positions, 4)

    partials = tuple(
        partial_attention(
            q_local,
            k_block,
            v_block,
            global_block_index=block_index,
            q_positions=q_positions,
            k_positions=position_blocks[block_index],
        )
        for block_index, (k_block, v_block) in enumerate(
            zip(k_blocks, v_blocks, strict=True)
        )
    )
    actual_out, actual_lse = merge_attention_partials(partials)
    expected_out, expected_lse = _full_reference(
        q_local,
        k,
        v,
        q_positions=q_positions,
        k_positions=positions,
    )

    assert [block.shape[2] for block in k_blocks] == [3, 3, 2, 2]
    torch.testing.assert_close(actual_out, expected_out, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual_lse, expected_lse, atol=2e-6, rtol=2e-6)


def test_cp4_merge_is_bitwise_independent_of_partial_arrival_order():
    q, k, v = _qkv(seed=31, sq=2, skv=8)
    q_positions = torch.arange(6, 8)
    k_blocks = torch.tensor_split(k, 4, dim=2)
    v_blocks = torch.tensor_split(v, 4, dim=2)
    position_blocks = torch.tensor_split(torch.arange(8), 4)
    partials = tuple(
        partial_attention(
            q,
            k_block,
            v_block,
            global_block_index=block_index,
            q_positions=q_positions,
            k_positions=position_blocks[block_index],
        )
        for block_index, (k_block, v_block) in enumerate(
            zip(k_blocks, v_blocks, strict=True)
        )
    )
    expected_out, expected_lse = merge_attention_partials(partials)

    for arrival_order in itertools.permutations(partials):
        actual_out, actual_lse = merge_attention_partials(arrival_order)
        assert torch.equal(actual_out, expected_out)
        assert torch.equal(actual_lse, expected_lse)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("q_positions", (0,), "q_positions must have shape"),
        ("k_positions", (0,), "k_positions must have shape"),
    ],
)
def test_partial_attention_rejects_position_shape_mismatch(field, value, message):
    q, k, v = _qkv(seed=41, sq=2, skv=3)
    kwargs = {
        "global_block_index": 0,
        "q_positions": (0, 1),
        "k_positions": (0, 1, 2),
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        partial_attention(q, k, v, **kwargs)
