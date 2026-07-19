# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""WS2 Attention CP contract and contract-aware dispatch tests (issue #235)."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rl_engine.kernels.attention_contract import (
    AttentionBackendCapability,
    AttentionContract,
    AttentionContractError,
    AttentionDType,
    AttentionMode,
    AttentionRole,
    KVCacheSpec,
    ReductionSpec,
    ShardingSpec,
)
from rl_engine.kernels.registry import KernelRegistry, OpBackend


def _sharding(
    *,
    tp_rank: int = 0,
    tp_world_size: int = 4,
    cp_rank: int = 0,
    cp_world_size: int = 4,
    global_sequence_length: int = 4096,
    local_sequence_length: int = 1024,
    global_block_indices: tuple[int, ...] = (0,),
    global_block_token_starts: tuple[int, ...] = (0,),
    local_block_offsets: tuple[int, ...] = (0, 1024),
    packed_sequence_offsets: tuple[int, ...] | None = None,
) -> ShardingSpec:
    local_q_heads = 32 // tp_world_size
    local_kv_heads = 8 // tp_world_size
    return ShardingSpec(
        tp_rank=tp_rank,
        tp_world_size=tp_world_size,
        cp_rank=cp_rank,
        cp_world_size=cp_world_size,
        global_q_heads=32,
        global_kv_heads=8,
        local_q_head_start=tp_rank * local_q_heads,
        local_q_heads=local_q_heads,
        local_kv_head_start=tp_rank * local_kv_heads,
        local_kv_heads=local_kv_heads,
        global_sequence_length=global_sequence_length,
        local_sequence_length=local_sequence_length,
        global_block_indices=global_block_indices,
        global_block_token_starts=global_block_token_starts,
        local_block_offsets=local_block_offsets,
        packed_sequence_offsets=packed_sequence_offsets,
    )


def _contract(
    *,
    role: str = "infer",
    mode: str = "prefill",
    sharding: ShardingSpec | None = None,
    kv_cache: KVCacheSpec | None = None,
    causal_offsets: tuple[int, ...] = (0,),
    batch_size: int = 1,
) -> AttentionContract:
    resolved_sharding = sharding or _sharding()
    return AttentionContract(
        role=role,
        mode=mode,
        dtype="bf16",
        batch_size=batch_size,
        query_sequence_length=(1 if mode == "decode" else resolved_sharding.local_sequence_length),
        head_dim=128,
        causal=True,
        causal_offsets=causal_offsets,
        sharding=resolved_sharding,
        reduction=ReductionSpec(),
        kv_cache=kv_cache,
    )


def _declared_cp_backend() -> AttentionBackendCapability:
    return AttentionBackendCapability(
        backend_id="test-deterministic-cp-attention",
        roles=frozenset({AttentionRole.TRAIN, AttentionRole.INFER}),
        modes=frozenset(
            {AttentionMode.PREFILL, AttentionMode.CHUNKED_PREFILL, AttentionMode.DECODE}
        ),
        dtypes=frozenset({AttentionDType.BF16}),
        tp_world_sizes=(4,),
        cp_world_sizes=(1, 2, 4),
        exports_attention_lse=True,
        deterministic_cp_merge=True,
        supports_packed_varlen=True,
        supports_kv_cache=True,
        implementation_kind="deterministic",
    )


def test_qwen3_tp4_cp4_contract_is_representable_and_serializable():
    contract = _contract()

    assert contract.sharding.local_q_heads == 8
    assert contract.sharding.local_kv_heads == 2
    assert contract.reduction.acc_dtype is AttentionDType.FP32
    assert contract.to_dict()["reduction"] == {
        "merge": "online_softmax_lse",
        "acc_dtype": "fp32",
        "order": "global_block_index",
        "downcast_at": "final_write",
        "engine": "in_op_reference",
    }
    json.dumps(contract.to_dict())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("tp_rank", 4, "tp_rank=4"),
        ("cp_rank", 4, "cp_rank=4"),
        ("global_block_indices", (), "must not be empty"),
        ("global_block_indices", (1, 0), "strictly increasing"),
    ],
)
def test_invalid_rank_and_cp_order_metadata_fail_loudly(field, value, message):
    values = {
        "tp_rank": 0,
        "tp_world_size": 4,
        "cp_rank": 0,
        "cp_world_size": 4,
        "global_q_heads": 32,
        "global_kv_heads": 8,
        "local_q_head_start": 0,
        "local_q_heads": 8,
        "local_kv_head_start": 0,
        "local_kv_heads": 2,
        "global_sequence_length": 4096,
        "local_sequence_length": 1024,
        "global_block_indices": (0,),
        "global_block_token_starts": (0,),
        "local_block_offsets": (0, 1024),
    }
    values[field] = value

    with pytest.raises(AttentionContractError, match=message):
        ShardingSpec(**values)


def test_tp_local_heads_must_preserve_global_gqa_mapping():
    with pytest.raises(AttentionContractError, match="local TP head counts"):
        replace(_sharding(), local_q_heads=7)

    with pytest.raises(AttentionContractError, match="head starts"):
        replace(_sharding(tp_rank=1), local_q_head_start=0)


def test_sequence_range_and_packed_offsets_are_validated():
    with pytest.raises(AttentionContractError, match="exceeds global_sequence_length"):
        _sharding(global_block_token_starts=(4000,))

    with pytest.raises(AttentionContractError, match="final packed_sequence_offsets"):
        _sharding(packed_sequence_offsets=(0, 512))

    sharding = _sharding(packed_sequence_offsets=(0, 256, 1024))
    assert sharding.packed_sequence_offsets == (0, 256, 1024)


def test_non_contiguous_cp_blocks_have_explicit_global_and_local_offsets():
    sharding = _sharding(
        global_block_indices=(0, 7),
        global_block_token_starts=(0, 3584),
        local_block_offsets=(0, 512, 1024),
    )

    assert sharding.global_block_indices == (0, 7)
    assert sharding.global_block_token_starts == (0, 3584)
    assert sharding.local_block_offsets == (0, 512, 1024)

    with pytest.raises(AttentionContractError, match="non-overlapping and ordered"):
        _sharding(
            global_block_indices=(0, 1),
            global_block_token_starts=(0, 256),
            local_block_offsets=(0, 512, 1024),
        )


def test_reduction_requires_fp32_accumulation():
    with pytest.raises(AttentionContractError, match="must be fp32"):
        ReductionSpec(acc_dtype="bf16")


def test_causal_attention_requires_explicit_offset():
    contract = _contract()
    with pytest.raises(AttentionContractError, match="causal_offsets are required"):
        replace(contract, causal_offsets=None)


def test_decode_requires_complete_kv_cache_identity():
    with pytest.raises(AttentionContractError, match="kv_cache metadata is required"):
        _contract(mode="decode")

    cache = KVCacheSpec(
        cache_positions=(16,),
        kv_seq_lens=(17,),
        block_table=((0, 1, -1),),
        global_token_positions=tuple(range(17)),
        page_size=16,
        prefix_cache_enabled=True,
        prefix_cache_key="prefix:sample-0",
    )
    contract = _contract(mode="decode", kv_cache=cache)
    assert contract.to_dict()["kv_cache"]["block_table"] == [[0, 1, -1]]


def test_prefix_cache_key_is_required_only_when_prefix_cache_is_enabled():
    with pytest.raises(AttentionContractError, match="prefix_cache_key is required"):
        KVCacheSpec(
            cache_positions=(16,),
            kv_seq_lens=(17,),
            block_table=((0, 1),),
            global_token_positions=tuple(range(17)),
            page_size=16,
            prefix_cache_enabled=True,
        )


def test_cache_positions_must_match_kv_sequence_count():
    with pytest.raises(AttentionContractError, match="one entry per kv_seq_lens"):
        KVCacheSpec(
            cache_positions=(1,),
            kv_seq_lens=(2, 2),
            block_table=((0,), (1,)),
            global_token_positions=(0, 1, 0, 1),
            page_size=2,
        )


@pytest.mark.parametrize("positions", [(7, 6), (7, 7)])
def test_kv_cache_positions_must_be_strictly_increasing_per_sequence(positions):
    with pytest.raises(AttentionContractError, match="strictly increasing"):
        KVCacheSpec(
            cache_positions=(7,),
            kv_seq_lens=(2,),
            block_table=((0,),),
            global_token_positions=positions,
            page_size=2,
        )


@pytest.mark.parametrize(
    ("block_table", "message"),
    [
        ((0, -1, 1), "padding must be trailing"),
        ((0, 0, -1), "duplicate active page ids"),
        ((0, -1, -1), "active page count"),
    ],
)
def test_kv_cache_block_table_page_mapping_is_validated(block_table, message):
    with pytest.raises(AttentionContractError, match=message):
        KVCacheSpec(
            cache_positions=(16,),
            kv_seq_lens=(17,),
            block_table=(block_table,),
            global_token_positions=tuple(range(17)),
            page_size=16,
        )


def test_prefix_pages_may_be_shared_across_sequences():
    cache = KVCacheSpec(
        cache_positions=(1, 1),
        kv_seq_lens=(2, 2),
        block_table=((3,), (3,)),
        global_token_positions=(0, 1, 0, 1),
        page_size=2,
        prefix_cache_enabled=True,
        prefix_cache_key="shared-prefix",
    )

    assert cache.block_table == ((3,), (3,))


def test_current_ws1_backend_rejects_strict_cp_contract_without_fallback():
    registry = KernelRegistry()

    with pytest.raises(RuntimeError) as exc_info:
        registry.get_attention_op(_contract())

    message = str(exc_info.value)
    assert "CP=4 is unsupported" in message
    assert "attention-domain LSE export is unsupported" in message
    assert "deterministic CP (out, lse) merge is unsupported" in message


def test_undeclared_backend_capability_is_never_selected():
    registry = KernelRegistry()
    platform = registry._platform()
    registry._priority_map[platform]["attention"] = [OpBackend.PYTORCH_ATTN]

    with pytest.raises(RuntimeError, match="no AttentionBackendCapability declared"):
        registry.get_attention_op(_contract())


def test_declared_compatible_backend_resolves_and_records_provenance():
    registry = KernelRegistry()
    registry._attention_capabilities[OpBackend.PYTORCH_NATIVE_ATTENTION] = _declared_cp_backend()

    result = registry.get_attention_op(_contract(), requested_backend="deterministic")

    assert result.op is not None
    assert result.capability.backend_id == "test-deterministic-cp-attention"
    assert result.provenance["requested_backend"] == "deterministic"
    assert result.provenance["actual_backend"] == "test-deterministic-cp-attention"
    assert result.provenance["fallback"] is False
    assert result.provenance["contract"]["sharding"]["cp_world_size"] == 4
    json.dumps(result.provenance)


def test_requested_stable_backend_id_is_enforced():
    registry = KernelRegistry()
    registry._attention_capabilities[OpBackend.PYTORCH_NATIVE_ATTENTION] = _declared_cp_backend()

    with pytest.raises(RuntimeError, match="does not match requested_backend=another-backend"):
        registry.get_attention_op(_contract(), requested_backend="another-backend")

    result = registry.get_attention_op(
        _contract(), requested_backend="test-deterministic-cp-attention"
    )
    assert result.provenance["actual_backend"] == "test-deterministic-cp-attention"


def test_packed_layout_requires_declared_backend_support():
    capability = replace(_declared_cp_backend(), supports_packed_varlen=False)
    contract = _contract(
        sharding=_sharding(packed_sequence_offsets=(0, 512, 1024)),
        causal_offsets=(0, 0),
        batch_size=2,
    )

    assert capability.incompatibilities(contract) == ("packed varlen layout is unsupported",)


def test_packed_sequence_count_must_match_logical_batch_size():
    sharding = _sharding(packed_sequence_offsets=(0, 512, 1024))

    with pytest.raises(AttentionContractError, match="must equal logical batch_size"):
        _contract(sharding=sharding, causal_offsets=(0, 0), batch_size=1)

    contract = _contract(sharding=sharding, causal_offsets=(0, 0), batch_size=2)
    assert contract.batch_size == 2
