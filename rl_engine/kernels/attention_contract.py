# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Typed WS2 contract for context-parallel standard softmax attention.

The objects in this module describe a distributed attention invocation.  They
do not shard tensors, launch collectives, or implement the ``(out, lse)``
merge.  Keeping description and materialization separate lets dispatch reject
an incompatible backend before any numerically different path is launched.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, TypeVar

_EnumT = TypeVar("_EnumT", bound=Enum)


class AttentionContractError(ValueError):
    """Raised when attention metadata does not describe a valid invocation."""


class AttentionRole(str, Enum):
    TRAIN = "train"
    INFER = "infer"


class AttentionMode(str, Enum):
    PREFILL = "prefill"
    CHUNKED_PREFILL = "chunked_prefill"
    DECODE = "decode"


class AttentionDType(str, Enum):
    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"


class AttentionMerge(str, Enum):
    ONLINE_SOFTMAX_LSE = "online_softmax_lse"


class ReductionOrder(str, Enum):
    GLOBAL_BLOCK_INDEX = "global_block_index"


class DowncastPoint(str, Enum):
    FINAL_WRITE = "final_write"


class ReductionEngine(str, Enum):
    IN_OP_REFERENCE = "in_op_reference"


def _enum_value(enum_type: type[_EnumT], value: Any, field: str) -> _EnumT:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise AttentionContractError(f"{field} must be one of: {allowed}; got {value!r}") from exc


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AttentionContractError(f"{field} must be a positive integer; got {value!r}")
    return value


def _non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AttentionContractError(f"{field} must be a non-negative integer; got {value!r}")
    return value


def _integer_tuple(values: Iterable[int], field: str) -> tuple[int, ...]:
    try:
        result = tuple(values)
    except TypeError as exc:
        raise AttentionContractError(f"{field} must be an iterable of integers") from exc
    for index, value in enumerate(result):
        if isinstance(value, bool) or not isinstance(value, int):
            raise AttentionContractError(f"{field}[{index}] must be an integer; got {value!r}")
    return result


@dataclass(frozen=True)
class ShardingSpec:
    """Logical TP/CP ownership for one attention invocation.

    TP head shards are currently required to be equal and contiguous.  CP
    sequence ownership may be uneven, but every local block must carry a stable
    logical global index so a later implementation can merge by logical order
    instead of collective arrival order.
    """

    tp_rank: int
    tp_world_size: int
    cp_rank: int
    cp_world_size: int
    global_q_heads: int
    global_kv_heads: int
    local_q_head_start: int
    local_q_heads: int
    local_kv_head_start: int
    local_kv_heads: int
    global_sequence_length: int
    local_sequence_length: int
    global_block_indices: tuple[int, ...]
    global_block_token_starts: tuple[int, ...]
    local_block_offsets: tuple[int, ...]
    packed_sequence_offsets: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        tp_world_size = _positive_int(self.tp_world_size, "tp_world_size")
        cp_world_size = _positive_int(self.cp_world_size, "cp_world_size")
        tp_rank = _non_negative_int(self.tp_rank, "tp_rank")
        cp_rank = _non_negative_int(self.cp_rank, "cp_rank")
        if tp_rank >= tp_world_size:
            raise AttentionContractError(
                f"tp_rank={tp_rank} must be smaller than tp_world_size={tp_world_size}"
            )
        if cp_rank >= cp_world_size:
            raise AttentionContractError(
                f"cp_rank={cp_rank} must be smaller than cp_world_size={cp_world_size}"
            )

        global_q_heads = _positive_int(self.global_q_heads, "global_q_heads")
        global_kv_heads = _positive_int(self.global_kv_heads, "global_kv_heads")
        if global_q_heads % global_kv_heads != 0:
            raise AttentionContractError(
                f"global_q_heads={global_q_heads} must be divisible by "
                f"global_kv_heads={global_kv_heads} for GQA"
            )
        if global_q_heads % tp_world_size != 0 or global_kv_heads % tp_world_size != 0:
            raise AttentionContractError(
                "global Q/KV heads must be evenly divisible by tp_world_size; "
                f"got Hq={global_q_heads}, Hkv={global_kv_heads}, TP={tp_world_size}"
            )

        local_q_heads = _positive_int(self.local_q_heads, "local_q_heads")
        local_kv_heads = _positive_int(self.local_kv_heads, "local_kv_heads")
        expected_q_heads = global_q_heads // tp_world_size
        expected_kv_heads = global_kv_heads // tp_world_size
        if local_q_heads != expected_q_heads or local_kv_heads != expected_kv_heads:
            raise AttentionContractError(
                "local TP head counts do not preserve the global Q/KV mapping; "
                f"expected ({expected_q_heads}, {expected_kv_heads}), got "
                f"({local_q_heads}, {local_kv_heads})"
            )

        local_q_head_start = _non_negative_int(self.local_q_head_start, "local_q_head_start")
        local_kv_head_start = _non_negative_int(self.local_kv_head_start, "local_kv_head_start")
        expected_q_start = tp_rank * expected_q_heads
        expected_kv_start = tp_rank * expected_kv_heads
        if local_q_head_start != expected_q_start or local_kv_head_start != expected_kv_start:
            raise AttentionContractError(
                "local TP head starts do not match contiguous rank ownership; "
                f"expected ({expected_q_start}, {expected_kv_start}), got "
                f"({local_q_head_start}, {local_kv_head_start})"
            )

        global_sequence_length = _positive_int(
            self.global_sequence_length, "global_sequence_length"
        )
        local_sequence_length = _positive_int(self.local_sequence_length, "local_sequence_length")

        block_indices = _integer_tuple(self.global_block_indices, "global_block_indices")
        if not block_indices:
            raise AttentionContractError("global_block_indices must not be empty")
        if any(index < 0 for index in block_indices):
            raise AttentionContractError("global_block_indices must be non-negative")
        if any(
            left >= right for left, right in zip(block_indices, block_indices[1:], strict=False)
        ):
            raise AttentionContractError(
                "global_block_indices must be unique and strictly increasing"
            )
        object.__setattr__(self, "global_block_indices", block_indices)

        block_token_starts = _integer_tuple(
            self.global_block_token_starts, "global_block_token_starts"
        )
        local_block_offsets = _integer_tuple(self.local_block_offsets, "local_block_offsets")
        if len(block_token_starts) != len(block_indices):
            raise AttentionContractError(
                "global_block_token_starts must contain one entry per global_block_indices entry"
            )
        if any(start < 0 for start in block_token_starts):
            raise AttentionContractError("global_block_token_starts must be non-negative")
        if len(local_block_offsets) != len(block_indices) + 1:
            raise AttentionContractError(
                "local_block_offsets must contain one boundary more than global_block_indices"
            )
        if local_block_offsets[0] != 0 or local_block_offsets[-1] != local_sequence_length:
            raise AttentionContractError(
                "local_block_offsets must start at 0 and end at local_sequence_length"
            )
        if any(
            left >= right
            for left, right in zip(local_block_offsets, local_block_offsets[1:], strict=False)
        ):
            raise AttentionContractError("local_block_offsets must be strictly increasing")

        previous_global_end = 0
        for index, (global_start, local_start, local_end) in enumerate(
            zip(
                block_token_starts,
                local_block_offsets[:-1],
                local_block_offsets[1:],
                strict=True,
            )
        ):
            global_end = global_start + (local_end - local_start)
            if global_end > global_sequence_length:
                raise AttentionContractError(
                    f"global block {block_indices[index]} exceeds global_sequence_length"
                )
            if index > 0 and global_start < previous_global_end:
                raise AttentionContractError(
                    "global block token ranges must be non-overlapping and ordered"
                )
            previous_global_end = global_end
        object.__setattr__(self, "global_block_token_starts", block_token_starts)
        object.__setattr__(self, "local_block_offsets", local_block_offsets)

        if self.packed_sequence_offsets is not None:
            offsets = _integer_tuple(self.packed_sequence_offsets, "packed_sequence_offsets")
            if len(offsets) < 2 or offsets[0] != 0:
                raise AttentionContractError(
                    "packed_sequence_offsets must start at 0 and contain an end offset"
                )
            if any(left >= right for left, right in zip(offsets, offsets[1:], strict=False)):
                raise AttentionContractError("packed_sequence_offsets must be strictly increasing")
            if offsets[-1] != local_sequence_length:
                raise AttentionContractError(
                    "the final packed_sequence_offsets value must equal local_sequence_length; "
                    f"got {offsets[-1]} and {local_sequence_length}"
                )
            object.__setattr__(self, "packed_sequence_offsets", offsets)


@dataclass(frozen=True)
class ReductionSpec:
    """Deterministic CP ``(out, lse)`` merge semantics."""

    merge: AttentionMerge = AttentionMerge.ONLINE_SOFTMAX_LSE
    acc_dtype: AttentionDType = AttentionDType.FP32
    order: ReductionOrder = ReductionOrder.GLOBAL_BLOCK_INDEX
    downcast_at: DowncastPoint = DowncastPoint.FINAL_WRITE
    engine: ReductionEngine = ReductionEngine.IN_OP_REFERENCE

    def __post_init__(self) -> None:
        object.__setattr__(self, "merge", _enum_value(AttentionMerge, self.merge, "merge"))
        object.__setattr__(
            self, "acc_dtype", _enum_value(AttentionDType, self.acc_dtype, "acc_dtype")
        )
        object.__setattr__(self, "order", _enum_value(ReductionOrder, self.order, "order"))
        object.__setattr__(
            self, "downcast_at", _enum_value(DowncastPoint, self.downcast_at, "downcast_at")
        )
        object.__setattr__(self, "engine", _enum_value(ReductionEngine, self.engine, "engine"))
        if self.acc_dtype is not AttentionDType.FP32:
            raise AttentionContractError(
                f"CP attention accumulation must be fp32; got {self.acc_dtype.value}"
            )


@dataclass(frozen=True)
class KVCacheSpec:
    """Logical identity of the paged/block KV cache used for replay."""

    cache_positions: tuple[int, ...]
    kv_seq_lens: tuple[int, ...]
    block_table: tuple[tuple[int, ...], ...]
    global_token_positions: tuple[int, ...]
    page_size: int
    prefix_cache_enabled: bool = False
    prefix_cache_key: str | None = None

    def __post_init__(self) -> None:
        cache_positions = _integer_tuple(self.cache_positions, "cache_positions")
        kv_seq_lens = _integer_tuple(self.kv_seq_lens, "kv_seq_lens")
        page_size = _positive_int(self.page_size, "page_size")
        global_token_positions = _integer_tuple(
            self.global_token_positions, "global_token_positions"
        )
        if not cache_positions or any(position < 0 for position in cache_positions):
            raise AttentionContractError("cache_positions must contain non-negative positions")
        if not kv_seq_lens or any(length <= 0 for length in kv_seq_lens):
            raise AttentionContractError("kv_seq_lens must contain positive sequence lengths")
        if len(cache_positions) != len(kv_seq_lens):
            raise AttentionContractError(
                "cache_positions must contain one entry per kv_seq_lens entry"
            )
        if not global_token_positions or any(position < 0 for position in global_token_positions):
            raise AttentionContractError(
                "global_token_positions must contain non-negative positions"
            )
        if len(global_token_positions) != sum(kv_seq_lens):
            raise AttentionContractError(
                "global_token_positions must describe every logical cached token; "
                f"expected {sum(kv_seq_lens)}, got {len(global_token_positions)}"
            )
        token_offset = 0
        for sequence_index, sequence_length in enumerate(kv_seq_lens):
            sequence_positions = global_token_positions[
                token_offset : token_offset + sequence_length
            ]
            if any(
                left >= right
                for left, right in zip(sequence_positions, sequence_positions[1:], strict=False)
            ):
                raise AttentionContractError(
                    "global_token_positions must be strictly increasing within each sequence; "
                    f"sequence {sequence_index} is invalid"
                )
            token_offset += sequence_length

        try:
            block_table = tuple(tuple(row) for row in self.block_table)
        except TypeError as exc:
            raise AttentionContractError(
                "block_table must be a two-dimensional integer table"
            ) from exc
        if len(block_table) != len(kv_seq_lens) or any(not row for row in block_table):
            raise AttentionContractError(
                "block_table must contain one non-empty row per kv_seq_lens entry"
            )
        for row_index, (row, sequence_length) in enumerate(
            zip(block_table, kv_seq_lens, strict=True)
        ):
            active_blocks: list[int] = []
            saw_padding = False
            for column_index, block in enumerate(row):
                if isinstance(block, bool) or not isinstance(block, int) or block < -1:
                    raise AttentionContractError(
                        "block_table entries must be integer block ids or -1 padding; "
                        f"got block_table[{row_index}][{column_index}]={block!r}"
                    )
                if block == -1:
                    saw_padding = True
                    continue
                if saw_padding:
                    raise AttentionContractError(
                        "block_table -1 padding must be trailing; "
                        f"row {row_index} contains an active block after padding"
                    )
                active_blocks.append(block)

            expected_blocks = (sequence_length + page_size - 1) // page_size
            if len(active_blocks) != expected_blocks:
                raise AttentionContractError(
                    "block_table active page count must match kv_seq_lens and page_size; "
                    f"row {row_index} expected {expected_blocks}, got {len(active_blocks)}"
                )
            if len(set(active_blocks)) != len(active_blocks):
                raise AttentionContractError(
                    f"block_table row {row_index} contains duplicate active page ids"
                )

        if not isinstance(self.prefix_cache_enabled, bool):
            raise AttentionContractError("prefix_cache_enabled must be a bool")
        if self.prefix_cache_enabled and not self.prefix_cache_key:
            raise AttentionContractError(
                "prefix_cache_key is required when prefix_cache_enabled=True"
            )
        if not self.prefix_cache_enabled and self.prefix_cache_key is not None:
            raise AttentionContractError(
                "prefix_cache_key must be None when prefix_cache_enabled=False"
            )

        object.__setattr__(self, "cache_positions", cache_positions)
        object.__setattr__(self, "kv_seq_lens", kv_seq_lens)
        object.__setattr__(self, "block_table", block_table)
        object.__setattr__(self, "global_token_positions", global_token_positions)


@dataclass(frozen=True)
class AttentionContract:
    """Complete semantic request consumed by contract-aware dispatch."""

    role: AttentionRole
    mode: AttentionMode
    dtype: AttentionDType
    batch_size: int
    query_sequence_length: int
    head_dim: int
    causal: bool
    causal_offsets: tuple[int, ...] | None
    sharding: ShardingSpec
    reduction: ReductionSpec
    kv_cache: KVCacheSpec | None = None
    export_lse: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _enum_value(AttentionRole, self.role, "role"))
        object.__setattr__(self, "mode", _enum_value(AttentionMode, self.mode, "mode"))
        object.__setattr__(self, "dtype", _enum_value(AttentionDType, self.dtype, "dtype"))
        batch_size = _positive_int(self.batch_size, "batch_size")
        _positive_int(self.query_sequence_length, "query_sequence_length")
        _positive_int(self.head_dim, "head_dim")
        if not isinstance(self.sharding, ShardingSpec):
            raise AttentionContractError("sharding must be a ShardingSpec")
        if not isinstance(self.reduction, ReductionSpec):
            raise AttentionContractError("reduction must be a ReductionSpec")
        if self.sharding.packed_sequence_offsets is not None:
            packed_sequence_count = len(self.sharding.packed_sequence_offsets) - 1
            if packed_sequence_count != batch_size:
                raise AttentionContractError(
                    "packed sequence count must equal logical batch_size; "
                    f"got {packed_sequence_count} packed sequences and batch_size={batch_size}"
                )
        if not isinstance(self.causal, bool):
            raise AttentionContractError("causal must be a bool")
        if self.causal:
            if self.causal_offsets is None:
                raise AttentionContractError("causal_offsets are required for causal attention")
        if self.causal_offsets is not None:
            causal_offsets = _integer_tuple(self.causal_offsets, "causal_offsets")
            if not causal_offsets or any(offset < 0 for offset in causal_offsets):
                raise AttentionContractError("causal_offsets must contain non-negative offsets")
            if self.sharding.packed_sequence_offsets is not None:
                expected_causal_offsets = batch_size
                offset_owner = "packed sequence"
            else:
                expected_causal_offsets = batch_size
                offset_owner = "batch entry"
            if len(causal_offsets) != expected_causal_offsets:
                raise AttentionContractError(
                    f"causal_offsets must contain one entry per {offset_owner}"
                )
            object.__setattr__(self, "causal_offsets", causal_offsets)

        if not isinstance(self.export_lse, bool) or not self.export_lse:
            raise AttentionContractError(
                "export_lse must be True for the WS2 attention-domain LSE contract"
            )

        if self.mode is AttentionMode.DECODE and self.kv_cache is None:
            raise AttentionContractError("kv_cache metadata is required for decode attention")
        if self.kv_cache is not None and not isinstance(self.kv_cache, KVCacheSpec):
            raise AttentionContractError("kv_cache must be a KVCacheSpec when provided")
        if self.mode is AttentionMode.DECODE and self.kv_cache is not None:
            if len(self.kv_cache.kv_seq_lens) != batch_size:
                raise AttentionContractError(
                    "decode kv_seq_lens must contain one entry per batch entry"
                )
            if len(self.kv_cache.cache_positions) != batch_size:
                raise AttentionContractError(
                    "decode cache_positions must contain one entry per batch entry"
                )

    def to_dict(self) -> dict[str, Any]:
        """Return stable, JSON-compatible requested-contract provenance."""

        sharding = {
            "tp_rank": self.sharding.tp_rank,
            "tp_world_size": self.sharding.tp_world_size,
            "cp_rank": self.sharding.cp_rank,
            "cp_world_size": self.sharding.cp_world_size,
            "global_q_heads": self.sharding.global_q_heads,
            "global_kv_heads": self.sharding.global_kv_heads,
            "local_q_head_start": self.sharding.local_q_head_start,
            "local_q_heads": self.sharding.local_q_heads,
            "local_kv_head_start": self.sharding.local_kv_head_start,
            "local_kv_heads": self.sharding.local_kv_heads,
            "global_sequence_length": self.sharding.global_sequence_length,
            "local_sequence_length": self.sharding.local_sequence_length,
            "global_block_indices": list(self.sharding.global_block_indices),
            "global_block_token_starts": list(self.sharding.global_block_token_starts),
            "local_block_offsets": list(self.sharding.local_block_offsets),
            "packed_sequence_offsets": (
                list(self.sharding.packed_sequence_offsets)
                if self.sharding.packed_sequence_offsets is not None
                else None
            ),
        }
        reduction = {
            "merge": self.reduction.merge.value,
            "acc_dtype": self.reduction.acc_dtype.value,
            "order": self.reduction.order.value,
            "downcast_at": self.reduction.downcast_at.value,
            "engine": self.reduction.engine.value,
        }
        kv_cache = None
        if self.kv_cache is not None:
            kv_cache = {
                "cache_positions": list(self.kv_cache.cache_positions),
                "kv_seq_lens": list(self.kv_cache.kv_seq_lens),
                "block_table": [list(row) for row in self.kv_cache.block_table],
                "global_token_positions": list(self.kv_cache.global_token_positions),
                "page_size": self.kv_cache.page_size,
                "prefix_cache_enabled": self.kv_cache.prefix_cache_enabled,
                "prefix_cache_key": self.kv_cache.prefix_cache_key,
            }
        return {
            "semantic_operator": "standard_softmax_attention",
            "role": self.role.value,
            "mode": self.mode.value,
            "dtype": self.dtype.value,
            "batch_size": self.batch_size,
            "query_sequence_length": self.query_sequence_length,
            "head_dim": self.head_dim,
            "causal": self.causal,
            "causal_offsets": (
                list(self.causal_offsets) if self.causal_offsets is not None else None
            ),
            "export_lse": self.export_lse,
            "lse_domain": "attention",
            "sharding": sharding,
            "reduction": reduction,
            "kv_cache": kv_cache,
        }


@dataclass(frozen=True)
class AttentionBackendCapability:
    """Capabilities a concrete backend declares to contract-aware dispatch."""

    backend_id: str
    roles: frozenset[AttentionRole]
    modes: frozenset[AttentionMode]
    dtypes: frozenset[AttentionDType]
    cp_world_sizes: tuple[int, ...]
    tp_world_sizes: tuple[int, ...] | None = None
    exports_attention_lse: bool = False
    deterministic_cp_merge: bool = False
    supports_packed_varlen: bool = False
    supports_kv_cache: bool = False
    implementation_kind: str = "production"

    def __post_init__(self) -> None:
        if not isinstance(self.backend_id, str) or not self.backend_id.strip():
            raise AttentionContractError("backend_id must be a non-empty string")
        roles = frozenset(_enum_value(AttentionRole, value, "roles") for value in self.roles)
        modes = frozenset(_enum_value(AttentionMode, value, "modes") for value in self.modes)
        dtypes = frozenset(_enum_value(AttentionDType, value, "dtypes") for value in self.dtypes)
        if not roles or not modes or not dtypes:
            raise AttentionContractError("backend roles, modes, and dtypes must not be empty")
        cp_world_sizes = _integer_tuple(self.cp_world_sizes, "cp_world_sizes")
        if not cp_world_sizes or any(size <= 0 for size in cp_world_sizes):
            raise AttentionContractError("cp_world_sizes must contain positive values")
        if len(set(cp_world_sizes)) != len(cp_world_sizes):
            raise AttentionContractError("cp_world_sizes must not contain duplicates")
        tp_world_sizes = None
        if self.tp_world_sizes is not None:
            tp_world_sizes = _integer_tuple(self.tp_world_sizes, "tp_world_sizes")
            if not tp_world_sizes or any(size <= 0 for size in tp_world_sizes):
                raise AttentionContractError("tp_world_sizes must contain positive values")
            if len(set(tp_world_sizes)) != len(tp_world_sizes):
                raise AttentionContractError("tp_world_sizes must not contain duplicates")
        for field in (
            "exports_attention_lse",
            "deterministic_cp_merge",
            "supports_packed_varlen",
            "supports_kv_cache",
        ):
            if not isinstance(getattr(self, field), bool):
                raise AttentionContractError(f"{field} must be a bool")
        if self.implementation_kind not in {"production", "reference", "deterministic"}:
            raise AttentionContractError(
                "implementation_kind must be production, reference, or deterministic"
            )
        object.__setattr__(self, "roles", roles)
        object.__setattr__(self, "modes", modes)
        object.__setattr__(self, "dtypes", dtypes)
        object.__setattr__(self, "cp_world_sizes", cp_world_sizes)
        object.__setattr__(self, "tp_world_sizes", tp_world_sizes)

    def incompatibilities(self, contract: AttentionContract) -> tuple[str, ...]:
        """Explain every reason this backend cannot materialize ``contract``."""

        reasons: list[str] = []
        if contract.role not in self.roles:
            reasons.append(f"role={contract.role.value} is unsupported")
        if contract.mode not in self.modes:
            reasons.append(f"mode={contract.mode.value} is unsupported")
        if contract.dtype not in self.dtypes:
            reasons.append(f"dtype={contract.dtype.value} is unsupported")
        tp_size = contract.sharding.tp_world_size
        cp_size = contract.sharding.cp_world_size
        if self.tp_world_sizes is not None and tp_size not in self.tp_world_sizes:
            reasons.append(f"TP={tp_size} is unsupported")
        if cp_size not in self.cp_world_sizes:
            reasons.append(f"CP={cp_size} is unsupported")
        if contract.export_lse and not self.exports_attention_lse:
            reasons.append("attention-domain LSE export is unsupported")
        if cp_size > 1 and not self.deterministic_cp_merge:
            reasons.append("deterministic CP (out, lse) merge is unsupported")
        if (
            contract.sharding.packed_sequence_offsets is not None
            and not self.supports_packed_varlen
        ):
            reasons.append("packed varlen layout is unsupported")
        if contract.kv_cache is not None and not self.supports_kv_cache:
            reasons.append("KV-cache identity materialization is unsupported")
        return tuple(reasons)

    def supports(self, contract: AttentionContract) -> bool:
        return not self.incompatibilities(contract)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "roles": sorted(role.value for role in self.roles),
            "modes": sorted(mode.value for mode in self.modes),
            "dtypes": sorted(dtype.value for dtype in self.dtypes),
            "tp_world_sizes": list(self.tp_world_sizes) if self.tp_world_sizes else None,
            "cp_world_sizes": list(self.cp_world_sizes),
            "exports_attention_lse": self.exports_attention_lse,
            "deterministic_cp_merge": self.deterministic_cp_merge,
            "supports_packed_varlen": self.supports_packed_varlen,
            "supports_kv_cache": self.supports_kv_cache,
            "implementation_kind": self.implementation_kind,
        }


@dataclass(frozen=True)
class AttentionDispatchResult:
    """A concrete backend plus the actual provenance bound to the request."""

    op: Any
    capability: AttentionBackendCapability
    provenance: dict[str, Any]


__all__ = [
    "AttentionContract",
    "AttentionContractError",
    "AttentionBackendCapability",
    "AttentionDispatchResult",
    "AttentionDType",
    "AttentionMerge",
    "AttentionMode",
    "AttentionRole",
    "DowncastPoint",
    "KVCacheSpec",
    "ReductionEngine",
    "ReductionOrder",
    "ReductionSpec",
    "ShardingSpec",
]
