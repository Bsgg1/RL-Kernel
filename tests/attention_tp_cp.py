# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Torchrun correctness runner for deterministic TP/CP attention.

Example on the current CPU-only development machine:

    torchrun --standalone --nproc_per_node=4 tests/attention_tp_cp.py \
        --backend gloo --tp 2 --cp 2 --dtype bf16
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.kernels.distributed.attention_mesh import (  # noqa: E402
    create_attention_parallel_mesh,
)
from rl_engine.kernels.gtest.tolerance import load_contract  # noqa: E402
from rl_engine.kernels.attention_contract import (  # noqa: E402
    AttentionContract,
    ReductionSpec,
    ShardingSpec,
)
from rl_engine.kernels.ops.pytorch.attention.cp_attention import (  # noqa: E402
    distributed_cp_attention,
    partial_attention,
)
from rl_engine.kernels.registry import KernelRegistry  # noqa: E402


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "gloo", "nccl"), default="auto")
    parser.add_argument(
        "--op-source", choices=("direct", "registry"), default="registry"
    )
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--cp", type=int, default=2)
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=None)
    return parser.parse_args()


def _resolve_backend(requested: str) -> str:
    if requested == "auto":
        return "nccl" if torch.cuda.is_available() else "gloo"
    return requested


def _resolve_device(backend: str) -> torch.device:
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL backend requires CUDA")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def _dtype(name: str) -> torch.dtype:
    return torch.float32 if name == "fp32" else torch.bfloat16


def _max_abs(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float() - expected.float()).abs().max().item())


def _reduction_tolerance(
    dtype: torch.dtype,
    *,
    atol_override: float | None,
    rtol_override: float | None,
) -> tuple[float, float]:
    dtype_name = {
        torch.float32: "float32",
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
    }[dtype]
    tolerance = load_contract()["accuracy"]["default"]["reduction"][dtype_name]
    atol = float(tolerance["atol"] if atol_override is None else atol_override)
    rtol = float(tolerance["rtol"] if rtol_override is None else rtol_override)
    return atol, rtol


def run(args) -> None:
    backend = _resolve_backend(args.backend)
    device = _resolve_device(backend)
    dist.init_process_group(backend=backend)
    try:
        mesh = create_attention_parallel_mesh(args.tp, args.cp)
        if args.seq_len % args.cp != 0:
            raise ValueError(
                "the first distributed runner requires seq_len divisible by CP"
            )
        if args.q_heads % args.tp != 0 or args.kv_heads % args.tp != 0:
            raise ValueError("Q and KV heads must be divisible by TP")
        if args.q_heads % args.kv_heads != 0:
            raise ValueError("Q heads must be divisible by KV heads for GQA")

        dtype = _dtype(args.dtype)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        q_global = torch.randn(
            args.batch,
            args.q_heads,
            args.seq_len,
            args.head_dim,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        k_global = torch.randn(
            args.batch,
            args.kv_heads,
            args.seq_len,
            args.head_dim,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        v_global = torch.randn(
            args.batch,
            args.kv_heads,
            args.seq_len,
            args.head_dim,
            generator=generator,
            device=device,
            dtype=dtype,
        )

        q_heads_per_rank = args.q_heads // args.tp
        kv_heads_per_rank = args.kv_heads // args.tp
        tokens_per_rank = args.seq_len // args.cp
        q_head_start = mesh.tp_rank * q_heads_per_rank
        kv_head_start = mesh.tp_rank * kv_heads_per_rank
        token_start = mesh.cp_rank * tokens_per_rank
        token_end = token_start + tokens_per_rank
        q_local = q_global[
            :,
            q_head_start : q_head_start + q_heads_per_rank,
            token_start:token_end,
        ].contiguous()
        k_tp = k_global[:, kv_head_start : kv_head_start + kv_heads_per_rank]
        v_tp = v_global[:, kv_head_start : kv_head_start + kv_heads_per_rank]
        k_local = k_tp[:, :, token_start:token_end].contiguous()
        v_local = v_tp[:, :, token_start:token_end].contiguous()
        q_positions = torch.arange(token_start, token_end, device=device)
        k_positions = torch.arange(token_start, token_end, device=device)

        contract = AttentionContract(
            role="infer",
            mode="prefill",
            dtype=args.dtype,
            batch_size=args.batch,
            query_sequence_length=tokens_per_rank,
            head_dim=args.head_dim,
            causal=True,
            causal_offsets=tuple(0 for _ in range(args.batch)),
            sharding=ShardingSpec(
                tp_rank=mesh.tp_rank,
                tp_world_size=args.tp,
                cp_rank=mesh.cp_rank,
                cp_world_size=args.cp,
                global_q_heads=args.q_heads,
                global_kv_heads=args.kv_heads,
                local_q_head_start=q_head_start,
                local_q_heads=q_heads_per_rank,
                local_kv_head_start=kv_head_start,
                local_kv_heads=kv_heads_per_rank,
                global_sequence_length=args.seq_len,
                local_sequence_length=tokens_per_rank,
                global_block_indices=(mesh.cp_rank,),
                global_block_token_starts=(token_start,),
                local_block_offsets=(0, tokens_per_rank),
            ),
            reduction=ReductionSpec(),
        )
        if args.op_source == "registry":
            dispatch = KernelRegistry().get_attention_op(contract)
            actual_out, actual_lse = dispatch.op(
                q_local,
                k_local,
                v_local,
                contract=contract,
                cp_group=mesh.cp_group,
            )
            actual_backend = dispatch.capability.backend_id
        else:
            actual_out, actual_lse = distributed_cp_attention(
                q_local,
                k_local,
                v_local,
                global_block_index=mesh.cp_rank,
                q_positions=q_positions,
                k_positions=k_positions,
                cp_group=mesh.cp_group,
            )
            actual_backend = "direct-pytorch-cp-reference"
        expected = partial_attention(
            q_local,
            k_tp,
            v_tp,
            global_block_index=0,
            q_positions=q_positions,
            k_positions=torch.arange(args.seq_len, device=device),
        )
        out_atol, out_rtol = _reduction_tolerance(
            actual_out.dtype,
            atol_override=args.atol,
            rtol_override=args.rtol,
        )
        lse_atol, lse_rtol = _reduction_tolerance(
            actual_lse.dtype,
            atol_override=args.atol,
            rtol_override=args.rtol,
        )

        local_ok = torch.allclose(
            actual_out.float(),
            expected.out.float(),
            atol=out_atol,
            rtol=out_rtol,
        ) and torch.allclose(
            actual_lse.float(),
            expected.lse.float(),
            atol=lse_atol,
            rtol=lse_rtol,
        )
        local_metrics = torch.tensor(
            [_max_abs(actual_out, expected.out), _max_abs(actual_lse, expected.lse)],
            device=device,
            dtype=torch.float64,
        )
        gathered_metrics = [
            torch.empty_like(local_metrics) for _ in range(mesh.world_size)
        ]
        dist.all_gather(gathered_metrics, local_metrics)
        global_ok = torch.tensor(1 if local_ok else 0, device=device, dtype=torch.int32)
        dist.all_reduce(global_ok, op=dist.ReduceOp.MIN)

        if mesh.rank == 0:
            print(
                f"TP={args.tp} CP={args.cp} backend={backend} dtype={args.dtype} "
                f"op_source={args.op_source} actual_backend={actual_backend} "
                f"out_tol=({out_atol:.1e},{out_rtol:.1e}) "
                f"lse_tol=({lse_atol:.1e},{lse_rtol:.1e}) "
                f"shape=(B={args.batch}, S={args.seq_len}, Hq={args.q_heads}, "
                f"Hkv={args.kv_heads}, D={args.head_dim})",
                flush=True,
            )
            for rank, metrics in enumerate(gathered_metrics):
                tp_rank = rank % args.tp
                cp_rank = rank // args.tp
                print(
                    f"rank={rank} tp_rank={tp_rank} cp_rank={cp_rank} "
                    f"out_max_abs={metrics[0].item():.6e} "
                    f"lse_max_abs={metrics[1].item():.6e}",
                    flush=True,
                )
        if not bool(global_ok.item()):
            raise AssertionError(
                "distributed TP/CP attention differs from the full reference"
            )
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    run(_parse_args())
