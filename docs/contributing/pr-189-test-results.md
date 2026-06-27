# PR 189 Test Results

Validated on 2026-06-27 UTC from branch `pr-189` at commit `ae7402e`
against `origin/main` at `ea196da`.

## Environment

| Item | Value |
| --- | --- |
| Python | 3.11.10 |
| PyTorch | 2.4.1+cu124 |
| CUDA runtime | 12.4 |
| CUDA toolkit | `nvcc` 12.4.131 from `/usr/local/cuda/bin` |
| Driver | 580.126.09 |
| GPU | 2 x NVIDIA H100 80GB HBM3, compute capability 9.0 |
| DeepSpeed | 0.19.2, installed with `DS_BUILD_OPS=0` |

The package was rebuilt in editable mode with the SM90 extension enabled:

```bash
PATH="/usr/local/cuda/bin:$PATH" \
CUDA_HOME=/usr/local/cuda \
KERNEL_ALIGN_FORCE_SM90=1 \
KERNEL_ALIGN_DEV_RPATH=1 \
MAX_JOBS=8 \
python -m pip install --no-build-isolation -e .
```

Post-build import checks reported `ext_available=True` and
`has_fused_linear_logp_sm90=True`.

## Passing Checks

| Area | Command | Result |
| --- | --- | --- |
| PR-focused linear logp unit tests | `python -m pytest tests/test_linear_logp.py -q -rs` | 27 passed in 8.86s |
| DeepSpeed worker contract tests | `python -m pytest tests/test_deepspeed_training_worker.py -q -rs` | 22 passed in 2.46s |
| Fused logp registry/fallback and CUDA loss-step regression | `python -m pytest tests/test_op_accuracy.py tests/test_rl_kernel_loss_step.py -q -rs --tb=short` | 21 passed in 2.18s |
| Full H100/SM90 pytest suite | `python -m pytest tests rl_engine/tests -q -rs --tb=short` | 283 passed, 82 skipped in 16.01s |
| CI dispatch baseline | `python -m pytest rl_engine/tests/test_dispatch.py -v` | 5 passed in 1.62s |
| CI attention baseline | `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_attention_correctness.py -q -rs` | 45 passed, 82 skipped in 2.32s |
| Type check | `python -m mypy --ignore-missing-imports rl_engine/` | Success: no issues found in 50 source files |
| Documentation build | `mkdocs build --strict -f mkdocs.yaml` | Passed in 0.77s |
| Pre-commit | `pre-commit run --all-files` | Passed |

The attention skips were environment-gated:

- 40 CUDA FlashAttentionOp cases skipped because neither external `flash_attn`
  nor `_C.flash_attn_forward` was available.
- 42 ROCm cases skipped because this was an NVIDIA CUDA environment.

The documentation build emitted non-fatal warnings that this new page has no git
revision history yet, plus the upstream Material for MkDocs notice about MkDocs
2.0.

## Production-Like CUDA Checks

The tensor-parallel validation script was run with two H100 GPUs through NCCL.

Direct SM90 backend:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PATH="/usr/local/cuda/bin:$PATH" \
CUDA_HOME=/usr/local/cuda \
OMP_NUM_THREADS=1 \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/test_linear_logp_tp.py \
  --op-source sm90 \
  --dtype bf16 \
  --reference-mode fp32 \
  --tokens 512 \
  --hidden-size 1024 \
  --vocab-size 8192 \
  --uneven-shards \
  --run-stress \
  --stress-tokens 8192 \
  --stress-hidden-size 4096 \
  --stress-vocab-size 65536
```

Result: PASS.

| Metric | Result |
| --- | --- |
| Correctness output max abs | 3.051758e-04 |
| Correctness hidden grad max abs | 6.250000e-02 |
| Correctness weight grad max abs | 6.250000e-02 |
| Correctness bias grad max abs | 1.562500e-02 |
| Stress finite check | PASS |
| Stress max rank elapsed | 184.532 ms |
| Stress max rank peak memory | 1.696 GiB |

The runtime logs confirmed `Using fused_linear_logp_sm90 tensor-parallel
local-shard path`.

Registry-dispatched backend:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PATH="/usr/local/cuda/bin:$PATH" \
CUDA_HOME=/usr/local/cuda \
OMP_NUM_THREADS=1 \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/test_linear_logp_tp.py \
  --op-source registry \
  --dtype bf16 \
  --reference-mode fp32 \
  --tokens 256 \
  --hidden-size 512 \
  --vocab-size 4096 \
  --uneven-shards
```

Result: PASS. The registry detected SM90 and selected the fused TMA linear logp
kernel. The tensor-parallel local-shard path was used.

| Metric | Result |
| --- | --- |
| Correctness output max abs | 1.068115e-04 |
| Correctness hidden grad max abs | 3.125000e-02 |
| Correctness weight grad max abs | 3.125000e-02 |
| Correctness bias grad max abs | 1.953125e-03 |

## Real DeepSpeed Smoke

After installing DeepSpeed 0.19.2 with `DS_BUILD_OPS=0`, a single-rank CUDA
smoke test was run with an initialized NCCL process group and launcher-style
environment variables.

Configuration:

- `device="cuda"`
- `dtype=torch.bfloat16`
- `zero_stage=0`
- `vocab_size=2048`
- `hidden_dim=256`
- `num_prompts=2`
- `samples_per_prompt=2`
- `completion_len=16`

Result: PASS.

| Metric | Result |
| --- | --- |
| Training backend | `deepspeed` |
| Training device | `cuda:0` |
| DeepSpeed engine | `DeepSpeedEngine` |
| Zero stage | 0 |
| Current logp backend | `FusedLinearLogpSM90Op` |
| Active tokens | 54 |
| Loss finite | True |
| Published weight version | 11 |

Two setup attempts before the passing run confirmed the required launch
preconditions:

- `dist_init_required=False` without a process group fails because DeepSpeed
  requires an initialized distributed backend.
- A manually initialized process group still requires `LOCAL_RANK`, `RANK`, and
  `WORLD_SIZE` to be set, matching normal launcher behavior.

## Resolved Full-Suite Regression

The first full local pytest run exposed three failures in `tests/test_op_accuracy.py`
when the legacy `logp` registry path selected `FusedLogpSM90Op` on H100:

- `test_accuracy` used fp16 logits, but `FusedLogpSM90Op` asserted bfloat16
  logits.
- `test_fused_logp_out_reuses_output_storage` expected `.out(...)`, which
  `FusedLogpSM90Op` did not expose.
- `test_fused_logp_fp32_output` expected `.apply_fp32(...)`, which
  `FusedLogpSM90Op` did not expose.

The first direct bf16 fast-path regression test then exposed that the legacy
`fused_logp_sm90` TMA path is not production-safe in this environment:
`cuTensorMapEncodeTiled` failed with the original tile shape, and a conservative
tile change caused the CUDA kernel to hang. The production fix is to leave the
legacy SM90 `logp` TMA path disabled by default and require
`RL_KERNEL_ENABLE_EXPERIMENTAL_SM90_LOGP=1` before it can be selected. The
`FusedLogpSM90Op` wrapper now delegates to `FusedLogpGenericOp` by default, and
the CUDA loss-step smoke accepts the SM90 wrapper so fallback behavior is
actually exercised instead of skipped.

This does not affect the PR's fused linear logp SM90 production path, which is
covered by the two-GPU tensor-parallel checks above.

The full H100/SM90 suite was then rerun:

```bash
PATH="/usr/local/cuda/bin:$PATH" \
CUDA_HOME=/usr/local/cuda \
KERNEL_ALIGN_FORCE_SM90=1 \
python -m pytest tests rl_engine/tests -q -rs --tb=short
```

Result: 283 passed, 82 skipped in 16.01s.
