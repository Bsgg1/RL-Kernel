# vime + RL-Kernel linear_logp 主宣传实验

## 0. 我们要做什么

本轮只保留一个主宣传实验：

```text
baseline:  vime benchmark branch, Qwen3-30B-A3B, 8xH100 colocate, RL-Kernel off
candidate: vime benchmark branch + RL-Kernel linear_logp, Qwen3-30B-A3B, 8xH100 colocate
```

目标：证明 RL-Kernel 的 `linear_logp` 接入 vime 后，在同一套 Qwen3-30B-A3B MoE 训练配置下，不降低训练质量，并降低 selected-logprob 路径耗时或显存压力。

范围收口：

- 只测 `linear_logp`。
- 只跑 Qwen3-30B-A3B 主宣传实验。
- 不跑 Qwen3-4B smoke、R3 单独对比、GLM-4.5、GB200/H200 硬件对照。
- 不测 `logp`、`ratio_kl`、`grpo_loss`、`sampling` 的 vime 端到端收益。
- 不做训推一致性专项 benchmark。
- 不接 MoE expert/router 算子。

## 1. H100 支持结论

vime 支持 H100：当前 vime 文档已有 `Qwen3-30B-A3B with 8xH100` 和 `Qwen3-4B with 8xH100` 示例，代码里也有 H100 hardware mapping。

所以本轮主方案使用：

```text
8xH100
```

A100 不作为本轮主宣传配置。

## 2. 当前代码边界

vime candidate 只暴露一个 RL-Kernel op：

```text
RL_KERNEL_SUPPORTED_OPS = ("linear_logp",)
RL_KERNEL_INTEGRATED_OPS = ("linear_logp",)
--rl-kernel-ops linear_logp
VIME_RL_KERNEL_OPS=linear_logp
```

主实验脚本：

```text
scripts/run-qwen3-30B-A3B.sh
```

该脚本已经按 8 卡主宣传实验参数化：

```text
NUM_GPUS=8
MEGATRON_TP=4
MEGATRON_EP=8
MEGATRON_CP=1
ROLLOUT_NUM_GPUS_PER_ENGINE=8
ROLLOUT_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=8
GLOBAL_BATCH_SIZE=256
MAX_TOKENS_PER_GPU=20480
VLLM_GPU_MEMORY_UTILIZATION=0.7
```

如遇 OOM，先降低：

```text
MAX_TOKENS_PER_GPU=8192
VLLM_GPU_MEMORY_UTILIZATION=0.55
ROLLOUT_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=32
```

## 3. 上卡准备

从官方仓库开始，不依赖当前本地目录：

```bash
cd /workspace
git clone https://github.com/RL-Align/vime.git vime-main
git clone https://github.com/RL-Align/vime.git vime-benchmark
git clone https://github.com/RL-Align/vime.git vime-rlk-integration
git clone https://github.com/RL-Align/RL-Kernel.git RL-Kernel
```

RL-Kernel 必须使用含 TP 版 `linear_logp` 接口的版本。vime 这边会调用：

```text
op(hidden, weight, target_ids, bias, tp_group=..., vocab_start_index=..., global_vocab_size=...)
```

当前使用 `RL-Align/RL-Kernel#189` 提供 TP 版 `linear_logp`。上卡后在 `/workspace/RL-Kernel` 里 checkout 该 PR 后再安装：

```bash
cd /workspace/RL-Kernel
git checkout main
git pull origin main
gh pr checkout 189
```

如果目标环境没有 `gh`，使用等价的 PR ref checkout：

```bash
cd /workspace/RL-Kernel
git checkout main
git pull origin main
git fetch origin pull/189/head:pr-189
git checkout pr-189
```

`vime-main` 只作为干净参考，不直接跑实验：

```bash
cd /workspace/vime-main
git checkout main
git pull origin main
```

vime candidate 已经准备成 draft PR，baseline 仍然要保持 benchmark-only，避免把 baseline 和 candidate 混在一起：

```text
vime-rlk-benchmark-8h100
只包含 8xH100 benchmark harness，不包含 RL-Kernel 集成代码。

RL-Align/vime#1
draft PR，基于 benchmark harness，再加入 RL-Kernel linear_logp 集成代码和测试。
```

baseline 从干净 main 新建 benchmark 分支：

```bash
cd /workspace/vime-benchmark
git checkout main
git pull origin main
git checkout -b vime-rlk-benchmark-8h100
# 只应用 benchmark harness 改动，例如 scripts/run-qwen3-30B-A3B.sh 的 8xH100 参数化。
# 不加入 --enable-rl-kernel、vime/utils/rl_kernel.py、megatron_utils/rl_kernel.py 等 RL-Kernel 集成改动。
```

candidate 直接 checkout draft PR `RL-Align/vime#1`：

```bash
cd /workspace/vime-rlk-integration
git checkout main
git pull origin main
gh pr checkout 1
```

安装：

```bash
cd /workspace/RL-Kernel
PATH="/usr/local/cuda/bin:$PATH" \
CUDA_HOME=/usr/local/cuda \
KERNEL_ALIGN_FORCE_SM90=1 \
KERNEL_ALIGN_DEV_RPATH=1 \
MAX_JOBS=8 \
python -m pip install --no-build-isolation -e .

python - <<'PY'
from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE

print("ext_available", _EXT_AVAILABLE)
print("has_fused_linear_logp_sm90", hasattr(_C, "fused_linear_logp_sm90"))
assert _EXT_AVAILABLE
assert hasattr(_C, "fused_linear_logp_sm90")
PY

cd /workspace/vime-benchmark
pip install -e .

cd /workspace/vime-rlk-integration
pip install -e .
```

下载模型和数据：

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login

hf download Qwen/Qwen3-30B-A3B --local-dir /root/Qwen3-30B-A3B

hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k

hf download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /root/aime-2024
```

转换 Megatron `torch_dist` checkpoint：

```bash
cd /workspace/vime-benchmark
source scripts/models/qwen3-30B-A3B.sh

PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
  tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint /root/Qwen3-30B-A3B \
  --save /root/Qwen3-30B-A3B_torch_dist
```

## 4. 运行主宣传实验

两边使用同一套 8 卡 colocate 环境变量：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_GPUS=8
export MEGATRON_TP=4
export MEGATRON_EP=8
export MEGATRON_CP=1
export ROLLOUT_NUM_GPUS_PER_ENGINE=8
export ROLLOUT_BATCH_SIZE=32
export N_SAMPLES_PER_PROMPT=8
export GLOBAL_BATCH_SIZE=256
export MAX_TOKENS_PER_GPU=20480
export VLLM_GPU_MEMORY_UTILIZATION=0.7
```

baseline：

```bash
cd /workspace/vime-benchmark
unset VIME_RL_KERNEL VIME_RL_KERNEL_OPS VIME_RL_KERNEL_STRICT
bash scripts/run-qwen3-30B-A3B.sh
```

candidate：

```bash
cd /workspace/vime-rlk-integration
export VIME_RL_KERNEL=1
export VIME_RL_KERNEL_OPS=linear_logp
export VIME_RL_KERNEL_STRICT=1
bash scripts/run-qwen3-30B-A3B.sh
```

每组至少跑 3 次；每次丢弃前 5-10 step warmup 后统计。

## 5. 必须记录

每个 run 保存：

```text
hardware
gpu_name
num_gpus
model
dataset
vime_commit
rl_kernel_commit
candidate_enabled
enabled_rl_kernel_ops
selected_rl_kernel_backend
tp
ep
cp
rollout_batch_size
n_samples_per_prompt
global_batch_size
max_tokens_per_gpu
mean_step_time_s
p50_step_time_s
p90_step_time_s
mean_log_probs_time_s
p50_log_probs_time_s
p90_log_probs_time_s
peak_vram_gb
raw_reward_mean
train_rollout_logprob_abs_diff_mean
rl_kernel_fallback_count
```

验收线：

```text
candidate 日志出现 RL-Kernel linear_logp backend
rl_kernel_fallback_count = 0
candidate raw_reward 不低于 baseline 同量级
candidate train_rollout_logprob_abs_diff 不持续高于 baseline
candidate mean_log_probs_time_s 或 peak_vram_gb 有可解释下降
```

## 6. 最终图表

只输出主宣传图：

1. `Qwen3-30B-A3B 8xH100 raw_reward`
2. `Qwen3-30B-A3B 8xH100 train_rollout_logprob_abs_diff`
3. `Qwen3-30B-A3B 8xH100 Step Time`
4. `Qwen3-30B-A3B 8xH100 Logprob Time / Peak VRAM`

图表风格对齐 `vime_blog.md`：白底、虚线网格、baseline 蓝色、candidate 红色。

## 7. 本地验证

当前无 GPU 环境已完成：

```text
# linear_logp 主路径与公共工具
pytest tests/test_rl_kernel_args.py tests/test_rl_kernel_linear_logp_integration.py tests/test_value_temperature.py tests/test_metric_report.py -q
结果：39 passed

# legacy logp compatibility regression，不属于本轮 benchmark 范围
pytest tests/test_rl_kernel_logp_integration.py tests/test_rl_kernel_args.py tests/test_rl_kernel_linear_logp_integration.py -q
结果：24 passed

pre-commit run --files <本轮 vime 相关文件>
结果：Passed
```

上卡后必须补跑：

```text
8xH100 baseline:  /workspace/vime-benchmark, benchmark-only branch
8xH100 candidate: /workspace/vime-rlk-integration, RL-Align/vime#1
```

## 8. 宣传口径

英文：

```text
RL-Kernel integrates with vime to accelerate the Qwen3-30B-A3B GRPO selected-logprob path through linear_logp. On the same 8xH100 setup, it reduces logprob-path cost while keeping reward and train-rollout logprob alignment stable.
```

中文：

```text
RL-Kernel 接入 vime 后，通过 linear_logp 加速 Qwen3-30B-A3B GRPO selected-logprob 路径。在相同 8xH100 配置下，RL-Kernel 降低 logprob 路径开销，同时保持 reward 和 train-rollout logprob alignment 稳定。
```
