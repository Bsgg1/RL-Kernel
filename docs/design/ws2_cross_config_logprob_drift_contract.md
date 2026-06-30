# WS2 Cross-Config Logprob Drift Contract

Status: RFC

Tracking issues:

- [#111: WS2 cross-config alignment](https://github.com/RL-Align/RL-Kernel/issues/111)
- [#108: WS1 numerical contract](https://github.com/RL-Align/RL-Kernel/issues/108)

## Motivation

WS2 covers rollout and training paths that use different parallelism strategies, such as
rollout tensor parallelism and training FSDP. The alignment problem is not a single-op
accuracy check. It is end-to-end floating-point drift across tokenizer, masks, serving,
rollout, and training recomputation before any optimizer update.

For PPO, GRPO, and related RL post-training algorithms, the most direct pre-update signal
is selected-token log probability drift. If rollout-side `old_logprobs` and train-side
recomputed log probabilities disagree for the same checkpoint, same token ids, same masks,
and same model version, classify the failure as infrastructure, precision, mask,
tokenizer, or serving-path drift. Do not classify that failure as an algorithm or reward
problem until pre-update logprob alignment is clean.

Aggregate KL-style diagnostics are useful but not sufficient as the primary WS2 contract.
In training-inference mismatch cases, KL estimates can stay flat or fail to expose the
early failure phase, because the first-order issue is token-level rollout-vs-training
probability disagreement before the optimizer update, not necessarily a large aggregate
policy-space shift.

## Scope

This RFC defines what WS2 cross-config alignment measures and how failures are classified.
It does not add a test harness, distributed tests, runtime gates, layer-wise probes, or
distributed fixes.

Out of scope for this document:

- Implementing multi-GPU test infrastructure.
- Adding runtime pass/fail gates.
- Adding automatic layer-wise drift probes.
- Fixing TP, FSDP, SP, cache, mask, tokenizer, or serving-path bugs.
- Defining a second numerical tolerance table.

## Measurement Contract

The primary metric is selected-token logprob drift:

```text
dlogp = train_recomputed_logp - rollout_old_logp
```

Compute `dlogp` only on active response/action tokens. Prompt tokens, padding tokens, and
masked-out response positions are excluded from every aggregate metric.

The comparison must use teacher-forcing scoring on the training side. The scored sequence
is the already-sampled rollout sequence; the training path must not resample or regenerate
tokens for this contract.

The rollout and training values are comparable only when they share the same logical
inputs:

- Same checkpoint and same model version.
- Same input token ids.
- Same selected response/action token ids.
- Same attention mask and action mask.
- Same tokenizer version and tokenization policy.
- Same padding layout semantics, including left-padding or right-padding behavior.
- Same pre-update state, before any optimizer step, weight sync, or policy mutation that
  belongs to the next training step.

If the implementation has explicit position ids, cache-position metadata, sequence ids, or
packed-sequence metadata, those inputs are part of the comparison contract as well.

## Primary Failure Signal

The pass/fail decision starts from `dlogp` over active tokens. Reward, gradnorm,
weightnorm, and update norm are downstream symptoms. They are useful for debugging and
triage, but they are not the primary contract for cross-config alignment.

The zero-update expectation is:

```text
train_recomputed_logp ~= rollout_old_logp
ratio0 ~= 1
approx_kl0 ~= 0
```

The acceptable meaning of `~=` is defined by the WS1 per-dtype numerical threshold table
from [#108](https://github.com/RL-Align/RL-Kernel/issues/108). This RFC defines the
measurement surface and classification rules only.

## Diagnostics

All diagnostics are computed on active response/action tokens only.

| Metric | Definition | Purpose |
| --- | --- | --- |
| `ratio0` | `exp(dlogp)` | Zero-update policy ratio implied by train-vs-rollout logprob drift. |
| `clipfrac0` | Mean indicator that `ratio0` falls outside the configured PPO/GRPO clip range. | Detects whether drift alone would trigger clipping before any update. |
| `approx_kl0` | Masked mean of `exp(dlogp) - 1 - dlogp`. | Zero-update approximate KL implied by logprob drift. |
| `mean_abs_dlogp` | Mean of `abs(dlogp)`. | Average selected-token drift. |
| `p95_abs_dlogp` | 95th percentile of `abs(dlogp)`. | Tail drift below outliers. |
| `p99_abs_dlogp` | 99th percentile of `abs(dlogp)`. | High-tail drift. |
| `max_abs_dlogp` | Maximum of `abs(dlogp)`. | Worst selected-token mismatch. |

When the run is distributed, report optional per-rank versions of the same metrics. The
per-rank view should preserve enough metadata to identify the rollout rank, training rank,
parallelism mode, dtype, padding side, cache mode, and local active-token count for that
rank.

## Tolerance Source

Do not define numeric tolerances in this RFC. The single source of truth for acceptable
numerical drift is the per-dtype threshold table owned by
[#108](https://github.com/RL-Align/RL-Kernel/issues/108).

Later WS2 tests should map the `dlogp` diagnostics in this document to the #108 thresholds
for the dtype under test. If the #108 table changes, WS2 inherits that policy without
editing this document or maintaining a second table.

## Decision Rule

Use this order when classifying a cross-config failure:

1. If pre-update selected-token logprobs do not match under the same checkpoint, same token
   ids, same masks, and same model version, treat the failure as infrastructure,
   precision, mask, tokenizer, or serving-path drift.
2. If KL or ratio diagnostics move before gradnorm or update norm moves, treat the failure
   as likely infrastructure or logprob plumbing.
3. If gradnorm or update norm moves first and KL moves later, treat the failure as more
   likely algorithmic tuning, such as learning rate, KL beta, reward scale, or advantage
   outliers.
4. If only some ranks drift, treat the failure as distributed infrastructure until rank
   placement, sharding, collective, mask, and cache-position issues are ruled out.
5. If reward rises and then collapses while pre-update logprob alignment is clean, treat
   the failure as more likely algorithmic, reward hacking, or insufficient KL constraint.

This classification does not prove root cause by itself. It defines the first branch in
the debugging tree so WS2 bugs do not get misfiled as reward or algorithm regressions
before the zero-update logprob contract is satisfied.

## Follow-Up Test Matrix

Later PRs should implement a systematic matrix around this contract. The minimum planned
coverage is:

- Single-process reference vs TP.
- Single-process reference vs FSDP.
- TP vs FSDP.
- SP on vs SP off.
- Batch size 1 vs batch size N.
- Left padding vs right padding.
- Cache on vs cache off.
- fp32, bf16, and fp16.
- Per-rank drift reporting.

Each test should collect the primary `dlogp` vector and the diagnostics above. The test
implementation belongs in later PRs, not in this RFC.
