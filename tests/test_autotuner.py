# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import json

import pytest

from rl_engine.platforms.autotuner import PersistentKernelAutotuner


def test_persistent_autotuner_selects_fastest_config_and_reuses_cache(tmp_path, monkeypatch):
    cache_path = tmp_path / ".kernel_align_cache"
    configs = [{"BLOCK_M": 32}, {"BLOCK_M": 64}]
    benchmark_calls = []
    profile_calls = []

    tuner = PersistentKernelAutotuner(cache_path, enabled=True, warmup=0, repeat=1)

    def benchmark(config):
        benchmark_calls.append(config["BLOCK_M"])

    def fake_profile(run, config, device):
        profile_calls.append(config["BLOCK_M"])
        run(config)
        return {32: 2.0, 64: 1.0}[config["BLOCK_M"]]

    monkeypatch.setattr(tuner, "_profile_config", fake_profile)

    best = tuner.tune(
        "test_op",
        {"shape": [1, 2, 3]},
        configs,
        benchmark,
        device="cpu",
        device_identity="gpu-a",
    )

    assert best == {"BLOCK_M": 64}
    assert profile_calls == [32, 64]
    assert benchmark_calls == [32, 64]

    cached_tuner = PersistentKernelAutotuner(cache_path, enabled=True, warmup=0, repeat=1)

    def fail_profile(run, config, device):
        pytest.fail("cached autotune result should avoid profiling")

    monkeypatch.setattr(cached_tuner, "_profile_config", fail_profile)

    cached = cached_tuner.tune(
        "test_op",
        {"shape": [1, 2, 3]},
        configs,
        benchmark,
        device="cpu",
        device_identity="gpu-a",
    )

    assert cached == {"BLOCK_M": 64}
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert len(payload["entries"]) == 1
    entry = next(iter(payload["entries"].values()))
    assert entry["profiled_configs"] == [
        {"config": {"BLOCK_M": 32}, "status": "pass", "latency_ms": 2.0},
        {"config": {"BLOCK_M": 64}, "status": "pass", "latency_ms": 1.0},
    ]


def test_persistent_autotuner_keys_by_shape_and_device(tmp_path, monkeypatch):
    cache_path = tmp_path / ".kernel_align_cache"
    configs = [{"name": "only"}]
    profile_calls = []
    tuner = PersistentKernelAutotuner(cache_path, enabled=True, warmup=0, repeat=1)

    def fake_profile(run, config, device):
        profile_calls.append(config["name"])
        return 1.0

    monkeypatch.setattr(tuner, "_profile_config", fake_profile)

    for shape, device_identity in (
        ({"shape": [1, 16]}, "gpu-a"),
        ({"shape": [1, 32]}, "gpu-a"),
        ({"shape": [1, 16]}, "gpu-b"),
    ):
        assert tuner.tune(
            "test_op",
            shape,
            configs,
            lambda config: None,
            device="cpu",
            device_identity=device_identity,
        ) == {"name": "only"}

    assert profile_calls == ["only", "only", "only"]
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert len(payload["entries"]) == 3


def test_persistent_autotuner_ignores_corrupt_cache(tmp_path, monkeypatch):
    cache_path = tmp_path / ".kernel_align_cache"
    cache_path.write_text("not json", encoding="utf-8")
    tuner = PersistentKernelAutotuner(cache_path, enabled=True, warmup=0, repeat=1)
    profile_calls = 0

    def fake_profile(run, config, device):
        nonlocal profile_calls
        profile_calls += 1
        return 1.0

    monkeypatch.setattr(tuner, "_profile_config", fake_profile)

    best = tuner.tune(
        "test_op",
        {"shape": [1]},
        [{"BLOCK_M": 64}],
        lambda config: None,
        device="cpu",
        device_identity="gpu-a",
    )

    assert best == {"BLOCK_M": 64}
    assert profile_calls == 1
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["version"] == 1


def test_persistent_autotuner_skips_failed_candidates(tmp_path, monkeypatch):
    cache_path = tmp_path / ".kernel_align_cache"
    tuner = PersistentKernelAutotuner(cache_path, enabled=True)

    def fake_profile(run, config, device):
        if config["name"] == "bad":
            raise RuntimeError("compile failed")
        return 0.5

    monkeypatch.setattr(tuner, "_profile_config", fake_profile)

    best = tuner.tune(
        "test_op",
        {"shape": [1]},
        [{"name": "bad"}, {"name": "good"}],
        lambda config: None,
        device="cpu",
        device_identity="gpu-a",
    )

    assert best == {"name": "good"}
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    entry = next(iter(payload["entries"].values()))
    assert entry["profiled_configs"][0]["status"] == "failed"
    assert entry["profiled_configs"][0]["config"] == {"name": "bad"}
    assert entry["profiled_configs"][1] == {
        "config": {"name": "good"},
        "status": "pass",
        "latency_ms": 0.5,
    }


def test_persistent_autotuner_can_be_disabled(tmp_path, monkeypatch):
    tuner = PersistentKernelAutotuner(tmp_path / ".kernel_align_cache", enabled=False)

    def fail_profile(run, config, device):
        pytest.fail("disabled autotuner should not profile candidates")

    monkeypatch.setattr(tuner, "_profile_config", fail_profile)

    best = tuner.tune(
        "test_op",
        {"shape": [1]},
        [{"name": "candidate"}],
        lambda config: None,
        fallback_config={"name": "fallback"},
    )

    assert best == {"name": "fallback"}
