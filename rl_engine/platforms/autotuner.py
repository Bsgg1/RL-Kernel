# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from rl_engine.utils.logger import logger

AutotuneConfig = Mapping[str, Any]
BenchmarkFn = Callable[[AutotuneConfig], Any]

_CACHE_VERSION = 1
_CACHE_LOCK = threading.Lock()
_DISABLED_VALUES = {"0", "false", "no", "off"}


def default_cache_path() -> Path:
    """Return the persistent autotune cache path."""
    path = os.getenv("RL_KERNEL_AUTOTUNE_CACHE")
    if path:
        return Path(path).expanduser()
    return Path.cwd() / ".kernel_align_cache"


def normalize_cache_value(value: Any) -> Any:
    """Convert objects used in autotune keys/configs into JSON-stable values."""
    if isinstance(value, Mapping):
        return {str(key): normalize_cache_value(val) for key, val in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [normalize_cache_value(item) for item in value]
    if isinstance(value, torch.Size):
        return list(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def tensor_signature(tensor: torch.Tensor) -> dict[str, Any]:
    """Shape key fragment for tensor-dependent autotuning."""
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "stride": list(tensor.stride()),
        "device_type": tensor.device.type,
    }


def _resolve_device(device: torch.device | int | str | None) -> torch.device:
    if isinstance(device, int):
        return torch.device("cuda", device)
    return torch.device(device) if device is not None else torch.device("cuda")


def get_gpu_identity(device: torch.device | int | str | None = None) -> str:
    """Best-effort stable GPU identity, including PCI bus id when PyTorch exposes it."""
    if not torch.cuda.is_available():
        return "cpu"

    resolved = _resolve_device(device)
    if resolved.type != "cuda":
        return resolved.type

    index = resolved.index if resolved.index is not None else torch.cuda.current_device()
    backend = "rocm" if getattr(torch.version, "hip", None) else "cuda"

    try:
        props = torch.cuda.get_device_properties(index)
    except Exception as exc:
        logger.warning(f"Failed to inspect GPU properties for autotune key: {exc}")
        return f"{backend}|index={index}|pci=unknown"

    pci_bus_id = getattr(props, "pci_bus_id", None)
    get_pci_bus_id = getattr(torch.cuda, "get_device_pci_bus_id", None)
    if pci_bus_id is None and callable(get_pci_bus_id):
        try:
            pci_bus_id = get_pci_bus_id(index)
        except Exception:
            pci_bus_id = None

    return "|".join(
        [
            backend,
            f"index={index}",
            f"name={props.name}",
            f"cc={getattr(props, 'major', 'unknown')}.{getattr(props, 'minor', 'unknown')}",
            f"pci={pci_bus_id or 'unknown'}",
        ]
    )


class PersistentKernelAutotuner:
    """
    Profiles candidate kernel launch configurations once and caches the fastest result.

    Operators provide a list of meta-parameter dictionaries and a benchmark callback that
    launches the kernel with one of those dictionaries. Cache entries are keyed by operator
    name, GPU identity, tensor shape metadata, and the candidate configuration space.
    """

    def __init__(
        self,
        cache_path: str | os.PathLike[str] | None = None,
        *,
        enabled: bool | None = None,
        warmup: int = 1,
        repeat: int = 5,
    ) -> None:
        self.cache_path = Path(cache_path).expanduser() if cache_path else default_cache_path()
        self.enabled = self._env_enabled() if enabled is None else enabled
        self.warmup = max(0, warmup)
        self.repeat = max(1, repeat)

    def tune(
        self,
        op_name: str,
        shape_key: Mapping[str, Any],
        candidate_configs: Sequence[AutotuneConfig],
        benchmark: BenchmarkFn,
        *,
        device: torch.device | int | str | None = None,
        device_identity: str | None = None,
        extra_key: Mapping[str, Any] | None = None,
        fallback_config: AutotuneConfig | None = None,
    ) -> dict[str, Any]:
        configs = [dict(config) for config in candidate_configs]
        if not configs and fallback_config is None:
            raise ValueError("PersistentKernelAutotuner requires at least one candidate config")

        fallback = dict(fallback_config or configs[0])
        if not self.enabled:
            return fallback

        identity = device_identity or get_gpu_identity(device)
        cache_key = self._make_cache_key(op_name, identity, shape_key, configs, extra_key)
        cache = self._read_cache()
        cached = cache["entries"].get(cache_key)
        if isinstance(cached, Mapping) and isinstance(cached.get("best_config"), Mapping):
            return deepcopy(dict(cached["best_config"]))

        best_config: dict[str, Any] | None = None
        best_latency_ms = float("inf")
        failures: list[str] = []
        profiled_configs: list[dict[str, Any]] = []

        for config in configs:
            try:
                latency_ms = self._profile_config(benchmark, config, device)
            except Exception as exc:
                failures.append(f"{config}: {exc}")
                profiled_configs.append(
                    {
                        "config": normalize_cache_value(config),
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                logger.warning(f"Autotune candidate failed for {op_name}: {config} ({exc})")
                continue

            profiled_configs.append(
                {
                    "config": normalize_cache_value(config),
                    "status": "pass",
                    "latency_ms": latency_ms,
                }
            )
            if latency_ms < best_latency_ms:
                best_latency_ms = latency_ms
                best_config = config

        if best_config is None:
            logger.warning(
                f"All autotune candidates failed for {op_name}; using fallback config {fallback}."
            )
            if failures:
                logger.warning(f"Autotune failures for {op_name}: {'; '.join(failures[:3])}")
            return fallback

        cache["entries"][cache_key] = {
            "op_name": op_name,
            "device_identity": identity,
            "shape_key": normalize_cache_value(shape_key),
            "extra_key": normalize_cache_value(extra_key or {}),
            "best_config": normalize_cache_value(best_config),
            "latency_ms": best_latency_ms,
            "profiled_configs": profiled_configs,
            "updated_at": time.time(),
        }
        self._write_cache(cache)
        logger.info_once(
            f"Autotuned {op_name}: {best_config} ({best_latency_ms:.3f} ms), "
            f"cache={self.cache_path}"
        )
        return deepcopy(best_config)

    def clear(self) -> None:
        with _CACHE_LOCK:
            try:
                self.cache_path.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _env_enabled() -> bool:
        value = os.getenv("RL_KERNEL_AUTOTUNE", "1").strip().lower()
        return value not in _DISABLED_VALUES

    def _profile_config(
        self,
        benchmark: BenchmarkFn,
        config: AutotuneConfig,
        device: torch.device | int | str | None,
    ) -> float:
        resolved = _resolve_device(device)
        for _ in range(self.warmup):
            benchmark(config)
        self._synchronize(resolved)

        if resolved.type == "cuda" and torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(self.repeat):
                benchmark(config)
            end.record()
            end.synchronize()
            return start.elapsed_time(end) / self.repeat

        start_time = time.perf_counter()
        for _ in range(self.repeat):
            benchmark(config)
        self._synchronize(resolved)
        return (time.perf_counter() - start_time) * 1000.0 / self.repeat

    @staticmethod
    def _synchronize(device: torch.device) -> None:
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    @staticmethod
    def _make_cache_key(
        op_name: str,
        device_identity: str,
        shape_key: Mapping[str, Any],
        candidate_configs: Sequence[AutotuneConfig],
        extra_key: Mapping[str, Any] | None,
    ) -> str:
        payload = {
            "version": _CACHE_VERSION,
            "op_name": op_name,
            "device_identity": device_identity,
            "shape_key": normalize_cache_value(shape_key),
            "candidate_configs": normalize_cache_value(candidate_configs),
            "extra_key": normalize_cache_value(extra_key or {}),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _read_cache(self) -> dict[str, Any]:
        with _CACHE_LOCK:
            try:
                with self.cache_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except FileNotFoundError:
                return self._empty_cache()
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning(f"Ignoring invalid autotune cache {self.cache_path}: {exc}")
                return self._empty_cache()

        if not isinstance(payload, Mapping):
            return self._empty_cache()
        if payload.get("version") != _CACHE_VERSION:
            return self._empty_cache()
        entries = payload.get("entries")
        if not isinstance(entries, Mapping):
            return self._empty_cache()
        return {"version": _CACHE_VERSION, "entries": dict(entries)}

    def _write_cache(self, payload: Mapping[str, Any]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with _CACHE_LOCK:
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{self.cache_path.name}.", suffix=".tmp", dir=self.cache_path.parent
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, sort_keys=True, indent=2)
                    handle.write("\n")
                Path(tmp_name).replace(self.cache_path)
            except Exception:
                try:
                    Path(tmp_name).unlink()
                except FileNotFoundError:
                    pass
                raise

    @staticmethod
    def _empty_cache() -> dict[str, Any]:
        return {"version": _CACHE_VERSION, "entries": {}}
