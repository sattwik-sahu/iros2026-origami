"""Benchmark utilities: latency and GFLOPs for wandb.

Latency is measured as wall-clock time for a single forward (and for
sample_actions) on CUDA, with proper warmup and synchronisation. GFLOPs are
measured via PyTorch's built-in ``FlopCounterMode`` (no extra dependency).
Both are logged to wandb as numeric scalars so they appear in the dashboard.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from origami_iros.models._typing import Observation


def measure_latency(
    model: torch.nn.Module,
    obs: Observation,
    n_warmup: int = 10,
    n_iter: int = 50,
) -> dict[str, float]:
    """Measure forward and sample latency.

    Args:
        model: The policy (on CUDA, eval mode).
        obs: A batch of observations (on CUDA).
        n_warmup: Warmup iterations (not timed).
        n_iter: Timed iterations.

    Returns:
        Dict with ``latency_forward_ms``, ``latency_sample_ms``,
        ``throughput_forward_qps``, ``throughput_sample_qps``.
    """
    model.eval()
    # warmup
    with torch.inference_mode():
        for _ in range(n_warmup):
            _ = model(obs)
            _ = model.sample_actions(obs)
    torch.cuda.synchronize()

    # forward
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(n_iter):
            _ = model(obs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    forward_ms = elapsed / n_iter * 1000
    qps = obs.batch_size[0] / (elapsed / n_iter) if elapsed > 0 else 0.0

    # sample_actions (ODE solver, may be slower)
    obs_small = obs  # already on cuda
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(n_iter):
            _ = model.sample_actions(obs_small)
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - start
    sample_ms = elapsed_s / n_iter * 1000
    qps_s = obs.batch_size[0] / (elapsed_s / n_iter) if elapsed_s > 0 else 0.0

    return {
        "latency_forward_ms": forward_ms,
        "throughput_forward_qps": qps,
        "latency_sample_ms": sample_ms,
        "throughput_sample_qps": qps_s,
    }


def measure_gflops(model: torch.nn.Module, obs: Observation) -> dict[str, Any]:
    """Measure GFLOPs via ``torch.utils.flop_counter``.

    Args:
        model: The policy.
        obs: A batch of observations.

    Returns:
        Dict with ``gflops_forward`` and per-module breakdown if available.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return {"gflops_forward": 0.0, "note": "FlopCounterMode not available"}

    # Use built-in FlopCounterMode (no extra deps). Example from torch 2.11 docs:
    # with FlopCounterMode(display=False) as counter: model(obs)
    # counter.get_total_flops()
    try:
        with FlopCounterMode(display=False) as counter:
            with torch.inference_mode():
                _ = model(obs)
        total = counter.get_total_flops()
        table = counter.get_table()
    except Exception as e:  # noqa: BLE001
        # fallback estimate: 2 * params
        params = sum(p.numel() for p in model.parameters())
        total = params * 2
        table = str(e)
        return {"gflops_forward": float(total / 1e9), "flops_forward": int(total), "table": table, "note": "fallback"}

    gflops = total / 1e9
    return {"gflops_forward": float(gflops), "flops_forward": int(total), "table": table}
