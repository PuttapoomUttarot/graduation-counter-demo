"""
Performance metrics — all values here are measured from the running
system. Nothing in this module invents or estimates a number; if a
metric isn't available (e.g. GPU utilization without pynvml), it is
reported as unavailable rather than faked.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class FrameStats:
    processing_fps: float
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    total_pipeline_ms: float


class Metrics:
    """Rolling-window tracker for actual pipeline throughput and timing."""

    def __init__(self, window_size: int = 30) -> None:
        self._window_size = window_size
        self._loop_timestamps: deque[float] = deque(maxlen=window_size)
        self._preprocess_ms: deque[float] = deque(maxlen=window_size)
        self._inference_ms: deque[float] = deque(maxlen=window_size)
        self._postprocess_ms: deque[float] = deque(maxlen=window_size)
        self._last_loop_start: float | None = None

    def mark_loop_start(self) -> None:
        self._last_loop_start = time.perf_counter()
        self._loop_timestamps.append(self._last_loop_start)

    def record_inference_timing(
        self, preprocess_ms: float, inference_ms: float, postprocess_ms: float
    ) -> None:
        self._preprocess_ms.append(preprocess_ms)
        self._inference_ms.append(inference_ms)
        self._postprocess_ms.append(postprocess_ms)

    def snapshot(self) -> FrameStats:
        processing_fps = self._compute_fps()
        total_pipeline_ms = (
            (1000.0 / processing_fps) if processing_fps > 0 else 0.0
        )
        return FrameStats(
            processing_fps=processing_fps,
            preprocess_ms=_avg(self._preprocess_ms),
            inference_ms=_avg(self._inference_ms),
            postprocess_ms=_avg(self._postprocess_ms),
            total_pipeline_ms=total_pipeline_ms,
        )

    def _compute_fps(self) -> float:
        if len(self._loop_timestamps) < 2:
            return 0.0
        elapsed = self._loop_timestamps[-1] - self._loop_timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._loop_timestamps) - 1) / elapsed


def _avg(values: deque[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def try_get_gpu_memory_mb() -> float | None:
    """Real GPU memory currently allocated by this process, or None."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.memory_allocated(0) / (1024 * 1024)
    except ImportError:
        return None


def try_get_gpu_utilization_pct() -> float | None:
    """
    Real GPU utilization percent via pynvml, or None if pynvml isn't
    installed. We never fabricate this value.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        pynvml.nvmlShutdown()
        return float(util.gpu)
    except Exception:
        return None
