"""
Phase 0 — Environment Check.

Verifies that PyTorch, CUDA, and Ultralytics are actually available and
working before any video processing starts. Never assumes CUDA works;
always reports the real, measured state.

Run standalone:
    python gpu_check.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GpuStatus:
    torch_version: str
    ultralytics_version: str
    cuda_available: bool
    cuda_version: str | None
    gpu_name: str | None
    gpu_count: int


def check_gpu() -> GpuStatus:
    """
    Import torch and ultralytics, query real CUDA state, and return it.
    Raises ImportError with a clear message if either package is missing
    — this must never be swallowed into a silent CPU fallback.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is not installed. Install it before continuing — "
            "see README.md 'NVIDIA GPU setup' for the correct CUDA build."
        ) from exc

    try:
        import ultralytics
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is not installed. Run: pip install -r requirements.txt"
        ) from exc

    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda if cuda_available else None
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    gpu_count = torch.cuda.device_count() if cuda_available else 0

    return GpuStatus(
        torch_version=torch.__version__,
        ultralytics_version=ultralytics.__version__,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
    )


def log_gpu_status(status: GpuStatus) -> None:
    logger.info("PyTorch version:      %s", status.torch_version)
    logger.info("Ultralytics version:  %s", status.ultralytics_version)
    if status.cuda_available:
        logger.info("CUDA available:       YES")
        logger.info("CUDA version:         %s", status.cuda_version)
        logger.info("GPU name:             %s", status.gpu_name)
        logger.info("GPU count:            %d", status.gpu_count)
    else:
        logger.warning("CUDA available:       NO — inference will run on CPU.")
        logger.warning(
            "If you expect a GPU (e.g. RTX 4050) here, your PyTorch build is "
            "likely CPU-only. Reinstall torch using the CUDA wheel from "
            "https://pytorch.org/get-started/locally/ before continuing."
        )


def print_gpu_status(status: GpuStatus) -> None:
    print("=" * 60)
    print("ENVIRONMENT CHECK")
    print("=" * 60)
    print(f"PyTorch version:      {status.torch_version}")
    print(f"Ultralytics version:  {status.ultralytics_version}")
    if status.cuda_available:
        print("CUDA available:       YES")
        print(f"CUDA version:         {status.cuda_version}")
        print(f"GPU name:             {status.gpu_name}")
        print(f"GPU count:            {status.gpu_count}")
    else:
        print("CUDA available:       NO (will use CPU)")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    result = check_gpu()
    print_gpu_status(result)
