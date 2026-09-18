"""
Configuration loading for the graduation ceremony counter demo.

All environment-specific values (video path, model file, thresholds,
device) come from `.env` — never hard-coded in business logic, per
project requirements. Phase 0-1 only needs a subset of the eventual
config surface; later phases will extend `AppConfig`, not replace it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class AppConfig:
    video_source: str
    yolo_model: str
    yolo_conf: float
    yolo_iou: float
    yolo_imgsz: int
    device: str  # "cpu" or a CUDA device index as a string, e.g. "0"
    log_level: str
    display_max_width: int
    display_max_height: int

    # ROI zone (pixel coordinates in the FULL-RESOLUTION frame that
    # inference runs on — not the resized display frame).
    zone_x1: int
    zone_y1: int
    zone_x2: int
    zone_y2: int

    # Head-keypoint confidence threshold (COCO indices 0-4: nose, eyes,
    # ears). A keypoint only counts as "present" if its confidence is
    # strictly greater than this.
    head_keypoint_conf: float

    # Consecutive fully-clear frames required before the zone is
    # considered vacated (debounces flicker — see zone_state_machine.py).
    empty_confirm_frames: int

    @property
    def wants_gpu(self) -> bool:
        return self.device.strip().lower() != "cpu"


def _get_env(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and (value is None or value.strip() == ""):
        raise ConfigError(
            f"Missing required environment variable: {name}. "
            f"Set it in your .env file (see .env.example)."
        )
    return value if value is not None else ""


def _get_float(name: str, default: str) -> float:
    raw = _get_env(name, default)
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name}={raw!r} is not a valid number.") from exc


def _get_int(name: str, default: str) -> int:
    raw = _get_env(name, default)
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name}={raw!r} is not a valid integer.") from exc


def load_config(env_file: str | Path | None = None) -> AppConfig:
    """
    Load configuration from a `.env` file (defaults to `.env` in the
    current working directory) plus process environment variables.
    Process environment always takes precedence over the file.
    """
    if env_file is not None:
        load_dotenv(dotenv_path=env_file, override=False)
    else:
        load_dotenv(override=False)

    return AppConfig(
        video_source=_get_env("VIDEO_SOURCE", required=True),
        yolo_model=_get_env("YOLO_MODEL", default="yolo26n-pose.pt"),
        yolo_conf=_get_float("YOLO_CONF", default="0.45"),
        yolo_iou=_get_float("YOLO_IOU", default="0.50"),
        yolo_imgsz=_get_int("YOLO_IMGSZ", default="640"),
        device=_get_env("DEVICE", default="0"),
        log_level=_get_env("LOG_LEVEL", default="INFO").upper(),
        display_max_width=_get_int("DISPLAY_MAX_WIDTH", default="1280"),
        display_max_height=_get_int("DISPLAY_MAX_HEIGHT", default="720"),
        zone_x1=_get_int("ZONE_X1", default="480"),
        zone_y1=_get_int("ZONE_Y1", default="120"),
        zone_x2=_get_int("ZONE_X2", default="800"),
        zone_y2=_get_int("ZONE_Y2", default="480"),
        head_keypoint_conf=_get_float("HEAD_KEYPOINT_CONF", default="0.5"),
        empty_confirm_frames=_get_int("EMPTY_CONFIRM_FRAMES", default="8"),
    )
