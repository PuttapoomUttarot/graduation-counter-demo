"""
Video source abstraction.

`VideoSource` is the seam that lets us swap:
    Video File  ->  Webcam  ->  RTSP camera  ->  Edge device
without touching detector.py, the tracker, zones, the state machine,
or app.py. Only a new subclass needs to be added later; nothing here
should assume "file" beyond `FileVideoSource` itself.

Phase 0-1 only implements `FileVideoSource`, per the current phase
scope. `FileVideoSource` also paces reads to the video's own frame
rate so the demo behaves like a live camera feed instead of racing
through the file as fast as possible (see README "Real-time behavior").
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np


class VideoSourceError(Exception):
    """Raised when a video source cannot be opened or read."""


class VideoSource(ABC):
    """Common interface every video source (file, webcam, RTSP...) implements."""

    @abstractmethod
    def read(self) -> np.ndarray | None:
        """Return the next frame, or None when the source is exhausted/closed."""

    @abstractmethod
    def release(self) -> None:
        """Release any underlying resources (file handles, camera handles)."""

    @property
    @abstractmethod
    def source_fps(self) -> float:
        """The native frame rate reported by the source."""


class FileVideoSource(VideoSource):
    """Reads a local video file and paces playback to its native FPS."""

    def __init__(self, path: str) -> None:
        video_path = Path(path)
        if not video_path.exists():
            raise VideoSourceError(f"Video source not found: {path}")

        self._cap = cv2.VideoCapture(str(video_path))
        if not self._cap.isOpened():
            raise VideoSourceError(
                f"Could not open video file (unsupported or corrupt?): {path}"
            )

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        # Some files/codecs report 0 or nonsense FPS; fall back to a sane default
        # rather than dividing by zero or pacing incorrectly.
        self._source_fps = fps if fps and fps > 1.0 else 25.0
        self._frame_interval = 1.0 / self._source_fps
        self._start_time = time.perf_counter()
        self._frame_index = 0

    @property
    def source_fps(self) -> float:
        return self._source_fps

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        if not ok:
            return None  # end of video, or read failure

        # Pace playback to the source's real frame rate. If processing has
        # fallen behind (inference slower than realtime), skip the sleep
        # instead of queuing up a backlog — we always grab the *next*
        # sequential frame, never an artificially buffered one.
        target_time = self._start_time + self._frame_index * self._frame_interval
        now = time.perf_counter()
        if now < target_time:
            time.sleep(target_time - now)

        self._frame_index += 1
        return frame

    def release(self) -> None:
        self._cap.release()
