"""
ROI Zone + State Machine counter — head-keypoint based, NO object
tracking, NO track IDs.

Each frame we get a flat list of head keypoints (`detector.HeadKeypoint`
— nose/left-eye/right-eye/left-ear/right-ear, COCO indices 0-4, already
filtered to confidence > threshold) from *every* person YOLO detected
this frame. There is no identity attached to any of them beyond
`person_index`, which is only stable within a single frame — it is
never used across frames. This module never looks at it.

Counting rule (see also the docstrings on `ZoneStateMachine.update`):

    1. Trigger (first head point enters):
       While the zone is NOT occupied, the moment ANY head keypoint
       from ANY person is inside the ROI, we:
         - increment `count`
         - set `zone_occupied = True`
       From that instant the counter is locked: no other keypoint or
       person can trigger another count while the zone stays occupied,
       no matter how many heads are in there.

    2. Hold & exit (all head points must leave):
       While occupied, the zone stays occupied as long as AT LEAST ONE
       head keypoint (from any person, any frame) is still inside the
       ROI. Only once EVERY head keypoint of every detected person is
       completely outside the ROI, for `empty_confirm_frames`
       consecutive frames in a row, do we clear `zone_occupied` back to
       False and re-arm the trigger.

       Requiring several consecutive clear frames (not just one) is
       what prevents flicker: a keypoint's confidence dipping below
       threshold for a single frame while a head is still physically
       in the doorway would otherwise look like "zone cleared" for
       that one frame and let the same person's next flicker
       re-trigger a double count. A brief clear streak has to hold
       for the whole confirm window before we trust it.

This is intentionally simpler than the old track-ID `LineCounter`: it
never needs identity, occlusion handling, or ID re-use reasoning,
because the "locked while occupied" rule already prevents the zone
from being re-triggered by the same person (or a crowd of people)
lingering in the frame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Zone:
    """An axis-aligned ROI rectangle in pixel coordinates of the frame
    being processed (NOT the display-resized frame — inference runs on
    the full-resolution frame, so zone coordinates must match that)."""
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError(
                f"Invalid zone: x1<x2 and y1<y2 required, got "
                f"({self.x1}, {self.y1}, {self.x2}, {self.y2})"
            )

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def as_int_tuple(self) -> tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)


@dataclass
class ZoneStateMachine:
    zone: Zone
    empty_confirm_frames: int = 8

    count: int = field(default=0, init=False)
    zone_occupied: bool = field(default=False, init=False)
    _empty_streak: int = field(default=0, init=False)

    def update(self, head_keypoints: list) -> bool:
        """
        Feed every valid head keypoint detected THIS frame (already
        confidence-filtered — see `detector.extract_head_keypoints`).
        `head_keypoints` items must have `.x` and `.y` attributes in
        the same pixel space as `self.zone`.

        Returns True if this call triggered a NEW count, False
        otherwise. Also updates `self.count` and `self.zone_occupied`
        in place.
        """
        any_inside = any(
            self.zone.contains(kp.x, kp.y) for kp in head_keypoints
        )

        if not self.zone_occupied:
            # --- Trigger logic: first head point enters ---
            if any_inside:
                self.count += 1
                self.zone_occupied = True
                self._empty_streak = 0
                logger.info(
                    "Zone entry detected. Total count = %d", self.count
                )
                return True
            return False

        # --- Hold & exit logic: zone currently occupied ---
        if any_inside:
            # At least one head point still inside -> stay locked,
            # reset the consecutive-empty-frame streak.
            self._empty_streak = 0
        else:
            self._empty_streak += 1
            if self._empty_streak >= self.empty_confirm_frames:
                self.zone_occupied = False
                self._empty_streak = 0
                logger.debug(
                    "Zone confirmed clear after %d consecutive empty "
                    "frames. Re-armed for next entry.",
                    self.empty_confirm_frames,
                )
        return False

    def reset(self) -> None:
        self.count = 0
        self.zone_occupied = False
        self._empty_streak = 0
