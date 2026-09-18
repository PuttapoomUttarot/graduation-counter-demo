"""
YOLO26 Pose detector.

Loads the model exactly once at startup (never inside the frame loop —
see project requirement "do not load the model repeatedly").

This module intentionally exposes only stateless, per-frame detection
(`infer`). The ROI-zone + state-machine counting approach (see
`zone_state_machine.py`) needs no persistent identity between frames —
it only asks "is a head keypoint inside the zone right now?" — so
there is no multi-object tracker here and no track IDs anywhere in
this pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# COCO pose keypoint indices 0-4: Nose, Left Eye, Right Eye, Left Ear,
# Right Ear. These are the only keypoints the zone/state-machine logic
# looks at — everything below the neck (indices 5-16) is irrelevant to
# "did a head enter the ROI".
HEAD_KEYPOINT_INDICES: tuple[int, ...] = (0, 1, 2, 3, 4)
HEAD_KEYPOINT_NAMES: tuple[str, ...] = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
)


@dataclass(frozen=True)
class InferenceTiming:
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float


@dataclass(frozen=True)
class HeadKeypoint:
    """A single head-region keypoint (one of nose/eyes/ears) from one
    detected person, already filtered to confidence > threshold."""
    x: float
    y: float
    conf: float
    keypoint_index: int  # 0-4, see HEAD_KEYPOINT_INDICES
    person_index: int    # which detected person this came from, this frame


class DetectorError(Exception):
    """Raised when the YOLO26 Pose model fails to load or run."""


class PoseDetector:
    def __init__(
        self,
        model_path: str,
        device: str,
        conf: float,
        iou: float,
        imgsz: int,
    ) -> None:
        self._device = device
        self._conf = conf
        self._iou = iou
        self._imgsz = imgsz

        logger.info("Loading YOLO26 Pose model: %s", model_path)
        try:
            self._model = YOLO(model_path)
        except Exception as exc:
            raise DetectorError(
                f"Failed to load YOLO26 Pose model '{model_path}'. "
                f"Confirm the filename is a valid Ultralytics pose model "
                f"(e.g. yolo26n-pose.pt) and that ultralytics can reach the "
                f"network to download it on first use, or that the file "
                f"exists locally. Original error: {exc}"
            ) from exc
        logger.info("Model loaded successfully.")

    def infer(self, frame: np.ndarray):
        """
        Run one forward pass — fresh, stateless detection only. No
        tracker, no persistence between calls, no track IDs. This is
        the only inference path this pipeline needs: the ROI-zone
        state machine only ever asks about the *current* frame's head
        keypoint positions.
        """
        results = self._model.predict(
            source=frame,
            device=self._device,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            verbose=False,
        )
        return results

    @staticmethod
    def extract_timing(results) -> InferenceTiming:
        """
        Ultralytics reports real measured stage timings on each Results
        object via `.speed` (ms). We surface those directly rather than
        timing it ourselves, so the numbers reflect the library's own
        preprocess/inference/postprocess split.
        """
        speed = results[0].speed
        return InferenceTiming(
            preprocess_ms=speed.get("preprocess", 0.0),
            inference_ms=speed.get("inference", 0.0),
            postprocess_ms=speed.get("postprocess", 0.0),
        )

    @staticmethod
    def draw_overlay(results) -> np.ndarray:
        """
        Bounding boxes, pose skeletons, track/class labels, and confidence
        are drawn using Ultralytics' own renderer, which is the verified
        API for this — we don't hand-roll skeleton drawing.
        """
        return results[0].plot()

    @staticmethod
    def person_count(results) -> int:
        """Raw detections in this frame. NOT a ceremony count — see README
        and project spec: person_detected != completed_recipient. This
        exists only for the on-screen debug overlay."""
        boxes = results[0].boxes
        return 0 if boxes is None else len(boxes)

    @staticmethod
    def extract_head_keypoints(
        results, conf_threshold: float
    ) -> list[HeadKeypoint]:
        """
        Pull every head-region keypoint (nose, left eye, right eye,
        left ear, right ear — COCO indices 0-4) from every detected
        person in this frame, keeping only keypoints whose confidence
        is strictly greater than `conf_threshold`.

        No identity/tracking is involved: this is a flat, order-
        independent list of "head points visible this frame with
        decent confidence", exactly what the ROI zone / state machine
        needs to answer "is any head point inside the zone right now".
        A person contributes 0-5 entries here depending on how many of
        their five head keypoints are visible and confident enough.
        """
        result = results[0]
        if result.keypoints is None:
            return []

        kpts_xy = result.keypoints.xy  # (num_people, num_kpts, 2)
        kpts_conf = result.keypoints.conf  # (num_people, num_kpts) or None
        if kpts_xy is None or kpts_conf is None:
            return []

        xy = kpts_xy.cpu().numpy()
        conf = kpts_conf.cpu().numpy()

        head_points: list[HeadKeypoint] = []
        num_people = xy.shape[0]
        for person_index in range(num_people):
            for kpt_index in HEAD_KEYPOINT_INDICES:
                if kpt_index >= xy.shape[1]:
                    continue  # model variant without this keypoint
                score = float(conf[person_index, kpt_index])
                if score <= conf_threshold:
                    continue
                x, y = xy[person_index, kpt_index]
                head_points.append(
                    HeadKeypoint(
                        x=float(x),
                        y=float(y),
                        conf=score,
                        keypoint_index=kpt_index,
                        person_index=person_index,
                    )
                )
        return head_points

