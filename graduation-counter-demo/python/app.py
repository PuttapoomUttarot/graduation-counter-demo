"""
Graduation ceremony ROI-zone counter — video demo.

Phase 0 — Environment Check: verify PyTorch/CUDA/Ultralytics before
doing anything else, and report the real state clearly.

Phase 1/2 — Video File + YOLO26 Pose + GPU + ROI Zone State-Machine
Count: load the model once, read a video file paced to its native FPS,
run YOLO26 Pose per-frame detection (stateless — NO object tracking,
NO track IDs), and count a ceremony recipient using a head-keypoint
ROI-zone state machine:

    - Every frame, extract the 5 head keypoints (COCO indices 0-4:
      nose, left eye, right eye, left ear, right ear) of every
      detected person, keeping only keypoints above a confidence
      threshold.
    - The FIRST time any head keypoint from any person enters the ROI
      zone while the zone is unoccupied, count += 1 and the zone locks
      (`zone_occupied = True`) — no other keypoint/person can trigger
      another count while locked.
    - The zone only unlocks once ALL head keypoints of all persons
      have been completely outside the ROI for
      `EMPTY_CONFIRM_FRAMES` consecutive frames in a row (debounces
      flicker).

See zone_state_machine.py for the exact rule and detector.py's
`extract_head_keypoints` for the keypoint extraction.

Explicitly NOT implemented yet (later phases): presenter exclusion,
Node.js, WebSocket, persistence, and real cameras (webcam/RTSP).
"""

from __future__ import annotations

import logging
import sys

import cv2
import numpy as np

from config import AppConfig, ConfigError, load_config
from detector import DetectorError, PoseDetector
from gpu_check import check_gpu, log_gpu_status
from metrics import Metrics, try_get_gpu_memory_mb, try_get_gpu_utilization_pct
from video_source import FileVideoSource, VideoSourceError
from zone_state_machine import Zone, ZoneStateMachine

logger = logging.getLogger(__name__)

WINDOW_NAME = "Graduation Ceremony Counter — Demo"


def setup_logging(level_name: str) -> None:
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def resolve_device(config: AppConfig, cuda_available: bool) -> str:
    """
    Never silently swap CPU for GPU or vice versa without saying so.
    """
    if config.wants_gpu and not cuda_available:
        logger.warning(
            "DEVICE=%s requested but CUDA is not available. Falling back "
            "to CPU. Inference will be significantly slower.",
            config.device,
        )
        return "cpu"
    return config.device


def resize_for_display(
    frame: np.ndarray, max_width: int, max_height: int
) -> np.ndarray:
    """
    Scale the frame down to fit within (max_width, max_height) while
    preserving aspect ratio, so the window fits on screen. This only
    affects what's displayed — inference already ran on the full-res
    frame before this is called. Never upscales a smaller frame.
    """
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale >= 1.0:
        return frame
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def draw_zone_and_count(
    frame: np.ndarray, zone: Zone, zone_occupied: bool, count: int
) -> None:
    """
    Draw the ROI rectangle (red while occupied/locked, green while
    clear and armed) plus the running count, top-left of the zone.
    """
    x1, y1, x2, y2 = zone.as_int_tuple()
    color = (0, 0, 255) if zone_occupied else (0, 200, 0)  # BGR
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    status = "OCCUPIED" if zone_occupied else "CLEAR"
    cv2.putText(frame, status, (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, status, (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    text = f"COMPLETED: {count}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
    h, w = frame.shape[:2]
    x = max(12, min((x1 + x2) // 2 - tw // 2, w - tw - 12))
    cv2.putText(frame, text, (x, th + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(frame, text, (x, th + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                (0, 255, 255), 2, cv2.LINE_AA)


def draw_head_keypoints(frame: np.ndarray, head_keypoints: list, zone: Zone) -> None:
    """
    Small marker on every head keypoint that passed the confidence
    filter this frame — yellow if outside the ROI, magenta if inside
    it, so it's visually obvious what the state machine is reacting
    to (on top of Ultralytics' own skeleton overlay, which is drawn
    separately).
    """
    for kp in head_keypoints:
        inside = zone.contains(kp.x, kp.y)
        color = (255, 0, 255) if inside else (0, 255, 255)  # BGR
        cv2.circle(frame, (int(kp.x), int(kp.y)), 5, color, -1, cv2.LINE_AA)


def draw_stats_overlay(
    frame,
    *,
    video_fps: float,
    processing_fps: float,
    inference_ms: float,
    detection_count: int,
    gpu_name: str | None,
    gpu_mem_mb: float | None,
    gpu_util_pct: float | None,
) -> None:
    lines = [
        f"Video FPS (source):  {video_fps:.1f}",
        f"Processing FPS:      {processing_fps:.1f}",
        f"Inference time:      {inference_ms:.1f} ms",
        f"Detections:          {detection_count}",
        f"GPU: {gpu_name or 'CPU (no CUDA)'}",
    ]
    if gpu_mem_mb is not None:
        lines.append(f"GPU memory used:     {gpu_mem_mb:.0f} MB")
    if gpu_util_pct is not None:
        lines.append(f"GPU utilization:     {gpu_util_pct:.0f}%")

    y = 24
    for line in lines:
        cv2.putText(
            frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (60, 220, 60), 1, cv2.LINE_AA,
        )
        y += 22


def main() -> int:
    try:
        config = load_config()
    except ConfigError as exc:
        # Logging isn't configured yet if this fails, so print directly.
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    setup_logging(config.log_level)

    logger.info("=== Phase 0: Environment Check ===")
    try:
        gpu_status = check_gpu()
    except ImportError as exc:
        logger.error("%s", exc)
        return 1
    log_gpu_status(gpu_status)

    device = resolve_device(config, gpu_status.cuda_available)
    gpu_name = gpu_status.gpu_name if device != "cpu" else None

    logger.info("=== Phase 1: Video File + YOLO26 Pose ===")
    try:
        video_source = FileVideoSource(config.video_source)
    except VideoSourceError as exc:
        logger.error("%s", exc)
        return 1

    try:
        detector = PoseDetector(
            model_path=config.yolo_model,
            device=device,
            conf=config.yolo_conf,
            iou=config.yolo_iou,
            imgsz=config.yolo_imgsz,
        )
    except DetectorError as exc:
        logger.error("%s", exc)
        video_source.release()
        return 1

    try:
        zone = Zone(
            x1=config.zone_x1, y1=config.zone_y1,
            x2=config.zone_x2, y2=config.zone_y2,
        )
    except ValueError as exc:
        logger.error("Invalid ROI zone configuration: %s", exc)
        video_source.release()
        return 1

    metrics = Metrics()
    zone_machine = ZoneStateMachine(
        zone=zone, empty_confirm_frames=config.empty_confirm_frames
    )
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    window_sized = False

    logger.info(
        "Source video FPS: %.2f — press 'q' in the video window to stop.",
        video_source.source_fps,
    )

    try:
        while True:
            metrics.mark_loop_start()
            frame = video_source.read()
            if frame is None:
                logger.info(
                    "Video completed. Final count = %d", zone_machine.count
                )
                break

            results = detector.infer(frame)
            timing = PoseDetector.extract_timing(results)
            metrics.record_inference_timing(
                timing.preprocess_ms, timing.inference_ms, timing.postprocess_ms
            )

            head_keypoints = PoseDetector.extract_head_keypoints(
                results, conf_threshold=config.head_keypoint_conf
            )
            zone_machine.update(head_keypoints)

            annotated = PoseDetector.draw_overlay(results)
            draw_head_keypoints(annotated, head_keypoints, zone)
            draw_zone_and_count(
                annotated, zone, zone_machine.zone_occupied, zone_machine.count
            )
            stats = metrics.snapshot()
            draw_stats_overlay(
                annotated,
                video_fps=video_source.source_fps,
                processing_fps=stats.processing_fps,
                inference_ms=stats.inference_ms,
                detection_count=PoseDetector.person_count(results),
                gpu_name=gpu_name,
                gpu_mem_mb=try_get_gpu_memory_mb(),
                gpu_util_pct=try_get_gpu_utilization_pct(),
            )

            display_frame = resize_for_display(
                annotated, config.display_max_width, config.display_max_height
            )
            if not window_sized:
                h, w = display_frame.shape[:2]
                cv2.resizeWindow(WINDOW_NAME, w, h)
                window_sized = True

            cv2.imshow(WINDOW_NAME, display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Quit requested by user.")
                break
    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C).")
    finally:
        video_source.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
