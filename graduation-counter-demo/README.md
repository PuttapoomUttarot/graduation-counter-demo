# Graduation Ceremony Counter — Demo

**Current status: Phase 0 (Environment Check) + Phase 1 (Video File + YOLO26 Pose + GPU) only.**

This is a demo-only build. It does **not** yet track people, define ceremony
zones, run the completion state machine, count anyone, talk to Node.js, or
touch a real camera. Those are later phases — see "Roadmap" below. Say
**"Continue Phase 2"** when you're ready and Phase 2 (tracking) will be added
on top of this without changing what's already here.

## 1. Project overview

Long-term goal: count graduation recipients who complete the full ceremony
journey (entry → ceremony area → presenter interaction → exit), using
YOLO26 Pose + tracking + a per-person state machine — not raw detection
counts. Full spec is in the original prompt; this README only documents
what Phase 0-1 actually builds.

## 2. Architecture (today vs. eventual)

```
Video File ──► YOLO26 Pose ──► [draw + display]        (Phase 0-1, this README)

Video File ──► YOLO26 Pose ──► Tracking ──► Zones ──► State Machine
            ──► Counter ──► Node.js ──► WebSocket ──► Dashboard  (later phases)
```

`video_source.py` defines a `VideoSource` interface so that `Webcam`/`RTSP`
implementations can be added later without touching `detector.py` or the
processing loop.

## 3. Requirements

- Python 3.10+
- An NVIDIA GPU (this was written for an RTX 4050 Laptop GPU) — optional but
  recommended; the app runs on CPU too, just slower, and tells you clearly
  which one it's using.
- A ceremony (or any people-containing) video file.

## 4. NVIDIA GPU setup

Install a CUDA-enabled PyTorch build **before** installing the rest of the
requirements, or `pip` may pull a CPU-only torch wheel:

1. Go to https://pytorch.org/get-started/locally/
2. Pick your OS (Windows), package (pip), and your installed CUDA version.
3. Run the command it gives you, e.g.:
   ```
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```
4. Then install the rest of this project's requirements (step 6 below).

Verify afterward with:
```
python gpu_check.py
```
This must print `CUDA available: YES` and your GPU name before you rely on
GPU acceleration. If it doesn't, fix that first — the app will still run on
CPU, but will warn you loudly rather than silently using it.

## 5. Python setup

```
cd graduation-counter-demo
python -m venv venv
venv\Scripts\activate        # Windows
```

## 6. Installing dependencies

```
cd python
pip install -r requirements.txt
```

(`pynvml` is optional — it only adds a real GPU-utilization % to the
on-screen overlay. If you skip it, the app runs fine and just omits that
line instead of guessing a number.)

## 7. Downloading / locating the YOLO26 Pose model

You don't need to do anything manually: Ultralytics downloads
`yolo26n-pose.pt` (or whichever model you configure) automatically from the
official release the first time it's used. If you already have a weights
file, just point `YOLO_MODEL` at it.

## 8. Adding the demo video

Put your ceremony video at:
```
data/videos/graduation_demo.mp4
```
or any path you prefer — just update `VIDEO_SOURCE` in `.env` (step 9).

## 9. Configuring `.env`

```
cd graduation-counter-demo
copy .env.example .env
```
Then edit `.env` — at minimum check `VIDEO_SOURCE` points at your real video
file. Nothing in the Python code hard-codes this path; it's read entirely
from `.env`.

## 10. Running the demo

```
cd python
python app.py
```

A window opens showing the video with bounding boxes, pose skeletons, track
IDs, confidence scores, a **vertical counting line**, and a live
**COMPLETED: N** counter — plus a stats overlay:

- **Video FPS (source)** — the video file's own native frame rate.
- **Processing FPS** — actually measured pipeline throughput (read + infer +
  draw + display), not assumed.
- **Inference time** — real per-frame YOLO26 timing from Ultralytics'
  `.speed` reporting.
- **GPU** name, memory used, and utilization % (utilization only if
  `pynvml` is installed).

**How counting works (ID-free, "beam break" style):** every frame, the
system checks whether ANY detected body currently touches the vertical
line. The counter increments by 1 the moment the line goes from "clear" to
"occupied" — no track ID involved at all. It only resets to "ready to
count again" once the line is fully clear (nobody touching it), so one
person lingering or walking slowly across the line is still just 1 count.
Adjust `LINE_X_RATIO` in `.env` (0.0 = left edge, 1.0 = right edge) to match
where people actually pass in your footage.

**Known trade-off:** if two people touch the line at overlapping times —
one stepping off just as another steps on, so the line never goes fully
clear in between — this counts that overlap as a single crossing. That's
the nature of not tracking identity; it trades away the ID-switch
double-counting risk of the earlier tracking-based approach in exchange for
this different (rarer, in a single-file line) failure mode. If your footage
has people crossing shoulder-to-shoulder often, this is worth knowing.

Press **`q`** in the video window to stop early, or let it run to the end of
the file — either way the app exits cleanly and releases the video handle.

## 11. Expected output

Console (Phase 0 check, printed once at startup):
```
============================================================
ENVIRONMENT CHECK
============================================================
PyTorch version:      2.x.x
Ultralytics version:  8.x.x
CUDA available:       YES
CUDA version:         12.1
GPU name:             NVIDIA GeForce RTX 4050 Laptop GPU
GPU count:            1
============================================================
```
Then a video window (Phase 1) with live overlays as described above, and
log lines for model loading, video FPS, and any warnings.

## 12. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `Configuration error: Missing required environment variable: VIDEO_SOURCE` | You haven't created `.env` yet — copy `.env.example` to `.env`. |
| `Video source not found: ...` | The path in `.env` doesn't exist. Check it's relative to where you run `python app.py` from (the `python/` folder). |
| `CUDA available: NO` when you expect a GPU | Your torch build is CPU-only — reinstall using the CUDA wheel (step 4). Also confirm `nvidia-smi` works at all in a normal terminal. |
| Failed to load YOLO26 Pose model | Check your internet connection (first run downloads weights) or that `YOLO_MODEL` in `.env` is a valid filename like `yolo26n-pose.pt`. |
| Video window doesn't appear / crashes on open | Make sure you're running with a display attached (not a headless server/SSH session without X forwarding). |
| Low FPS on GPU | Try a smaller `YOLO_IMGSZ` (e.g. 416 or 512) in `.env`, per the performance-tuning plan for later phases. |

## 13. Project structure (current)

```
graduation-counter-demo/
├── python/
│   ├── app.py            # Phase 0-1 entry point
│   ├── config.py         # .env loading, no hard-coded values
│   ├── gpu_check.py       # Phase 0 environment verification
│   ├── detector.py        # YOLO26 Pose, loaded once
│   ├── video_source.py    # VideoSource abstraction + FileVideoSource
│   ├── metrics.py         # Real, measured FPS/timing/GPU stats
│   └── requirements.txt
├── data/
│   ├── videos/            # put graduation_demo.mp4 here
│   └── events/             # reserved for later phases
├── .env.example
└── README.md
```

Not yet created (arrive in later phases): `tracker.py`, `pose.py` (reference
point helper), `zones.py`, `state_machine.py`, `interaction.py`,
`counter.py`, `event_client.py`, the `server/` (Node.js) and `frontend/`
folders, `config/zones.json`, and `docker-compose.yml`.

## 14. Roadmap (not implemented yet)

Phase 2 tracking → Phase 3 zones → Phase 4 state machine → Phase 5 counting
→ Phase 6 interaction heuristic → Phase 7 Node.js → Phase 8 dashboard →
Phase 9 persistence → Phase 10 webcam/RTSP support. Each phase builds on top
of this one; say "Continue Phase 2" to start the next.

## 15. Future RTSP/camera architecture

`VideoSource` in `video_source.py` is the only seam that will change for
real hardware: a future `WebcamVideoSource` or `RTSPVideoSource` implements
the same `read()` / `release()` / `source_fps` interface. `detector.py`,
and every phase built after it, will not need to change.
