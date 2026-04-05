"""POC v4: YOLO + ByteTrack for persistent rider tracking across frames.

Uses ultralytics built-in ByteTrack tracker to maintain rider identity
across frames. Assigns car numbers based on first-lap ordering.

Usage:
    python3 scripts/poc_rider_tracking_v4.py /tmp/trial_video_poc/iizuka_20260404_R01_trial.mp4
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def run_persistent_tracking(video_path: str, output_dir: Path) -> None:
    model = YOLO("yolov8s.pt")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total / fps

    print(f"Video: {Path(video_path).name}")
    print(f"  {w_frame}x{h_frame} @ {fps:.1f}fps, {duration:.1f}s")

    min_y = int(h_frame * 0.20)
    trial_start = int(3.0 * fps)
    trial_end = int(min(35.0, duration - 5) * fps)

    track_history: dict[int, list[tuple[float, int, int, int, int, int, int]]] = defaultdict(list)
    annotated_frames: list[tuple[float, np.ndarray]] = []
    sample_interval = int(fps * 1.5)

    process_interval = max(int(fps / 6), 1)

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx < trial_start or idx > trial_end:
            idx += 1
            continue

        if idx % process_interval != 0:
            idx += 1
            continue

        t = idx / fps

        results = model.track(
            frame, persist=True, verbose=False,
            conf=0.1, imgsz=1280, classes=[0],
            tracker="bytetrack.yaml",
        )

        frame_dets = []
        for r in results:
            if r.boxes.id is None:
                continue
            for box, track_id in zip(r.boxes, r.boxes.id):
                tid = int(track_id)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                if cy > min_y:
                    track_history[tid].append((t, cx, cy, x1, y1, x2, y2))
                    frame_dets.append((tid, cx, cy, x1, y1, x2, y2))

        if idx % sample_interval == 0:
            annotated = frame.copy()
            for tid, cx, cy, x1, y1, x2, y2 in frame_dets:
                color = _track_color(tid)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"ID:{tid}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.circle(annotated, (cx, cy), 3, color, -1)

            n = len(frame_dets)
            cv2.putText(annotated, f"t={t:.1f}s  tracked={n}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            annotated_frames.append((t, annotated))

        idx += 1

    cap.release()

    print(f"\n  Total unique track IDs: {len(track_history)}")
    print(f"  Track durations:")
    for tid in sorted(track_history.keys()):
        pts = track_history[tid]
        t_start = pts[0][0]
        t_end = pts[-1][0]
        print(f"    ID {tid:3d}: {len(pts):4d} detections, "
              f"t={t_start:.1f}s - {t_end:.1f}s ({t_end - t_start:.1f}s span)")

    car_assignment = _assign_car_numbers(track_history, trial_start / fps)
    print(f"\n  Car number assignment (by first-lap X-order):")
    for car_num, tid in sorted(car_assignment.items()):
        pts = track_history[tid]
        print(f"    Car #{car_num} -> Track ID {tid} ({len(pts)} detections)")

    _save_grid(annotated_frames, output_dir / "bytetrack_grid.jpg")
    _plot_tracks(track_history, car_assignment, output_dir / "bytetrack_tracks.png")
    _plot_track_duration(track_history, output_dir / "bytetrack_durations.png")

    out_video = output_dir / "bytetrack_annotated.mp4"
    _render_video(video_path, model, car_assignment, track_history, out_video,
                  min_y, trial_start, trial_end)


def _assign_car_numbers(
    tracks: dict[int, list[tuple[float, int, int, int, int, int, int]]],
    first_lap_start: float,
) -> dict[int, int]:
    first_lap_end = first_lap_start + 5.0

    first_lap_positions: dict[int, float] = {}
    for tid, pts in tracks.items():
        early_pts = [p for p in pts if first_lap_start <= p[0] <= first_lap_end]
        if early_pts:
            first_lap_positions[tid] = np.mean([p[1] for p in early_pts])

    sorted_tids = sorted(first_lap_positions.keys(), key=lambda t: first_lap_positions[t])

    assignment: dict[int, int] = {}
    for rank, tid in enumerate(sorted_tids[:8]):
        assignment[rank + 1] = tid

    return assignment


def _track_color(tid: int) -> tuple[int, int, int]:
    colors = [
        (255, 255, 255), (0, 165, 255), (0, 0, 255), (0, 255, 255),
        (255, 0, 255), (0, 255, 0), (255, 165, 0), (255, 0, 0),
        (128, 128, 255), (255, 128, 128), (128, 255, 128), (200, 200, 0),
    ]
    return colors[tid % len(colors)]


def _save_grid(frames: list[tuple[float, np.ndarray]], path: Path) -> None:
    if not frames:
        return
    cols = 4
    h, w = frames[0][1].shape[:2]
    rows = min((len(frames) + cols - 1) // cols, 6)
    sel = frames[:rows * cols]
    sheet = np.zeros((rows * (h + 25), cols * w, 3), dtype=np.uint8)
    for i, (t, img) in enumerate(sel):
        r, c = divmod(i, cols)
        y, x = r * (h + 25), c * w
        sheet[y:y + h, x:x + w] = img
    cv2.imwrite(str(path), sheet)
    print(f"  Saved grid: {path}")


def _plot_tracks(
    tracks: dict[int, list],
    car_assignment: dict[int, int],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    tid_to_car = {tid: car for car, tid in car_assignment.items()}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    for tid, pts in tracks.items():
        times = [p[0] for p in pts]
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        car = tid_to_car.get(tid)
        label = f"Car #{car}" if car else f"ID {tid}"
        alpha = 0.8 if car else 0.2
        lw = 1.5 if car else 0.5
        ax1.plot(times, xs, label=label, alpha=alpha, linewidth=lw)
        ax2.plot(times, ys, label=label, alpha=alpha, linewidth=lw)

    ax1.set_ylabel("X position (px)")
    ax1.set_title("Track ID X-Position over Time")
    ax1.legend(fontsize=7, ncol=4, loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("Y position (px)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Track ID Y-Position over Time")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(path), dpi=150)
    plt.close()
    print(f"  Saved tracks: {path}")


def _plot_track_duration(tracks: dict[int, list], path: Path) -> None:
    import matplotlib.pyplot as plt

    tids = sorted(tracks.keys())
    starts = []
    ends = []
    counts = []

    for tid in tids:
        pts = tracks[tid]
        starts.append(pts[0][0])
        ends.append(pts[-1][0])
        counts.append(len(pts))

    fig, ax = plt.subplots(figsize=(12, max(len(tids) * 0.3, 4)))
    for i, tid in enumerate(tids):
        ax.barh(i, ends[i] - starts[i], left=starts[i], height=0.6,
                label=f"ID {tid} ({counts[i]} pts)")
    ax.set_yticks(range(len(tids)))
    ax.set_yticklabels([f"ID {t}" for t in tids])
    ax.set_xlabel("Time (s)")
    ax.set_title("Track ID Lifetimes")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(str(path), dpi=150)
    plt.close()
    print(f"  Saved durations: {path}")


def _render_video(
    video_path: str,
    model: YOLO,
    car_assignment: dict[int, int],
    tracks: dict[int, list],
    output_path: Path,
    min_y: int,
    trial_start: int,
    trial_end: int,
) -> None:
    tid_to_car = {tid: car for car, tid in car_assignment.items()}

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    process_interval = max(int(fps / 6), 1)
    last_dets: list[tuple[int, int, int, int, int, int, int]] = []

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = idx / fps
        annotated = frame.copy()

        if trial_start <= idx <= trial_end and idx % process_interval == 0:
            results = model.track(
                frame, persist=True, verbose=False,
                conf=0.1, imgsz=1280, classes=[0],
                tracker="bytetrack.yaml",
            )
            last_dets = []
            for r in results:
                if r.boxes.id is None:
                    continue
                for box, track_id in zip(r.boxes, r.boxes.id):
                    tid = int(track_id)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    if cy > min_y:
                        last_dets.append((tid, cx, cy, x1, y1, x2, y2))

        for tid, cx, cy, x1, y1, x2, y2 in last_dets:
            car = tid_to_car.get(tid)
            if car:
                color = _car_color_bgr(car)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"#{car}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(annotated, f"t={t:.1f}s", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        out.write(annotated)
        idx += 1

    cap.release()
    out.release()
    print(f"  Saved video: {output_path}")


def _car_color_bgr(car_num: int) -> tuple[int, int, int]:
    colors = {
        1: (255, 255, 255), 2: (255, 165, 0), 3: (0, 0, 255),
        4: (0, 255, 255), 5: (255, 0, 255), 6: (0, 255, 0),
        7: (0, 165, 255), 8: (255, 0, 0),
    }
    return colors.get(car_num, (128, 128, 128))


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    output_dir = Path(video_path).parent
    run_persistent_tracking(video_path, output_dir)


if __name__ == "__main__":
    main()
