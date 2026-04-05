"""POC v5: Focused tracking on the measurement lap (last lap of trial run).

Improvements over v4:
- Camera motion compensation via phase correlation before tracking
- YOLO inference on EVERY frame during measurement lap
- Focused time window: last lap only (~20-35s typically)

Usage:
    python3 scripts/poc_measurement_lap.py /tmp/trial_video_poc/iizuka_20260404_R01_trial.mp4
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def estimate_measurement_lap_window(video_path: str) -> tuple[float, float]:
    """Heuristic: measurement lap is roughly the last ~15s before timing display."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps
    cap.release()

    end = min(duration - 8, 38.0)
    start = max(end - 15.0, 5.0)
    return start, end


def run_measurement_lap_tracking(video_path: str, output_dir: Path) -> None:
    model = YOLO("yolov8s.pt")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total / fps

    lap_start, lap_end = estimate_measurement_lap_window(video_path)
    print(f"Video: {Path(video_path).name}")
    print(f"  {w_frame}x{h_frame} @ {fps:.1f}fps, {duration:.1f}s")
    print(f"  Measurement lap window: {lap_start:.1f}s - {lap_end:.1f}s")

    min_y = int(h_frame * 0.15)
    start_frame = int(lap_start * fps)
    end_frame = int(lap_end * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, max(start_frame - 1, 0))
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

    track_data: dict[int, list[dict]] = defaultdict(list)
    annotated_frames: list[tuple[float, np.ndarray]] = []
    stabilized_frames: list[tuple[float, np.ndarray]] = []

    cumulative_dx = 0.0
    cumulative_dy = 0.0

    sample_interval = max(int(fps * 1.0), 1)
    idx = start_frame

    while idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        t = idx / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_f = gray.astype(np.float64)

        shift, _ = cv2.phaseCorrelate(prev_gray, gray_f)
        dx, dy = shift
        cumulative_dx += dx
        cumulative_dy += dy

        results = model.track(
            frame, persist=True, verbose=False,
            conf=0.10, imgsz=1280, classes=[0],
            tracker="bytetrack.yaml",
        )

        frame_dets = []
        for r in results:
            if r.boxes.id is None:
                continue
            for box, track_id in zip(r.boxes, r.boxes.id):
                tid = int(track_id)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                conf = float(box.conf[0])

                if cy > min_y:
                    stab_cx = cx + cumulative_dx
                    stab_cy = cy + cumulative_dy

                    track_data[tid].append({
                        "t": t, "frame": idx,
                        "cx": cx, "cy": cy,
                        "stab_cx": stab_cx, "stab_cy": stab_cy,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "conf": conf,
                        "area": (x2 - x1) * (y2 - y1),
                    })
                    frame_dets.append((tid, cx, cy, x1, y1, x2, y2, conf))

        if idx % sample_interval == 0:
            annotated = frame.copy()
            for tid, cx, cy, x1, y1, x2, y2, conf in frame_dets:
                color = _color(tid)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"T{tid}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(annotated, f"t={t:.1f}s n={len(frame_dets)} shift=({dx:.1f},{dy:.1f})",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            annotated_frames.append((t, annotated))

        prev_gray = gray_f
        idx += 1

    cap.release()

    print(f"\n  Frames processed: {idx - start_frame}")
    print(f"  Unique track IDs: {len(track_data)}")
    print(f"  Cumulative camera shift: ({cumulative_dx:.1f}, {cumulative_dy:.1f}) px")

    long_tracks = {tid: pts for tid, pts in track_data.items() if len(pts) >= 10}
    print(f"\n  Tracks with >= 10 detections: {len(long_tracks)}")
    for tid in sorted(long_tracks.keys()):
        pts = long_tracks[tid]
        t0, t1 = pts[0]["t"], pts[-1]["t"]
        avg_area = np.mean([p["area"] for p in pts])
        avg_conf = np.mean([p["conf"] for p in pts])
        print(f"    T{tid}: {len(pts)} pts, {t0:.1f}-{t1:.1f}s ({t1-t0:.1f}s), "
              f"avg_area={avg_area:.0f}px, avg_conf={avg_conf:.2f}")

    car_assignment = _assign_cars_by_initial_order(long_tracks, lap_start)
    print(f"\n  Car assignment:")
    for car, tid in sorted(car_assignment.items()):
        print(f"    Car #{car} -> Track {tid}")

    _save_grid(annotated_frames, output_dir / "measurement_lap_grid.jpg")
    _plot_stabilized_tracks(long_tracks, car_assignment, output_dir / "measurement_lap_tracks.png")
    _plot_durations(track_data, long_tracks, output_dir / "measurement_lap_durations.png")

    _extract_rider_features(long_tracks, car_assignment)


def _assign_cars_by_initial_order(
    tracks: dict[int, list[dict]],
    lap_start: float,
) -> dict[int, int]:
    window_end = lap_start + 3.0
    initial_x: dict[int, float] = {}
    for tid, pts in tracks.items():
        early = [p for p in pts if p["t"] <= window_end]
        if early:
            initial_x[tid] = np.mean([p["cx"] for p in early])
        else:
            initial_x[tid] = pts[0]["cx"]

    sorted_tids = sorted(initial_x.keys(), key=lambda t: initial_x[t])
    return {rank + 1: tid for rank, tid in enumerate(sorted_tids[:8])}


def _extract_rider_features(
    tracks: dict[int, list[dict]],
    car_assignment: dict[int, int],
) -> None:
    print(f"\n  === Per-rider feature extraction ===")
    tid_to_car = {tid: car for car, tid in car_assignment.items()}

    for tid in sorted(tracks.keys()):
        pts = tracks[tid]
        car = tid_to_car.get(tid, "?")

        stab_xs = np.array([p["stab_cx"] for p in pts])
        stab_ys = np.array([p["stab_cy"] for p in pts])
        times = np.array([p["t"] for p in pts])
        areas = np.array([p["area"] for p in pts])

        if len(pts) < 5:
            continue

        dt = np.diff(times)
        dt = np.where(dt == 0, 1.0 / 30.0, dt)
        vx = np.diff(stab_xs) / dt
        vy = np.diff(stab_ys) / dt
        speed = np.sqrt(vx**2 + vy**2)

        y_variance = np.std(stab_ys)
        area_trend = np.polyfit(np.arange(len(areas)), areas, 1)[0] if len(areas) > 2 else 0

        print(f"    Car #{car} (T{tid}):")
        print(f"      Speed (stab px/s): mean={np.mean(speed):.1f}, std={np.std(speed):.1f}, "
              f"max={np.max(speed):.1f}")
        print(f"      Y-variance (lateral stability): {y_variance:.2f} px")
        print(f"      Area trend (approaching/receding): {area_trend:.2f} px²/frame")


def _color(tid: int) -> tuple[int, int, int]:
    colors = [
        (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255),
        (255, 0, 255), (255, 255, 0), (0, 165, 255), (255, 165, 0),
        (128, 0, 255), (0, 128, 255), (255, 128, 0), (128, 255, 0),
    ]
    return colors[tid % len(colors)]


def _save_grid(frames: list[tuple[float, np.ndarray]], path: Path) -> None:
    if not frames:
        return
    cols = 4
    h, w = frames[0][1].shape[:2]
    rows = min((len(frames) + cols - 1) // cols, 5)
    sel = frames[:rows * cols]
    sheet = np.zeros((rows * (h + 20), cols * w, 3), dtype=np.uint8)
    for i, (t, img) in enumerate(sel):
        r, c = divmod(i, cols)
        sheet[r * (h + 20):r * (h + 20) + h, c * w:(c + 1) * w] = img
    cv2.imwrite(str(path), sheet)
    print(f"  Saved: {path}")


def _plot_stabilized_tracks(tracks: dict[int, list[dict]], car_assign: dict[int, int], path: Path) -> None:
    import matplotlib.pyplot as plt
    tid_to_car = {tid: car for car, tid in car_assign.items()}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for tid, pts in sorted(tracks.items()):
        times = [p["t"] for p in pts]
        stab_x = [p["stab_cx"] for p in pts]
        stab_y = [p["stab_cy"] for p in pts]
        car = tid_to_car.get(tid)
        label = f"Car #{car}" if car else f"T{tid}"
        ax1.plot(times, stab_x, ".-", markersize=3, label=label, alpha=0.8)
        ax2.plot(times, stab_y, ".-", markersize=3, label=label, alpha=0.8)

    ax1.set_ylabel("Stabilized X (px)")
    ax1.set_title("Stabilized X-Position (camera-compensated)")
    ax1.legend(fontsize=7, ncol=4)
    ax1.grid(True, alpha=0.3)
    ax2.set_ylabel("Stabilized Y (px)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Stabilized Y-Position (lateral movement = stability indicator)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(path), dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def _plot_durations(all_tracks: dict, long_tracks: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    tids = sorted(all_tracks.keys())
    for i, tid in enumerate(tids):
        pts = all_tracks[tid]
        t0 = pts[0]["t"]
        t1 = pts[-1]["t"]
        is_long = tid in long_tracks
        color = "steelblue" if is_long else "lightgray"
        ax.barh(i, t1 - t0, left=t0, height=0.7, color=color,
                edgecolor="gray" if is_long else "none", linewidth=0.5)
    ax.set_yticks(range(len(tids)))
    ax.set_yticklabels([f"T{t}" for t in tids], fontsize=7)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Track Lifetimes (blue = long tracks, {len(long_tracks)} of {len(all_tracks)})")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(str(path), dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    output_dir = Path(video_path).parent
    run_measurement_lap_tracking(video_path, output_dir)


if __name__ == "__main__":
    main()
