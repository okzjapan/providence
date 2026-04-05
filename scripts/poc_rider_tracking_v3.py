"""POC v3: YOLO-based rider tracking with position-based car number assignment.

Uses YOLOv8s with 1280px upscaling to detect riders as "person" objects,
filters by track region, assigns car numbers by spatial ordering in the
first lap, and tracks across frames.

Usage:
    python3 scripts/poc_rider_tracking_v3.py /tmp/trial_video_poc/iizuka_20260404_R01_trial.mp4
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def run_tracking(video_path: str, output_dir: Path) -> None:
    model = YOLO("yolov8s.pt")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total / fps

    print(f"Video: {Path(video_path).name}")
    print(f"  {w_frame}x{h_frame} @ {fps:.1f}fps, {duration:.1f}s, {total} frames")

    trial_start_sec = 3.0
    trial_end_sec = min(35.0, duration - 5)
    min_y = int(h_frame * 0.20)

    sample_interval = int(fps * 1.0)
    annotated_frames: list[tuple[float, np.ndarray]] = []

    all_detections: list[list[tuple[int, int, int, int, int, int, float]]] = []
    frame_times: list[float] = []

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = idx / fps
        if t < trial_start_sec or t > trial_end_sec:
            idx += 1
            continue

        if idx % max(int(fps / 5), 1) == 0:
            results = model(frame, verbose=False, conf=0.1, imgsz=1280, classes=[0])

            dets = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    if cy > min_y:
                        dets.append((cx, cy, x1, y1, x2, y2, conf))

            dets.sort(key=lambda d: d[0])
            all_detections.append(dets)
            frame_times.append(t)

            if idx % sample_interval == 0:
                annotated = frame.copy()
                for rank, (cx, cy, x1, y1, x2, y2, conf) in enumerate(dets):
                    car_num = rank + 1
                    color = _car_color(car_num)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = f"#{car_num}"
                    cv2.putText(annotated, label, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                    cv2.circle(annotated, (cx, cy), 3, color, -1)

                n_det = len(dets)
                cv2.putText(annotated, f"t={t:.1f}s  riders={n_det}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                annotated_frames.append((t, annotated))

        idx += 1

    cap.release()

    print(f"\n  Processed {len(all_detections)} frames")
    det_counts = [len(d) for d in all_detections]
    print(f"  Detections per frame: min={min(det_counts)}, max={max(det_counts)}, "
          f"median={sorted(det_counts)[len(det_counts)//2]}, mean={np.mean(det_counts):.1f}")

    _plot_detection_timeseries(frame_times, all_detections, output_dir / "rider_count_timeseries.png")
    _save_annotated_grid(annotated_frames, output_dir / "yolo_tracking_v3.jpg")
    _plot_rider_positions(frame_times, all_detections, output_dir / "rider_positions.png")


def _car_color(car_num: int) -> tuple[int, int, int]:
    colors = {
        1: (255, 255, 255),
        2: (0, 0, 0),
        3: (0, 0, 255),
        4: (0, 255, 255),
        5: (255, 0, 255),
        6: (0, 255, 0),
        7: (0, 165, 255),
        8: (255, 0, 0),
    }
    return colors.get(car_num, (128, 128, 128))


def _save_annotated_grid(frames: list[tuple[float, np.ndarray]], path: Path) -> None:
    if not frames:
        return
    cols = 4
    h, w = frames[0][1].shape[:2]
    rows = min((len(frames) + cols - 1) // cols, 8)
    max_n = rows * cols
    sel = frames[:max_n]

    sheet = np.zeros((rows * (h + 25), cols * w, 3), dtype=np.uint8)
    for i, (t, img) in enumerate(sel):
        r, c = divmod(i, cols)
        y, x = r * (h + 25), c * w
        sheet[y:y + h, x:x + w] = img
    cv2.imwrite(str(path), sheet)
    print(f"  Saved grid: {path}")


def _plot_detection_timeseries(
    times: list[float],
    detections: list[list],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    counts = [len(d) for d in detections]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, counts, marker=".", markersize=3, linewidth=0.8)
    ax.axhline(y=8, color="red", linestyle="--", alpha=0.5, label="Expected (8 riders)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Detected riders")
    ax.set_title("Rider Detection Count over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(path), dpi=150)
    plt.close()
    print(f"  Saved timeseries: {path}")


def _plot_rider_positions(
    times: list[float],
    detections: list[list],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 6))

    for frame_idx, (t, dets) in enumerate(zip(times, detections)):
        for rank, det in enumerate(dets):
            cx = det[0]
            car_num = rank + 1
            if car_num <= 8:
                ax.scatter(t, cx, c=f"C{car_num-1}", s=8, alpha=0.5)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("X position (pixels)")
    ax.set_title("Rider X-Positions over Time (color = positional rank)")
    ax.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=f"C{i}",
                       markersize=8, label=f"#{i+1}") for i in range(8)]
    ax.legend(handles=handles, loc="upper right", ncol=4, fontsize=8)

    plt.tight_layout()
    plt.savefig(str(path), dpi=150)
    plt.close()
    print(f"  Saved positions: {path}")


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    output_dir = Path(video_path).parent
    run_tracking(video_path, output_dir)


if __name__ == "__main__":
    main()
